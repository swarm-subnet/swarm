"""
Enhanced Video Recorder for Flight Simulations
Uses original swarm modules directly to stay synchronized with updates
"""

import os
import subprocess
import time
from pathlib import Path
import numpy as np
import pybullet as p
import sys
from pybullet_utils import bullet_client
from tqdm import tqdm
import contextlib
import io

# Add the miner directory to Python path to find swarm modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import original swarm modules
from swarm.core.flying_strategy import flying_strategy
from swarm.validator.task_gen import random_task
from swarm.validator.replay import replay_once
from swarm.validator.reward import flight_reward
from swarm.protocol import MapTask, FlightPlan, ValidationResult
from swarm.validator.forward import SIM_DT, HORIZON_SEC
from swarm.constants import CAM_HZ
from swarm.core.drone import track_drone

@contextlib.contextmanager
def suppress_cpp_output():
    """Context manager that redirects low-level C stdout and stderr to /dev/null."""
    if sys.platform.startswith('win'):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
        return

    # For Linux/macOS
    original_stdout_fd = sys.stdout.fileno()
    original_stderr_fd = sys.stderr.fileno()
    
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stdout_fd = os.dup(original_stdout_fd)
    saved_stderr_fd = os.dup(original_stderr_fd)

    try:
        os.dup2(devnull_fd, original_stdout_fd)
        os.dup2(devnull_fd, original_stderr_fd)
        yield
    finally:
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)


def silence_at_exit():
    """
    Redirects stdout and stderr to /dev/null to prevent final C++ cleanup
    messages from PyBullet from being printed upon script exit.
    """
    if sys.platform.startswith('win'):
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, sys.stdout.fileno())
        os.dup2(devnull_fd, sys.stderr.fileno())
    except io.UnsupportedOperation:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')


class EnhancedVideoRecorder:
    """
    Video recorder that uses original swarm modules directly
    to ensure compatibility with future updates.
    """
    def __init__(self, output_folder="flight_videos"):
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.ffmpeg_process = None
        self.display_process = None
        self.display = ":99"
        self.current_video_name = None
        
        self._check_requirements()
    
    def _check_requirements(self):
        """Check if all required tools are available"""
        required_tools = ["Xvfb", "ffmpeg"]
        missing_tools = []
        
        for tool in required_tools:
            try:
                subprocess.run([tool, "--help"], capture_output=True, timeout=5)
            except (subprocess.TimeoutExpired, FileNotFoundError):
                missing_tools.append(tool)
        
        if missing_tools:
            print(f"Warning: Missing required tools: {', '.join(missing_tools)}")
            print("Install with: sudo apt-get install xvfb ffmpeg")
        else:
            print("All required tools available")
        
    def get_next_video_number(self):
        """Get the next available video number for naming"""
        existing_videos = self.output_folder.glob("video_*.mp4")
        max_num = 0
        for video in existing_videos:
            try:
                num = int(video.stem.split('_')[1])
                if num > max_num:
                    max_num = num
            except (IndexError, ValueError):
                continue
        return max_num + 1
        
    def start_virtual_display(self):
        """Start virtual display for headless recording"""
        print(f"Starting virtual display {self.display}")
        self.display_process = subprocess.Popen(
            ["Xvfb", self.display, "-screen", "0", "1024x768x24", "-ac", "+extension", "GLX"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        time.sleep(3)
        os.environ["DISPLAY"] = self.display
        
        # Verify display is working
        try:
            result = subprocess.run(["xdpyinfo", "-display", self.display], 
                                  capture_output=True, timeout=5)
            if result.returncode == 0:
                print(f"Virtual display verified and ready on {self.display}")
            else:
                print("Virtual display may have issues, but continuing...")
        except subprocess.TimeoutExpired:
            print("Display verification timeout, but continuing...")
        except FileNotFoundError:
            print("xdpyinfo not found, but display should be ready...")

    def stop_virtual_display(self):
        """Stop virtual display"""
        if self.display_process:
            print("Stopping virtual display...")
            self.display_process.terminate()
            self.display_process.wait()
            print("Virtual display stopped")
            self.display_process = None

    def start_recording(self, video_name):
        """Start video recording with ffmpeg"""
        video_path = self.output_folder / video_name
        command = [
            "ffmpeg", "-y", "-f", "x11grab", "-s", "1024x768",
            "-r", "25", "-i", f"{self.display}.0+0,0",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "23", "-pix_fmt", "yuv420p", 
            "-loglevel", "error",
            str(video_path),
        ]
        print(f"Starting recording: {video_name}")
        self.ffmpeg_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        time.sleep(2)
        
        if self.ffmpeg_process.poll() is None:
            print("Recording started successfully")
            return True
        else:
            print("FFmpeg failed to start!")
            print("Check if virtual display is working and ffmpeg has correct parameters")
            self.ffmpeg_process = None
            return False
        
    def stop_recording(self):
        """Stop video recording"""
        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            print("Stopping recording...")
            
            try:
                self.ffmpeg_process.stdin.write(b'q')
                self.ffmpeg_process.stdin.flush()
            except:
                pass
            
            try:
                self.ffmpeg_process.wait(timeout=10)
                print("Recording stopped gracefully")
            except subprocess.TimeoutExpired:
                print("Force stopping recording...")
                self.ffmpeg_process.terminate()
                try:
                    self.ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.ffmpeg_process.kill()
                print("Recording force stopped")
        self.ffmpeg_process = None

    def run_simulation(self):
        """
        Run simulation with robust loop for smooth video recording
        """
        print("Initializing simulation and preparing for recording...")
        env = None
        try:
            # Setup simulation
            task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC)
            result = flying_strategy(task, gui=False)
            cmds = result[0] if isinstance(result, tuple) else result

            if not cmds:
                print("No valid flight plan generated! Aborting.")
                return

            plan = FlightPlan(commands=cmds, sha256="")

            from swarm.core.env_builder import _TAO_TEX_ID
            _TAO_TEX_ID.clear()

            from swarm.utils.env_factory import make_env
            with suppress_cpp_output():
                env = make_env(task, gui=True, raw_rpm=True)
            cli = env.getPyBulletClient()
            
            # Initialize physics
            for _ in range(50):
                p.stepSimulation(physicsClientId=cli)
                time.sleep(task.sim_dt)

            with suppress_cpp_output():
                start_pos = np.asarray(task.start, dtype=float)
                start_orientation = p.getQuaternionFromEuler([0, 0, 0])
                p.resetBasePositionAndOrientation(
                    env.DRONE_IDS[0],
                    start_pos,
                    start_orientation,
                    physicsClientId=cli,
                )
                p.resetBaseVelocity(env.DRONE_IDS[0], linearVelocity=[0,0,0], angularVelocity=[0,0,0], physicsClientId=cli)
            
            time.sleep(2)

            print("Starting recording...")
            if not self.start_recording(self.current_video_name):
                print("Failed to start recording! Aborting.")
                return
            print("Flight in progress...")

            # Simulation loop
            from swarm.validator.replay import _plan_to_table
            from swarm.constants import PROP_EFF, LANDING_PLATFORM_RADIUS as _PR, STABLE_LANDING_SEC
            
            last_t = plan.commands[-1].t
            max_steps = int(round(last_t / task.sim_dt)) + 1
            rpm_table = _plan_to_table(plan.commands, max_steps, task.sim_dt)
            
            frames_per_cam = max(1, int(round(1.0 / (task.sim_dt * CAM_HZ))))
            energy = 0.0
            success = False
            collided = False
            stable_landing_time = 0.0
            goal = np.asarray(task.goal, dtype=float)
            drone_id = env.DRONE_IDS[0]
            t_sim = 0.0

            # Retrieve avian simulation system if enabled
            bird_system = getattr(env, '_bird_system', None)
            
            # Retrieve atmospheric wind simulation system if enabled
            wind_system = getattr(env, '_wind_system', None)

            for k in tqdm(range(max_steps), desc="Simulating Flight", unit="step"):
                t_sim = k * task.sim_dt
                rpm_vec = rpm_table[k]
                obs, *_ = env.step(rpm_vec[None, :])
                pos = obs[0, :3]
                
                # Update avian behavioral states if system present
                if bird_system:
                    bird_system.update(task.sim_dt)
                
                # Update atmospheric wind simulation if system present
                if wind_system:
                    wind_system.update(task.sim_dt)
                    
                    # Apply wind force to drone
                    wind_force = wind_system.get_wind_force(pos)
                    if np.linalg.norm(wind_force) > 0.01:  # Only apply if force is significant
                        # Apply wind force as external force to drone
                        p.applyExternalForce(
                            drone_id,
                            -1,  # Link index (-1 for base)
                            wind_force.tolist(),
                            pos.tolist(),
                            p.WORLD_FRAME,
                            physicsClientId=cli
                        )

                if k % frames_per_cam == 0:
                    track_drone(cli, drone_id)

                energy += (np.square(rpm_vec).sum() * env.KF / PROP_EFF) * task.sim_dt

                if not collided:
                    contacts = p.getContactPoints(bodyA=drone_id, physicsClientId=cli)
                    if contacts:
                        bird_collision = False
                        allowed = True
                        for cp in contacts:
                            body_b = cp[2]
                            
                            # Detect avian collision events with drone
                            if bird_system and body_b in bird_system.bird_ids:
                                bird_collision = True
                                bird_system.handle_bird_collision(body_b)
                                break
                            
                            cpos = cp[5]
                            if isinstance(cpos, (list, tuple)) and len(cpos) >= 3:
                                cx, cy, cz = cpos[:3]
                                horiz = np.linalg.norm([cx - goal[0], cy - goal[1]])
                                vert = abs(cz - goal[2])
                                if horiz < _PR + 0.05 and vert < 0.3:
                                    continue
                            allowed = False
                            break
                        
                        if bird_collision or not allowed:
                            collided = True
                            success = False
                            break
                
                horizontal_distance = np.linalg.norm(pos[:2] - goal[:2])
                vertical_distance = abs(pos[2] - goal[2])
                tao_logo_radius = _PR * 0.8 * 1.06
                
                on_tao_logo = (horizontal_distance < tao_logo_radius and
                               vertical_distance < 0.3 and
                               pos[2] >= goal[2] - 0.1)
                
                if on_tao_logo:
                    stable_landing_time += task.sim_dt
                    if stable_landing_time >= STABLE_LANDING_SEC:
                        success = True
                        break
                else:
                    stable_landing_time = 0.0
                
                time.sleep(task.sim_dt)
            
            if collided:
                success = False

            score = flight_reward(success, t_sim, energy, task.horizon)
            
            # Flight report
            success_str = "Success" if success else "Failed"
            separator = "=" * 56
            
            def format_line(label, value):
                return f"| {label:<28} : {str(value):<23} |"

            print(f"\n+{separator}+")
            print(f"|{'FLIGHT REPORT'.center(56)}|")
            print(f"+{separator}+")
            print(format_line("Map Seed", task.map_seed))
            print(format_line("Successful Landing", success_str))
            print(format_line("Flight Time (s)", f"{t_sim:.2f} / {task.horizon:.1f}"))
            print(format_line("Energy Consumed", f"{energy:.2f}"))
            print(format_line("Final Score", f"{score:.4f}"))
            print(f"+{separator}+\n")
            
            print("Recording final scene for 3 seconds...")
            time.sleep(3)

        except Exception:
            print("\nAn error occurred during the simulation!")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_recording()
            if env:
                with suppress_cpp_output():
                    env.close()
                    time.sleep(0.5)
                print("Simulation finished.")

    def record_single_flight(self):
        """Record a single flight video"""
        video_number = self.get_next_video_number()
        video_name = f"video_{video_number}.mp4"
        
        try:
            print(f"=== Recording Flight Video #{video_number} ===")
            self.start_virtual_display()
            self.current_video_name = video_name
            self.run_simulation()
            
            print("Simulation finished, finalizing video...")
            time.sleep(2)
                
        except Exception as e:
            print(f"Recording error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_recording()
            self.stop_virtual_display()

        video_path = self.output_folder / video_name
        time.sleep(1)
        
        if video_path.exists():
            size_bytes = video_path.stat().st_size
            size_mb = size_bytes / (1024 * 1024)
            
            if size_bytes > 100000:  # 100KB minimum
                print(f"Video saved successfully: {video_name} ({size_mb:.1f} MB)")
                return True
            else:
                print(f"Video file too small: {video_name} ({size_bytes} bytes)")
                return False
        else:
            print(f"Video file not found: {video_name}")
            return False


def main():
    """Main function to run the video recorder"""
    print("=== Flight Video Recorder ===")
    recorder = EnhancedVideoRecorder()

    try:
        user_input = input("How many videos do you want to record? [default 1]: ").strip()
        num_videos = int(user_input) if user_input else 1
    except (ValueError, EOFError):
        print("Invalid input. Defaulting to 1 video.")
        num_videos = 1

    successes = 0
    for i in range(1, num_videos + 1):
        print(f"\nStarting recording {i}/{num_videos}...")
        if recorder.record_single_flight():
            successes += 1
        else:
            print(f"Recording {i} failed.")

    print(f"\nFinished. Successfully recorded {successes}/{num_videos} video(s).")


if __name__ == "__main__":
    main()
    silence_at_exit() 