#!/usr/bin/env python3
import sys
import os
import zipfile
import shutil
import subprocess
from pathlib import Path

swarm_path = str(Path(__file__).resolve().parent.parent.parent)
if swarm_path not in sys.path:
    sys.path.insert(0, swarm_path)

def main():
    if len(sys.argv) != 2:
        print("Usage: container_launcher.py <model.zip>", file=sys.stderr)
        sys.exit(1)
    
    model_path = Path(sys.argv[1])
    submission_dir = Path("/tmp") / f"submission_{os.getpid()}"
    submission_dir.mkdir(exist_ok=True)
    
    try:
        template_dir = Path(swarm_path) / "swarm" / "submission_template"
        
        with zipfile.ZipFile(model_path, 'r') as zf:
            zf.extractall(submission_dir)
        
        shutil.copy(template_dir / "agent.capnp", submission_dir)
        shutil.copy(template_dir / "agent_server.py", submission_dir)
        shutil.copy(template_dir / "main.py", submission_dir)
        
        requirements_file = submission_dir / "requirements.txt"
        if requirements_file.exists():
            print("📦 Installing miner's requirements.txt...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements_file)],
                capture_output=True,
                timeout=180,
                check=False
            )
            if result.returncode == 0:
                print("✅ Requirements installed successfully")
            else:
                print(f"⚠️ Requirements installation had issues: {result.stderr.decode()[:200]}")
        
        os.chdir(submission_dir)
        subprocess.run([sys.executable, "main.py"], check=False)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        try:
            shutil.rmtree(submission_dir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
