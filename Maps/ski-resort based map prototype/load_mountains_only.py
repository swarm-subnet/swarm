import pybullet as p
import pybullet_data
import time
import math
import os
import random
import sys

# Add swarm directory to path (for consistent path handling)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SWARM_DIR = os.path.dirname(BASE_DIR)
ASSETS_ROOT = os.path.join(SWARM_DIR, "assets")
CUSTOM_ASSET_DIR = os.path.join(ASSETS_ROOT, "custom")

# ============================================================================
# CONFIG
# ============================================================================
VILLAGE_SIZE = 100.0
# SCALE_FACTOR = 5.0 # Unused for just mountains

# Colors
SNOW_COLOR = [0.98, 0.98, 1.0, 1]

# Global tracking
spawned_bodies = []
current_seed = 42

# Shape cache for faster spawning - stores (vis_id, col_id) per (path, scale) key
shape_cache = {}

# ============================================================================
# CAMERA CONTROLLER
# ============================================================================
class CameraController:
    def __init__(self):
        self.x = 0
        self.y = -80
        self.z = 60
        self.yaw = 0
        self.pitch = -35
        self.dist = 0.1
        self.speed = 0.1875
        self.mouse_sensitivity = 0.5
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.lmb_held = False
        self.update_camera()

    def update(self, keys=None):
        if keys is None:
            keys = p.getKeyboardEvents()
        mouse = p.getMouseEvents()
        dx = 0
        dy = 0
        MOUSE_MOVE = 1
        MOUSE_BTN = 2
        
        for e in mouse:
            if e[0] == MOUSE_MOVE:
                if self.lmb_held:
                    dx = e[1] - self.last_mouse_x
                    dy = e[2] - self.last_mouse_y
                self.last_mouse_x = e[1]
                self.last_mouse_y = e[2]
            if e[0] == MOUSE_BTN and e[3] == 0:
                self.lmb_held = (e[4] == 3 or e[4] == 1)
                if self.lmb_held:
                    self.last_mouse_x = e[1]
                    self.last_mouse_y = e[2]

        if self.lmb_held and (dx or dy):
            self.yaw -= dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            self.pitch = max(-89, min(89, self.pitch))

        move_speed = self.speed * (3.0 if keys.get(p.B3G_SHIFT, 0) else 1.0)
        
        rad_yaw = math.radians(self.yaw)
        f_x = -math.sin(rad_yaw)
        f_y = math.cos(rad_yaw)
        r_x = math.cos(rad_yaw)
        r_y = math.sin(rad_yaw)

        fwd_input = (1 if keys.get(ord('w'), 0) else 0) - (1 if keys.get(ord('s'), 0) else 0)
        right_input = (1 if keys.get(ord('d'), 0) else 0) - (1 if keys.get(ord('a'), 0) else 0)
        up_input = (1 if keys.get(ord('e'), 0) else 0) - (1 if keys.get(ord('q'), 0) else 0)

        self.x += (f_x * fwd_input + r_x * right_input) * move_speed
        self.y += (f_y * fwd_input + r_y * right_input) * move_speed
        self.z += up_input * move_speed
        
        self.update_camera()
    
    def update_camera(self):
        p.resetDebugVisualizerCamera(self.dist, self.yaw, self.pitch, [self.x, self.y, self.z])


# ============================================================================
# SPAWN FUNCTIONS
# ============================================================================
def spawn_asset(path, x, y, z, rotation_deg, scale, color=None):
    """Spawn a 3D asset at specified position. Uses shape caching for speed."""
    global shape_cache
    
    if not os.path.exists(path):
        print(f"Asset not found: {path}")
        return None
    
    if isinstance(scale, (list, tuple)):
        scale_vec = scale
        scale_key = tuple(scale)
    else:
        scale_vec = [scale, scale, scale]
        scale_key = scale
    
    # Create cache key based on path, scale, and color
    cache_key = (path, scale_key, tuple(color) if color else None)
    
    if cache_key in shape_cache:
        # Reuse cached shapes (FAST!)
        vis_id, col_id = shape_cache[cache_key]
    else:
        # Create new shapes and cache them
        kwargs = {'fileName': path, 'meshScale': scale_vec}
        if color:
            kwargs['rgbaColor'] = color
            kwargs['specularColor'] = [0.1, 0.1, 0.1]
        
        vis_id = p.createVisualShape(p.GEOM_MESH, **kwargs)
        col_id = p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=scale_vec)
        shape_cache[cache_key] = (vis_id, col_id)
    
    base_x_rot = 1.5708  # 90 degrees for Y-up to Z-up
    rad_z = math.radians(rotation_deg)
    orn = p.getQuaternionFromEuler([base_x_rot, 0, rad_z])
    
    body_id = p.createMultiBody(0, col_id, vis_id, [x, y, z], orn)
    spawned_bodies.append(body_id)
    return body_id


# ============================================================================
# ENVIRONMENT GENERATION
# ============================================================================
def clear_village():
    """Remove all spawned objects - INSTANT reset."""
    global spawned_bodies, shape_cache
    spawned_bodies = []
    shape_cache = {}  # Clear cache since resetSimulation invalidates shape IDs
    
    # Reset simulation completely (instant!)
    p.resetSimulation()
    
    # Re-setup basic simulation parameters
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    
    # Re-create ground plane - large enough to cover mountains
    ground_size = VILLAGE_SIZE * 20  # Increased to 20x (2000m)
    ground_half = ground_size / 2
    ground_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[ground_half, ground_half, 0.1],
        rgbaColor=SNOW_COLOR,  # Match hills color exactly
        specularColor=[0, 0, 0]  # No shine
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=ground_vis, basePosition=[0, 0, -0.1])


def spawn_mountains():
    """Spawn the surrounding mountain rings."""
    HILL_DIR = os.path.join(CUSTOM_ASSET_DIR, "mountains")
    PEAK_OBJ = os.path.join(CUSTOM_ASSET_DIR, "mountains", "mountain_peak.obj")
    PEAK_TEX_PATH = os.path.join(CUSTOM_ASSET_DIR, "mountains", "mountain_peak.png")
    
    peak_tex_id = -1
    if os.path.exists(PEAK_TEX_PATH):
        peak_tex_id = p.loadTexture(PEAK_TEX_PATH)
    
    # Discover all hill variants
    hill_candidates = []
    if os.path.exists(HILL_DIR):
        hill_candidates = [f for f in os.listdir(HILL_DIR) if f.endswith('.obj') and "mountain_peak" not in f]
    
    if not hill_candidates:
        print(f"Warning: No hill models found in {HILL_DIR}!")
        return
        
    print(f"Found {len(hill_candidates)} hill variants: {hill_candidates}")
    
    # Safety radius calculation
    radius_inner = 165.0  # User requested fixed radius
    
    print(f"Mountain Inner Radius: {radius_inner:.1f}m")
    
    # INNER RING (6 hills)
    for i in range(6):
        angle = (2 * math.pi / 6) * i
        r = radius_inner + random.uniform(-5, 5)
        x, y = r * math.cos(angle), r * math.sin(angle)
        # Quantize scale to nearest 0.5 for caching
        s = round(random.uniform(10.0, 16.0) * 2) / 2
        
        # Pick random hill variant
        hill_file = random.choice(hill_candidates)
        hill_path = os.path.join(HILL_DIR, hill_file)
        
        # Cache Lookup
        cache_key = (hill_path, s, tuple(SNOW_COLOR))
        if cache_key in shape_cache:
             hill_vis, hill_col = shape_cache[cache_key]
        else:
             hill_vis = p.createVisualShape(p.GEOM_MESH, fileName=hill_path, meshScale=[s, s, s], 
                                            rgbaColor=SNOW_COLOR, specularColor=[0, 0, 0])
             hill_col = p.createCollisionShape(p.GEOM_MESH, fileName=hill_path, meshScale=[s, s, s])
             shape_cache[cache_key] = (hill_vis, hill_col)
        hill_orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(random.uniform(0, 360))])
        
        # Spawn initially at 0 (or anywhere)
        body_id = p.createMultiBody(0, hill_col, hill_vis, [x, y, 0.0], hill_orn)
        
        # Dynamic Ground Snapping:
        min_aabb, max_aabb = p.getAABB(body_id)
        min_z = min_aabb[2]
        
        # Calculate how much we need to shift to get the bottom to Z=0
        z_correction = (0.0 - min_z) - 0.0
        
        # Apply the correction
        p.resetBasePositionAndOrientation(body_id, [x, y, z_correction], hill_orn)
    
    # MIDDLE RING (9 hills)
    for i in range(9):
        # Staggered Angle: Offset by half step (pi/9) 
        angle = (2 * math.pi / 9) * i + (math.pi / 9)
        # Increased radius to 320 to clear the large scale of inner hills
        r = 320.0 + random.uniform(-15, 15)
        x, y = r * math.cos(angle), r * math.sin(angle)
        # Quantize scale to nearest 0.5
        s = round(random.uniform(16.5, 21.0) * 2) / 2
        
        # Pick random hill variant
        hill_file = random.choice(hill_candidates)
        hill_path = os.path.join(HILL_DIR, hill_file)
        
        # Cache Lookup
        cache_key = (hill_path, s, tuple(SNOW_COLOR))
        if cache_key in shape_cache:
             hill_vis, hill_col = shape_cache[cache_key]
        else:
             hill_vis = p.createVisualShape(p.GEOM_MESH, fileName=hill_path, meshScale=[s, s, s],
                                            rgbaColor=SNOW_COLOR, specularColor=[0, 0, 0])
             hill_col = p.createCollisionShape(p.GEOM_MESH, fileName=hill_path, meshScale=[s, s, s])
             shape_cache[cache_key] = (hill_vis, hill_col)
        hill_orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(random.uniform(0, 360))])
        
        # Standard Collision
        body_id = p.createMultiBody(0, hill_col, hill_vis, [x, y, 0.0], hill_orn)
        
        # Dynamic Ground Snapping:
        min_aabb, max_aabb = p.getAABB(body_id)
        min_z = min_aabb[2]
        z_correction = (0.0 - min_z) - 0.0
        p.resetBasePositionAndOrientation(body_id, [x, y, z_correction], hill_orn)
    
    # OUTER RING (10 peaks)
    if os.path.exists(PEAK_OBJ):
        for i in range(10):
            angle = (2 * math.pi / 10) * i
            r = 550.0 + random.uniform(-20, 20)
            x, y = r * math.cos(angle), r * math.sin(angle)
            # Reduced scale by 20% + Quantize
            scale_vals = [126.4, 126.4, 158.4]
            s_var = round(random.uniform(0.9, 1.2), 1) # Round variation to 0.1
            final_scale = [round(v * s_var, 2) for v in scale_vals] # Round final dimensions
            
            # Cache Lookup
            cache_key = (PEAK_OBJ, tuple(final_scale), tuple(SNOW_COLOR))
            if cache_key in shape_cache:
                 vis_id, col_id = shape_cache[cache_key]
            else:
                 vis_id = p.createVisualShape(p.GEOM_MESH, fileName=PEAK_OBJ, meshScale=final_scale,
                                             rgbaColor=SNOW_COLOR, specularColor=[0, 0, 0])
                 col_id = p.createCollisionShape(p.GEOM_MESH, fileName=PEAK_OBJ, meshScale=final_scale)
                 shape_cache[cache_key] = (vis_id, col_id)
            orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(random.uniform(0, 360))])
            
            # Standard Collision
            peak_body = p.createMultiBody(0, col_id, vis_id, [x, y, 0.0], orn)
            
            # Dynamic Ground Snapping 
            min_aabb, max_aabb = p.getAABB(peak_body)
            min_z = min_aabb[2]
            z_correction = (0.0 - min_z) - 10.0
            p.resetBasePositionAndOrientation(peak_body, [x, y, z_correction], orn)

            if peak_tex_id >= 0:
                p.changeVisualShape(peak_body, -1, textureUniqueId=peak_tex_id)


def generate_environment(seed=42):
    """Generate ONLY the environment (Mountains & Ground)."""
    global current_seed
    start_time = time.time()
    current_seed = seed
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"🌄  GENERATING ENVIRONMENT ONLY (Seed: {seed}) 🌄")
    print(f"{'='*60}")

    # Pause rendering
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    
    clear_village()
    
    print("\n🏔️  Spawning mountains...")
    spawn_mountains()
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✨ ENVIRONMENT GENERATION COMPLETE! ✨")
    print(f"{'-'*60}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"⏱️  Total Time : {duration:.2f} seconds")
    print(f"{'='*60}")
    print(f"👉 Press 'R' to Regenerate | '1' for Wireframe")
    print(f"{'='*60}\n")
    
    # Re-enable rendering
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)


# ============================================================================
# MAIN
# ============================================================================
def load_mountains_only():
    """Main entry point."""
    try:
        if not p.isConnected():
            p.connect(p.GUI)
    except Exception:
        pass # Already connected?

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Visual Tweaks
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    # Disable default keyboard shortcuts (prevents W from toggling wireframe)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.resetDebugVisualizerCamera(cameraDistance=100, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0,0,0])
    
    generate_environment(seed=42)
    
    # Interactive Loop
    cam = CameraController()
    
    print("\nSimulation Running...")
    print("CONTROLS:")
    print("  WASD - Move camera")
    print("  E/Q - Up/Down (Vertical)")
    print("  Shift - Fast movement")
    print("  R - Regenerate environment")
    print("  1 - Toggle Wireframe")
    
    wireframe_on = False
    
    while p.isConnected():
        # KEY HANDLERS - Fetch once per frame!
        keys = p.getKeyboardEvents()
        
        # Update camera with same keys
        cam.update(keys)
        
        # 'R' to Regenerate
        if keys.get(ord('r'), 0) & p.KEY_WAS_TRIGGERED:
            new_seed = random.randint(0, 999999)
            generate_environment(seed=new_seed)
            
        # '1' to Toggle Wireframe
        if keys.get(ord('1'), 0) & p.KEY_WAS_TRIGGERED:
            wireframe_on = not wireframe_on
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1 if wireframe_on else 0)
            print(f"Wireframe: {'ON' if wireframe_on else 'OFF'}")
            
        p.stepSimulation()
        time.sleep(1./240.)
    
    p.disconnect()

if __name__ == "__main__":
    load_mountains_only()
