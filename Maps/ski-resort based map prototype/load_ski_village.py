import pybullet as p
import pybullet_data
import time
import math
import os
import random
import sys

# Add swarm directory to path to import city_gen
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SWARM_DIR = BASE_DIR
if SWARM_DIR not in sys.path:
    sys.path.append(SWARM_DIR)

import city_gen

# ============================================================================
# CONFIG
# ============================================================================
VILLAGE_SIZE = 100.0
ROAD_WIDTH = 6.0
SCALE_FACTOR = 5.0  # Scale for buildings and assets

# Paths
ASSETS_ROOT = os.path.join(os.path.dirname(SWARM_DIR), "assets")
KENNEY_ASSET_DIR = os.path.join(ASSETS_ROOT, "kenney")
CUSTOM_ASSET_DIR = os.path.join(ASSETS_ROOT, "custom")

ROAD_ASSET_DIR = os.path.join(KENNEY_ASSET_DIR, "kenney_roads", "Models", "OBJ format")
SUBURBAN_ASSET_DIR = os.path.join(KENNEY_ASSET_DIR, "kenney_suburban")
HOLIDAY_ASSET_DIR = os.path.join(KENNEY_ASSET_DIR, "holiday")
CAR_ASSET_DIR = os.path.join(KENNEY_ASSET_DIR, "kenney_car-kit", "Models", "OBJ format")
LANTERN_PATH = os.path.join(HOLIDAY_ASSET_DIR, "lantern.obj")
LANTERN_ROOF_PATH = os.path.join(CUSTOM_ASSET_DIR, "buildings", "SnowRoofs", "lantern_roof.obj")

# Car models available (from kenney_car-kit)
CAR_ASSETS = ['sedan.obj', 'hatchback-sports.obj', 'taxi.obj', 'police.obj', 'suv.obj', 'van.obj', 'truck.obj']

# Colors
SNOW_COLOR = [0.98, 0.98, 1.0, 1]
ROAD_COLOR = [0.35, 0.35, 0.38, 1]

# Road Assets
ROAD_ASSETS = {
    'intersection': 'road-crossroad.obj',
    'corner': 'road-bend.obj',
    't_junction': 'road-intersection.obj',
    'straight_v': 'road-straight.obj',
    'straight_h': 'road-straight.obj',
    'crossing': 'road-crossing.obj',
    'roundabout': 'road-roundabout.obj',
    'dead_end': 'road-end-round.obj',
}

# New path for winter buildings
WINTER_ASSET_DIR = os.path.join(CUSTOM_ASSET_DIR, "buildings")

# House Specs with actual dimensions (width, depth at scale 1.0)
# Format: (filename, width, depth)
# House Specs - Full List restored (All set to Brown now)
HOUSE_SPECS = [
    ('building-type-a.obj', 6.50, 5.14),
    ('building-type-b.obj', 9.14, 5.70),
    ('building-type-c.obj', 6.44, 5.14),
    ('building-type-d.obj', 8.79, 5.14),
    ('building-type-e.obj', 6.50, 5.14),
    ('building-type-f.obj', 7.14, 7.03),
    ('building-type-g.obj', 7.25, 5.89),
    ('building-type-h.obj', 6.50, 4.58),
    ('building-type-i.obj', 6.44, 5.14),
    ('building-type-j.obj', 6.85, 4.58),
    ('building-type-k.obj', 4.61, 5.10),
    ('building-type-l.obj', 5.20, 5.12),
    ('building-type-m.obj', 7.14, 7.14),
    ('building-type-n.obj', 8.93, 6.89),
    ('building-type-o.obj', 6.35, 5.14),
    ('building-type-p.obj', 6.20, 4.95),
    ('building-type-q.obj', 6.20, 4.43),
    ('building-type-r.obj', 5.16, 5.12),
    ('building-type-s.obj', 7.03, 5.44),
    ('building-type-t.obj', 6.60, 7.05),
    ('building-type-u.obj', 7.14, 5.44),
]


# Global tracking
spawned_bodies = []
placed_buildings = []  # Track placed building positions for collision avoidance
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
        self.speed = 0.1875  # Quarter speed (was 0.375)
        self.mouse_sensitivity = 0.5
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.lmb_held = False
        self.update_camera()

    def update(self):
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
        # Use average for cache key if needed, or just tuple
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

# Cache for OBJ bounds to avoid re-parsing files
obj_bounds_cache = {}

def get_obj_min_y(path):
    """Parse OBJ file to find minimum Y (height) value. Caches result."""
    global obj_bounds_cache
    if path in obj_bounds_cache:
        return obj_bounds_cache[path]
    
    min_y = float('inf')
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '): # Vertex line
                    parts = line.split()
                    if len(parts) >= 3:
                        # OBJ is Y-up usually
                        y = float(parts[2])
                        if y < min_y:
                            min_y = y
    except Exception as e:
        print(f"Error parsing OBJ {path}: {e}")
        return 0.0
        
    obj_bounds_cache[path] = min_y
    return min_y


def spawn_road_tile(tile, road_width, offset_x, offset_y, z_height=0.08):
    """Spawn a road tile based on city_gen.RoadTile data. Uses shape caching."""
    global shape_cache
    
    asset_name = ROAD_ASSETS.get(tile.type)
    if not asset_name:
        return
    
    path = os.path.join(ROAD_ASSET_DIR, asset_name)
    if not os.path.exists(path):
        return
    
    scale = road_width
    cache_key = (path, scale, tuple(ROAD_COLOR))
    
    if cache_key in shape_cache:
        vis_id, col_id = shape_cache[cache_key]
    else:
        vis_id = p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=[scale, scale, scale], 
                                     rgbaColor=ROAD_COLOR, specularColor=[0.1, 0.1, 0.1])
        col_id = p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=[scale, scale, scale])
        shape_cache[cache_key] = (vis_id, col_id)
    
    base_x_rot = 1.5708
    rad_z = math.radians(tile.rotation) + math.radians(90)
    orn = p.getQuaternionFromEuler([base_x_rot, 0, rad_z])
    
    final_x = tile.x - offset_x + (road_width / 2)
    final_y = tile.y - offset_y + (road_width / 2)
    
    body_id = p.createMultiBody(0, col_id, vis_id, [final_x, final_y, z_height], orn)
    spawned_bodies.append(body_id)


def spawn_house(x, y, rotation_deg=0, scale=None):
    """Spawn a complete house from kenney_suburban."""
    if scale is None:
        scale = SCALE_FACTOR * 0.8  # Slightly smaller for village
    
    house_file = random.choice(HOUSE_ASSETS)
    path = os.path.join(SUBURBAN_ASSET_DIR, house_file)
    
    return spawn_asset(path, x, y, 0, rotation_deg, scale)


def spawn_lantern(x, y, rotation_deg=0):
    """Spawn holiday lantern."""
    # Use non-uniform scale to keep height while reducing width.
    lantern_scale = [1.05, 1.62, 1.05] 
    lantern_z = 0.15
    spawn_asset(LANTERN_PATH, x, y, lantern_z, rotation_deg, lantern_scale)
    
    # Spawn lantern roof if exists
    if os.path.exists(LANTERN_ROOF_PATH):
        # Match scale for roof
        spawn_asset(LANTERN_ROOF_PATH, x, y, lantern_z + 0.01, rotation_deg, lantern_scale)


def spawn_car(x, y, rotation_deg=0):
    """Spawn a random car on the road."""
    car_file = random.choice(CAR_ASSETS)
    path = os.path.join(CAR_ASSET_DIR, car_file)
    scale = 0.58  # Reduced by another 20% (approx 0.72 * 0.8)
    return spawn_asset(path, x, y, 0.1, rotation_deg, scale)


# ============================================================================
# VILLAGE GENERATION
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
    ground_size = VILLAGE_SIZE * 20  # Increased to 20x (2000m) to cover distant mountains
    ground_half = ground_size / 2
    ground_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[ground_half, ground_half, 0.1],
        rgbaColor=SNOW_COLOR,  # Match hills color exactly
        specularColor=[0, 0, 0]  # No shine, same as hills
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=ground_vis, basePosition=[0, 0, -0.1])


def generate_village(seed=42):
    """Generate the complete village with roads, buildings, and lamps."""
    global current_seed
    start_time = time.time() # Start Timer
    current_seed = seed
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"❄️  GENERATING SNOW VILLAGE (Seed: {seed}) ❄️")
    print(f"{'='*60}")
    
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0) # Pause rendering
    
    # Clear existing
    clear_village()
    
    # 1. Spawn mountains/hills first (they're the backdrop)
    print("\n🏔️  Spawning mountains...")
    spawn_mountains()
    
    rng = city_gen.SeededRNG(seed)
    
    # 1. Generate Road Network
    print("🛣️  Generating roads...")
    # Smaller blocks = more roads! min_spacing=8, target_area=100
    v_pos = city_gen.generate_road_positions(rng, min_spacing=8, target_area=100, tile_size=ROAD_WIDTH, map_size=VILLAGE_SIZE)
    h_pos = city_gen.generate_road_positions(rng, min_spacing=8, target_area=100, tile_size=ROAD_WIDTH, map_size=VILLAGE_SIZE)
    
    # max_block_size=18 for smaller blocks, difficulty=0 to reduce crossings
    tiles, blocked, rm_h, rm_v, add_h, add_v = city_gen.generate_road_tiles(
        v_pos, h_pos, rng, max_block_size=18, tile_size=ROAD_WIDTH, difficulty=0, map_size=VILLAGE_SIZE
    )
    
    # Extract blocks for building placement
    min_area = 50
    blocks = city_gen.extract_blocks(v_pos, h_pos, min_area, rm_h, rm_v, add_h, add_v, effective_tile_size=ROAD_WIDTH)
    
    print(f"Roads: {len(tiles)} tiles")
    print(f"Blocks: {len(blocks)}")
    
    # Spawn roads and collect roundabout positions
    offset = VILLAGE_SIZE / 2
    roundabout_centers = []  # Store roundabout positions for exclusion
    
    for tile in tiles:
        if tile.type == 'roundabout_arm':
            continue
        spawn_road_tile(tile, ROAD_WIDTH, offset, offset)
        
        # Collect roundabout centers (in village-centered coords)
        if tile.type == 'roundabout':
            ra_cx = tile.x - offset + ROAD_WIDTH / 2
            ra_cy = tile.y - offset + ROAD_WIDTH / 2
            roundabout_centers.append((ra_cx, ra_cy))
    
    print(f"Roundabouts found: {len(roundabout_centers)}")
    
    # 2. Place Buildings - Proper Perimeter Filling (Like City Generator)
    print("🏠 Placing buildings...")
    building_count = 0
    roof_count = 0
    
    HOUSE_SCALE = 2.0
    GAP = 0.3  # Small gap between houses
    
    def get_house_for_depth(target_depth):
        """Get a house that fits the target depth."""
        # Sort by how close the depth matches
        candidates = []
        for spec in HOUSE_SPECS:
            filename, raw_w, raw_d = spec
            scaled_d = raw_d * HOUSE_SCALE / 5.0
            candidates.append((filename, raw_w * HOUSE_SCALE / 5.0, scaled_d))
        
        # Filter houses that fit
        fitting = [c for c in candidates if c[2] <= target_depth + 0.5]
        if fitting:
            return random.choice(fitting)
        return random.choice(candidates)
    
    def spawn_house_at(x, y, rotation, filename):
        """Spawn a house, checking roundabout clearance first."""
        nonlocal building_count, roof_count
        
        # Check if too close to any roundabout
        for (ra_x, ra_y) in roundabout_centers:
            dist = math.sqrt((x - ra_x)**2 + (y - ra_y)**2)
            if dist < 10.0:  # 10m clearance from roundabout center
                return  # Skip this house
        
        winter_path = os.path.join(WINTER_ASSET_DIR, filename)
        suburban_path = os.path.join(SUBURBAN_ASSET_DIR, filename)
        
        if os.path.exists(winter_path):
            path = winter_path
            
            # Keep house facades aligned with road-facing generation rotation.
            if rotation % 180 == 0:
                final_rot = rotation + 180
            else:
                final_rot = rotation
            
            spawn_asset(path, x, y, 0, final_rot, HOUSE_SCALE)
            
            # Spawn Roof if exists
            roof_name = filename.replace(".obj", "_roof.obj")
            roof_path = os.path.join(WINTER_ASSET_DIR, "SnowRoofs", roof_name)
            
            if os.path.exists(roof_path):
                spawn_asset(roof_path, x, y, 0.01, final_rot, HOUSE_SCALE)
                roof_count += 1
                
        else:
            path = suburban_path
            # Apply same conditional for consistency
            if rotation % 180 == 0:
                final_rot = rotation + 180
            else:
                final_rot = rotation
            spawn_asset(path, x, y, 0, final_rot, HOUSE_SCALE)
            
        building_count += 1
    
    # Process each block
    for block in blocks:
        bx, by, bw, bh = block.rect.x, block.rect.y, block.rect.w, block.rect.h
        
        # Skip only extremely small blocks (reduced from 8 to 5)
        if bw < 5 or bh < 5:
            continue
        
        road_margin = 1.5  # Reduced distance from road (was 2.0)
        row_depth = 3.0    # Depth of building row
        corner_reserve = row_depth + 0.5  # Reserve space for corners (avoid overlap)
        
        # Block edges in centered coords
        left = (bx - offset) + road_margin
        right = (bx - offset) + bw - road_margin
        bottom = (by - offset) + road_margin
        top = (by - offset) + bh - road_margin
        
        row_width = right - left
        col_height = top - bottom
        
        # Helper: Pre-calculate houses that fit, then center them
        def fill_row_centered(start, end, fixed_pos, rotation, is_vertical=False):
            """Fill a row/column with houses, centered in available space."""
            available = end - start
            if available < 2.0:  # Too small for any house
                return
            
            # First pass: collect houses that fit
            houses = []
            total_width = 0
            attempts = 0
            while total_width < available and attempts < 50:
                attempts += 1
                filename, w, d = get_house_for_depth(row_depth)
                house_size = w  # Width (or height when vertical)
                if total_width + house_size + (GAP if houses else 0) > available:
                    break
                houses.append((filename, house_size))
                total_width += house_size + (GAP if houses else 0)
            
            if not houses:
                return
            
            # Calculate total used space
            used_space = sum(h[1] for h in houses) + GAP * (len(houses) - 1)
            
            # Center offset
            start_offset = (available - used_space) / 2
            
            # Place houses
            curr_pos = start + start_offset
            for filename, size in houses:
                center_pos = curr_pos + size / 2
                if is_vertical:
                    spawn_house_at(fixed_pos, center_pos, rotation, filename)
                else:
                    spawn_house_at(center_pos, fixed_pos, rotation, filename)
                curr_pos += size + GAP
        
        # Check block size to determine placement strategy
        # Use 2 rows only if block can fit them (need at least 10m for both rows)
        is_small_block = (bw < 10 or bh < 10)
        
        if is_small_block:
            # SMALL BLOCK: Single centered row (horizontal or vertical based on shape)
            if bw >= bh:
                # Wider - single horizontal row in middle
                y_pos = (bottom + top) / 2
                fill_row_centered(left, right, y_pos, 0, is_vertical=False)
            else:
                # Taller - single vertical column in middle
                x_pos = (left + right) / 2
                fill_row_centered(bottom, top, x_pos, 90, is_vertical=True)
        else:
            # NORMAL BLOCK: Full perimeter
            # === BOTTOM ROW (Facing Up - 0 degrees) - CENTERED ===
            y_pos = bottom + row_depth / 2
            fill_row_centered(left, right, y_pos, 0, is_vertical=False)
            
            # === TOP ROW (Facing Down - 180 degrees) - CENTERED ===
            y_pos = top - row_depth / 2
            fill_row_centered(left, right, y_pos, 180, is_vertical=False)
            
            # === LEFT COLUMN (Facing Right - 90 degrees) - CENTERED ===
            x_pos = left + row_depth / 2
            col_bottom = bottom + corner_reserve
            col_top = top - corner_reserve
            fill_row_centered(col_bottom, col_top, x_pos, 90, is_vertical=True)
            
            # === RIGHT COLUMN (Facing Left - 270 degrees) - CENTERED ===
            x_pos = right - row_depth / 2
            fill_row_centered(col_bottom, col_top, x_pos, 270, is_vertical=True)
    
    # 3. Place Lanterns (Holiday style)
    print("🏮 Placing lanterns...")
    lamp_count = 0
    lamp_freq = 4  # Every 4th road tile (reduced 20% more)
    
    half_tile = ROAD_WIDTH / 2
    lamp_offset = half_tile - 0.3  # Offset from road center to edge
    
    road_idx = 0
    for tile in tiles:
        if tile.type == 'intersection':
            continue
        if tile.type == 'roundabout_arm':
            continue
            
        road_idx += 1
        if road_idx % lamp_freq != 0:
            continue
        
        # Center of tile
        cx = tile.x - offset + half_tile
        cy = tile.y - offset + half_tile
        
        if tile.type == 'straight_v':
            # Vertical road: lanterns on left and right
            spawn_lantern(cx - lamp_offset, cy, 180)
            spawn_lantern(cx + lamp_offset, cy, 0)
            lamp_count += 2
        elif tile.type == 'straight_h':
            # Horizontal road: lanterns on top and bottom
            spawn_lantern(cx, cy - lamp_offset, 270)
            spawn_lantern(cx, cy + lamp_offset, 90)
            lamp_count += 2
    
    # 4. Spawn Cars on Roads
    print("🚗 Spawning cars...")
    car_count = 0
    lane_offset = ROAD_WIDTH * 0.18  # Lane offset from center
    
    for tile in tiles:
        if tile.type in ['straight_v', 'straight_h']:
            # 35% chance to spawn a car on this tile
            if random.random() < 0.35:
                cx = tile.x - offset + ROAD_WIDTH / 2
                cy = tile.y - offset + ROAD_WIDTH / 2
                
                direction = random.choice([1, -1])  # Random lane
                
                if tile.type == 'straight_v':
                    # Vertical road
                    if direction == 1:  # North lane (right side = east)
                        car_x = cx + lane_offset
                        car_rot = 180  # Face south
                    else:  # South lane (right side = west)
                        car_x = cx - lane_offset
                        car_rot = 0  # Face north
                    car_y = cy
                else:
                    # Horizontal road
                    if direction == 1:  # East lane (right side = south)
                        car_y = cy - lane_offset
                        car_rot = 90  # Face east
                    else:  # West lane (right side = north)
                        car_y = cy + lane_offset
                        car_rot = 270  # Face west
                    car_x = cx
                
                spawn_car(car_x, car_y, car_rot)
                car_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✨ VILLAGE GENERATION COMPLETE! ✨")
    print(f"{'-'*60}")
    print(f"  🛣️  Road Tiles : {len(tiles)}")
    print(f"  🏠 Buildings  : {building_count}")
    if roof_count != building_count:
        print(f"  ⚠️  Roofs      : {roof_count} (MISSING: {building_count - roof_count})")
    print(f"  🏮 Lanterns   : {lamp_count}")
    print(f"  🚗 Cars       : {car_count}")
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"{'-'*60}")
    print(f"⏱️  Total Time : {duration:.2f} seconds")
    print(f"{'='*60}")
    print(f"👉 Press 'R' to Regenerate | '1' for Wireframe")
    print(f"{'='*60}\n")
    
    # OPTIMIZATION: Re-enable rendering and shadows
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)


def spawn_mountains():
    """Spawn the surrounding mountain rings."""
    global shape_cache # Use global cache
    
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
    radius_inner = 165.0  # User requested fixed radius (Increased +10m)
    
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
        # Get the bounding box to find the lowest point of the mesh
        min_aabb, max_aabb = p.getAABB(body_id)
        min_z = min_aabb[2]
        
        z_correction = (0.0 - min_z) - 0.0
        
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
        
        body_id = p.createMultiBody(0, hill_col, hill_vis, [x, y, 0.0], hill_orn)
        
        # Dynamic Ground Snapping:
        min_aabb, max_aabb = p.getAABB(body_id)
        min_z = min_aabb[2]
        z_correction = (0.0 - min_z) - 0.0
        p.resetBasePositionAndOrientation(body_id, [x, y, z_correction], hill_orn)
        # Texture enabled
    
    # OUTER RING (10 peaks)
    if os.path.exists(PEAK_OBJ):
        for i in range(10):
            angle = (2 * math.pi / 10) * i
            # Increased radius to 550 to clear the middle ring
            r = 550.0 + random.uniform(-20, 20)
            x, y = r * math.cos(angle), r * math.sin(angle)
            # Reduced scale by 20% as requested (158*0.8=126.4, 198*0.8=158.4)
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
            
            peak_body = p.createMultiBody(0, col_id, vis_id, [x, y, 0.0], orn)
            
            # Dynamic Ground Snapping for Peaks too
            min_aabb, max_aabb = p.getAABB(peak_body)
            min_z = min_aabb[2]
            z_correction = (0.0 - min_z) - 10.0
            p.resetBasePositionAndOrientation(peak_body, [x, y, z_correction], orn)

            if peak_tex_id >= 0:
                p.changeVisualShape(peak_body, -1, textureUniqueId=peak_tex_id)


# ============================================================================
# MAIN
# ============================================================================
def load_ski_village():
    """Main entry point - loads the ski village simulation."""
    global current_seed
    
    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0) # Start disabled
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)  # Disable 'w' wireframe toggle
    
    # Ground (initial - will be recreated on regenerate)
    ground_half_extents = [500, 500, 1]
    ground_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=ground_half_extents)
    ground_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=ground_half_extents, 
                                     rgbaColor=SNOW_COLOR, specularColor=[0, 0, 0])
    p.createMultiBody(0, ground_col, ground_vis, [0, 0, -2])
    
    # Initial village generation (includes mountains)
    generate_village(seed=42)
    
    # Main loop
    cam = CameraController()
    r_pressed = False
    wireframe_pressed = False
    wireframe_enabled = False
    
    print("\n" + "="*50)
    print("CONTROLS:")
    print("  WASD - Move camera")
    print("  E/Q - Up/Down")
    print("  Shift - Fast movement")
    print("  R - Regenerate village")
    print("="*50 + "\n")
    
    while p.isConnected():
        keys = p.getKeyboardEvents()
        
        # R key = Regenerate
        if keys.get(ord('r'), 0) == 1:
            if not r_pressed:
                r_pressed = True
                new_seed = int(time.time())
                generate_village(seed=new_seed)
        else:
            r_pressed = False

        # 1 key = Toggle Wireframe
        if keys.get(ord('1'), 0) == 1:
            if not wireframe_pressed:
                wireframe_pressed = True
                wireframe_enabled = not wireframe_enabled
                p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1 if wireframe_enabled else 0)
                print(f"Wireframe Mode: {'ON' if wireframe_enabled else 'OFF'}")
        else:
            wireframe_pressed = False
        
        cam.update()
        p.stepSimulation()
        time.sleep(1/60)


if __name__ == "__main__":
    load_ski_village()

