import pybullet as p
import pybullet_data
import time
import math
import city_gen
import os
import random

# ============================================================================
# CONFIG
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SWARM_ROOT = os.path.dirname(BASE_DIR)
ASSETS_ROOT = os.path.join(SWARM_ROOT, "assets")
KENNEY_DIR = os.path.join(ASSETS_ROOT, "kenney")
CUSTOM_DIR = os.path.join(ASSETS_ROOT, "custom")
OTHER_SOURCES_DIR = os.path.join(ASSETS_ROOT, "other_sources")
SCALE_FACTOR = 5.0
MODEL_BASE_SIZE = 1.8 


def resolve_asset_path(rel_path):
    """Resolve asset path from unified asset root."""
    rel_norm = rel_path.replace("\\", "/")
    if rel_norm.startswith("obj_converted/"):
        return os.path.join(OTHER_SOURCES_DIR, rel_path)
    return os.path.join(KENNEY_DIR, rel_path)

COLORS = {
    'grass': [0.13, 0.55, 0.13, 1], 
    'road': [0.3, 0.3, 0.3, 1],
}

ASSET_MAP = {
    'intersection': ['kenney_roads/Models/OBJ format/road-crossroad.obj'],
    'corner': ['kenney_roads/Models/OBJ format/road-bend.obj'], # Rounded corner per user request
    't_junction': ['kenney_roads/Models/OBJ format/road-intersection.obj'],
    'straight': ['kenney_roads/Models/OBJ format/road-straight.obj'],
    'crossing': ['kenney_roads/Models/OBJ format/road-crossing.obj'],
    'roundabout': ['kenney_roads/Models/OBJ format/road-roundabout.obj'],
    'roundabout_arm': [], # Render nothing (Ghost road to prevent building spawn)
    'dead_end': ['kenney_roads/Models/OBJ format/road-end-round.obj'],
    'streetlight': ['obj_converted/streetlight.obj'],
    'traffic_light': ['obj_converted/trafficlight_A.obj'], # Using CityBits asset (fallback)
    'tree': ['kenney_suburban/tree-large.obj', 'kenney_suburban/tree-small.obj'],
    
    # RESTORED for Fallback
    'house': [
        'kenney_suburban/building-type-a.obj', 'kenney_suburban/building-type-b.obj',
        'kenney_suburban/building-type-c.obj', 'kenney_suburban/building-type-d.obj',
        'kenney_suburban/building-type-e.obj', 'kenney_suburban/building-type-f.obj',
        'kenney_suburban/building-type-g.obj', 'kenney_suburban/building-type-h.obj',
        'kenney_suburban/building-type-i.obj', 'kenney_suburban/building-type-j.obj',
        'kenney_suburban/building-type-k.obj', 'kenney_suburban/building-type-l.obj',
        'kenney_suburban/building-type-m.obj', 'kenney_suburban/building-type-n.obj',
        'kenney_suburban/building-type-o.obj', 'kenney_suburban/building-type-p.obj',
        'kenney_suburban/building-type-q.obj', 'kenney_suburban/building-type-r.obj',
        'kenney_suburban/building-type-s.obj', 'kenney_suburban/building-type-t.obj',
        'kenney_suburban/building-type-u.obj'
    ],
    'apt': [
        'kenney_commercial/building-a.obj', 'kenney_commercial/building-b.obj',
        'kenney_commercial/building-c.obj', 'kenney_commercial/building-d.obj',
        'kenney_commercial/building-e.obj', 'kenney_commercial/building-f.obj',
        'kenney_commercial/building-g.obj', 'kenney_commercial/building-h.obj',
        'kenney_commercial/building-i.obj', 'kenney_commercial/building-j.obj',
        'kenney_commercial/building-k.obj', 'kenney_commercial/building-l.obj',
        'kenney_commercial/building-m.obj', 'kenney_commercial/building-n.obj'
    ],
    'tower': [
        'kenney_commercial/building-skyscraper-a.obj', 'kenney_commercial/building-skyscraper-b.obj',
        'kenney_commercial/building-skyscraper-c.obj', 'kenney_commercial/building-skyscraper-d.obj',
        'kenney_commercial/building-skyscraper-e.obj'
    ],
    
    # TRAFFIC
    'sedan': ['kenney_car-kit/Models/OBJ format/sedan.obj'],
    'taxi': ['kenney_car-kit/Models/OBJ format/taxi.obj'],
    'police': ['kenney_car-kit/Models/OBJ format/police.obj'],
    'suv': ['kenney_car-kit/Models/OBJ format/suv.obj'],
    'truck': ['kenney_car-kit/Models/OBJ format/truck.obj'],
    'van': ['kenney_car-kit/Models/OBJ format/van.obj'],
    'hatchback-sports': ['kenney_car-kit/Models/OBJ format/hatchback-sports.obj']
}

spawned_bodies = []
ASSET_CACHE = {} # Cache for (path, tuple(scale), tuple(color)) -> (vis_id, col_id)

# ============================================================================
# LOGIC
# ============================================================================

def ceil_half(x):
    return math.ceil(x * 2) / 2

# Raw Data (Filename, Width_X, Depth_Z) -> Uses Z as Depth because Y is Up/Height
RAW_DATA = [
    # HOUSES (X, Z)
    ('house', 'kenney_suburban/building-type-a.obj', 6.50, 5.14),
    ('house', 'kenney_suburban/building-type-b.obj', 9.14, 5.70),
    ('house', 'kenney_suburban/building-type-c.obj', 6.44, 5.14),
    ('house', 'kenney_suburban/building-type-d.obj', 8.79, 5.14),
    ('house', 'kenney_suburban/building-type-e.obj', 6.50, 5.14),
    ('house', 'kenney_suburban/building-type-f.obj', 7.14, 7.03),
    ('house', 'kenney_suburban/building-type-g.obj', 7.25, 5.89),
    ('house', 'kenney_suburban/building-type-h.obj', 6.50, 4.58),
    ('house', 'kenney_suburban/building-type-i.obj', 6.44, 5.14),
    ('house', 'kenney_suburban/building-type-j.obj', 6.85, 4.58),
    ('house', 'kenney_suburban/building-type-k.obj', 4.61, 5.10),
    ('house', 'kenney_suburban/building-type-l.obj', 5.20, 5.12), # Calibrated
    ('house', 'kenney_suburban/building-type-m.obj', 7.14, 7.14),
    ('house', 'kenney_suburban/building-type-n.obj', 8.93, 6.89),
    ('house', 'kenney_suburban/building-type-o.obj', 6.35, 5.14),
    ('house', 'kenney_suburban/building-type-p.obj', 6.20, 4.95),
    ('house', 'kenney_suburban/building-type-q.obj', 6.20, 4.43),
    ('house', 'kenney_suburban/building-type-r.obj', 5.16, 5.12), # Calibrated
    ('house', 'kenney_suburban/building-type-s.obj', 7.03, 5.44),
    ('house', 'kenney_suburban/building-type-t.obj', 6.60, 7.05), # Calibrated
    ('house', 'kenney_suburban/building-type-u.obj', 7.14, 5.44),
    
    # APTS (X, Z)
    ('apt', 'kenney_commercial/building-a.obj', 4.42, 4.70),
    ('apt', 'kenney_commercial/building-b.obj', 4.85, 4.70),
    ('apt', 'kenney_commercial/building-c.obj', 4.42, 5.45),
    ('apt', 'kenney_commercial/building-d.obj', 4.20, 4.50),
    ('apt', 'kenney_commercial/building-e.obj', 8.20, 5.04),
    ('apt', 'kenney_commercial/building-f.obj', 4.20, 5.15),
    ('apt', 'kenney_commercial/building-g.obj', 4.85, 4.61),
    ('apt', 'kenney_commercial/building-h.obj', 4.42, 5.04),
    ('apt', 'kenney_commercial/building-i.obj', 6.20, 6.51),
    ('apt', 'kenney_commercial/building-j.obj', 10.42, 6.70),
    ('apt', 'kenney_commercial/building-k.obj', 10.42, 4.71),
    ('apt', 'kenney_commercial/building-l.obj', 6.85, 7.01),
    ('apt', 'kenney_commercial/building-m.obj', 6.20, 6.21),
    ('apt', 'kenney_commercial/building-n.obj', 11.60, 9.10),
    
    # TOWERS (X, Z)
    ('tower', 'kenney_commercial/building-skyscraper-a.obj', 6.80, 6.80),
    ('tower', 'kenney_commercial/building-skyscraper-b.obj', 6.80, 6.80),
    ('tower', 'kenney_commercial/building-skyscraper-c.obj', 6.40, 6.94),
    ('tower', 'kenney_commercial/building-skyscraper-d.obj', 6.40, 6.94),
    ('tower', 'kenney_commercial/building-skyscraper-e.obj', 6.48, 6.21),
]

ASSET_SPECS = {'house':[], 'apt':[], 'tower':[]}

def init_specs():
    for cat, path, raw_w, raw_d in RAW_DATA:
        # Use slightly looser margin for fit (1.1x)
        margin = 1.10
        final_w = ceil_half(raw_w * margin)
        final_d = ceil_half(raw_d * margin)
        
        ASSET_SPECS[cat].append({
            'w': final_w,
            'd': final_d,
            'path': resolve_asset_path(path),
            'area': final_w * final_d,
            'raw_w': raw_w,
            'raw_d': raw_d
        })

init_specs()

def configure_templates():
    """Inject RECTANGULAR templates into city_gen."""
    print("Configuring Rectangular Templates...")
    new_templates = {'house': [], 'apt': [], 'tower': []}
    
    for cat in ASSET_SPECS:
        seen_dims = set()
        for spec in ASSET_SPECS[cat]:
            dims = (spec['w'], spec['d'])
            if dims not in seen_dims:
                new_templates[cat].append({'w': spec['w'], 'd': spec['d']})
                seen_dims.add(dims)
                
    city_gen.TEMPLATES = new_templates
    print(f"  Loaded {len(new_templates['house'])} House footprints.")
    print(f"  Loaded {len(new_templates['apt'])} Apt footprints.")
    print(f"  Loaded {len(new_templates['tower'])} Tower footprints.")


def get_asset_for_rect(category, lot_w, lot_d):
    """Find asset that fits the SPECIFIC WxH of the lot."""
    if category not in ASSET_SPECS: return None
    
    matches = []
    candidates = ASSET_SPECS[category]
    
    for spec in candidates:
        # Check standard orientation
        if abs(spec['w'] - lot_w) < 0.1 and abs(spec['d'] - lot_d) < 0.1:
            matches.append(spec)
        # Check rotated orientation
        elif abs(spec['d'] - lot_w) < 0.1 and abs(spec['w'] - lot_d) < 0.1:
            matches.append(spec)
            
    if matches:
        return random.choice(matches)['path']
    
    # Fallback 1: Closest Match by AREA
    if candidates:
        target_area = lot_w * lot_d
        closest = min(candidates, key=lambda x: abs(x['area'] - target_area))
        return closest['path']
        
    return None

def get_asset_for_zone(category, lot_w, lot_d, city_type, block_style=None):
    """
    Zone-aware asset selection.
    city_type: 1=Residential (suburban only), 2=Mixed, 3=Urban (commercial only)
    block_style: For Mixed mode, 'suburban' or 'commercial' (decided per-block)
    """
    
    # ===== TOWER ENFORCEMENT - SKYSCRAPERS ONLY =====
    # Any tower request gets ONLY skyscraper assets - NO small buildings!
    if category == 'tower':
        if 'tower' not in ASSET_SPECS:
            print("Warning: No tower category in ASSET_SPECS.")
            return None
        
        # ONLY allow skyscraper files
        skyscrapers = [s for s in ASSET_SPECS['tower'] 
                       if 'skyscraper' in s['path'].lower()]
        
        if not skyscrapers:
            print("Warning: No skyscraper assets found.")
            return None
        
        # Find exact dimension match
        for spec in skyscrapers:
            if abs(spec['w'] - lot_w) < 0.5 and abs(spec['d'] - lot_d) < 0.5:
                return spec['path']
            if abs(spec['d'] - lot_w) < 0.5 and abs(spec['w'] - lot_d) < 0.5:
                return spec['path']
        
        # Fallback: Closest skyscraper by area
        if skyscrapers:
            target_area = lot_w * lot_d
            closest = min(skyscrapers, key=lambda x: abs(x['area'] - target_area))
            return closest['path']
        
        return None
    # ===== END TOWER ENFORCEMENT =====
    
    # OVERRIDE category based on zone
    effective_category = category
    folder_filter = None
    
    if city_type == 1:
        # RESIDENTIAL: Force everything to suburban houses
        effective_category = 'house'
        folder_filter = 'kenney_suburban'
    elif city_type == 3:
        # URBAN: No suburban allowed - if house requested, use apt instead
        if category == 'house':
            effective_category = 'apt'
        folder_filter = 'kenney_commercial'
    else:
        # MIXED: Use block_style
        if block_style == 'suburban':
            effective_category = 'house'
            folder_filter = 'kenney_suburban'
        elif block_style == 'commercial':
            if category == 'house':
                effective_category = 'apt'
            folder_filter = 'kenney_commercial'
    
    # Get candidates
    if effective_category not in ASSET_SPECS: 
        return None
        
    candidates = ASSET_SPECS[effective_category]
    
    if folder_filter:
        filtered = [s for s in candidates if folder_filter in s['path']]
        if filtered:
            candidates = filtered
        # If filter returns nothing, DON'T fallback - stay strict
    
    # HARD MODE TOWER FILTER: ONLY skyscraper files allowed - NOTHING ELSE!
    if city_type == 3 and effective_category == 'tower':
        # STRICT: Only 'skyscraper' keyword - no 'office', 'tower', 'large' etc.
        skyscraper_only = [s for s in candidates if 'skyscraper' in s['path'].lower()]
        if skyscraper_only:
            candidates = skyscraper_only
        else:
            # NO SKYSCRAPERS FOUND - Return None to skip this building
            return None
    
    if not candidates:
        return None
    
    # Find matching dimensions
    matches = []
    for spec in candidates:
        if abs(spec['w'] - lot_w) < 0.1 and abs(spec['d'] - lot_d) < 0.1:
            matches.append(spec)
        elif abs(spec['d'] - lot_w) < 0.1 and abs(spec['w'] - lot_d) < 0.1:
            matches.append(spec)
            
    if matches:
        return random.choice(matches)['path']
    
    # Fallback: Closest by area - BUT DISABLED for Hard Mode towers
    if city_type == 3 and effective_category == 'tower':
        return None
    
    if candidates:
        target_area = lot_w * lot_d
        closest = min(candidates, key=lambda x: abs(x['area'] - target_area))
        return closest['path']
        
    return None

def get_asset_path(name):
    """Generic Fallback."""
    variants = ASSET_MAP.get(name, [])
    if not variants and name in ['straight_v', 'straight_h']: variants = ASSET_MAP['straight']
    if variants: return resolve_asset_path(random.choice(variants))
    return None

# ============================================================================
# ZONE-BASED BUILDING DISTRIBUTION
# ============================================================================
def get_building_zone(x, y, map_size=200):
    """Calculate zone based on distance from city center."""
    center = map_size / 2
    distance = math.sqrt((x - center)**2 + (y - center)**2)
    max_dist = math.sqrt(2) * (map_size / 2)
    ratio = distance / max_dist
    
    if ratio > 0.6:
        return 'outer'
    elif ratio > 0.3:
        return 'middle'
    else:
        return 'center'

def get_zone_building_type(zone, city_type):
    """Return building type based on zone probabilities."""
    if city_type == 1:
        return 'house'
    
    r = random.random()
    
    if zone == 'outer':
        if city_type == 2:
            if r < 0.80: return 'house'
            elif r < 0.95: return 'apt'
            else: return 'tower'
        else:
            if r < 0.50: return 'house'
            elif r < 0.85: return 'apt'
            else: return 'tower'
    elif zone == 'middle':
        if city_type == 2:
            if r < 0.30: return 'house'
            elif r < 0.80: return 'apt'
            else: return 'tower'
        else:
            if r < 0.15: return 'house'
            elif r < 0.55: return 'apt'
            else: return 'tower'
    else:  # center
        if city_type == 2:
            if r < 0.05: return 'house'
            elif r < 0.40: return 'apt'
            else: return 'tower'
        else:
            if r < 0.02: return 'house'
            elif r < 0.25: return 'apt'
            else: return 'tower'

def spawn_asset_exact(path, x, y, z, rotation_deg, scale_vec, rgbaColor=None):
    if not os.path.exists(path): return None
    
    # Cache key: include scale and color as they affect the shape creation
    cache_key = (path, tuple(scale_vec), tuple(rgbaColor) if rgbaColor else None)
    
    if cache_key in ASSET_CACHE:
        vis_id, col_id = ASSET_CACHE[cache_key]
    else:
        vis_id = p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=scale_vec, rgbaColor=rgbaColor)
        col_id = p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=scale_vec)
        ASSET_CACHE[cache_key] = (vis_id, col_id)

    base_x_rot = 1.5708
    rad_z = math.radians(rotation_deg)
    orn = p.getQuaternionFromEuler([base_x_rot, 0, rad_z])
    body_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id, basePosition=[x,y,z], baseOrientation=orn)
    spawned_bodies.append(body_id)
    
    try:
        p.getAABB(body_id)
    except:
        pass
        
    return body_id

def spawn_asset_with_random_color(path, x, y, z, rotation_deg, scale_vec, warm_mode=False):
    """Spawn an asset with random color variant (for commercial buildings)."""
    import tempfile
    import shutil
    
    # Warm Mode (Hard Mode Towers): 10% chance to be Earth-tone
    if warm_mode:
        # Earth tones: Bronze, Warm Concrete (NO DARK)
        earth_tones = [
            [0.70, 0.55, 0.45, 1], # Lighter Bronze
            [0.75, 0.70, 0.65, 1]  # Light Warm Concrete
        ]
        chosen_color = random.choice(earth_tones)
        return spawn_asset_exact(path, x, y, z, rotation_deg, scale_vec, rgbaColor=chosen_color)

    # Only apply color variants to commercial buildings
    if 'kenney_commercial' not in path:
        return spawn_asset_exact(path, x, y, z, rotation_deg, scale_vec)
    
    # Randomly select color: original (50%), green (25%), orange (25%)
    color_choice = random.random()
    if color_choice < 0.5:
        # Original color - no modification needed
        return spawn_asset_exact(path, x, y, z, rotation_deg, scale_vec)
    elif color_choice < 0.75:
        mtl_suffix = '-green'
    else:
        mtl_suffix = '-orange'
    
    # Read original OBJ and modify MTL reference
    try:
        with open(path, 'r') as f:
            obj_content = f.read()
        
        # Find and replace the mtllib reference
        # e.g., "mtllib building-a.mtl" -> "mtllib building-a-green.mtl"
        import re
        def replace_mtl(match):
            original = match.group(1)
            base_name = original.replace('.mtl', '')
            return f'mtllib {base_name}{mtl_suffix}.mtl'
        
        modified_content = re.sub(r'mtllib\s+(\S+\.mtl)', replace_mtl, obj_content)
        
        # Write to temp file
        temp_dir = os.path.dirname(path)
        temp_path = os.path.join(temp_dir, f'_temp_colored{mtl_suffix}.obj')
        with open(temp_path, 'w') as f:
            f.write(modified_content)
        
        result = spawn_asset_exact(temp_path, x, y, z, rotation_deg, scale_vec)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
        return result
    except Exception as e:
        print(f"Warning: Color variant failed ({e}), using original asset.")
        return spawn_asset_exact(path, x, y, z, rotation_deg, scale_vec)

def load_asset_simple(name, x, y, z, rotation_deg):
    path = get_asset_path(name)
    if not path: return
    # Apply scale (slightly smaller than 2.0 as requested)
    if 'kenney_roads' in path: 
        scale_mod = 1.5 
    elif 'kenney_car-kit' in path:
        scale_mod = 0.25 # Reduced to 0.25 as per user request (was 0.5)
    elif name == 'streetlight':
        scale_mod = 0.8 # Reduced by 20% from default 1.0
    else: 
        scale_mod = 1.0
        
    s_val = SCALE_FACTOR * scale_mod
    
    # Kenney lights need +90 deg rotation to face correct way relative to road.
    final_rot = rotation_deg + 90 if 'kenney_roads' in path else rotation_deg
    
    spawn_asset_exact(path, x, y, z, final_rot, [s_val, s_val, s_val])

def spawn_grass_block(x, y, w, h):
    half_w = w / 2; half_h = h / 2; thickness = 0.05
    vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_w, half_h, thickness], rgbaColor=COLORS['grass'])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_id, basePosition=[x + half_w, y + half_h, -0.1])

# ============================================================================
# UNREAL CAMERA CONTROLLER (FIXED FPS)
# ============================================================================
class CameraController:
    def __init__(self):
        self.x = 100; self.y = 100; self.z = 50
        self.yaw = 0; self.pitch = -30; self.dist = 0.1
        self.speed = 0.5; self.mouse_sensitivity = 0.5
        self.last_mouse_x = 0; self.last_mouse_y = 0; self.lmb_held = False
        self.update_camera()

    def update(self, keys=None):
        if keys is None:
            keys = p.getKeyboardEvents()
        mouse = p.getMouseEvents()
        dx = 0; dy = 0
        MOUSE_MOVE = 1; MOUSE_BTN = 2
        
        for e in mouse:
            if e[0] == MOUSE_MOVE:
                if self.lmb_held: dx = e[1] - self.last_mouse_x; dy = e[2] - self.last_mouse_y
                self.last_mouse_x = e[1]; self.last_mouse_y = e[2]
            if e[0] == MOUSE_BTN and e[3] == 0:
                self.lmb_held = (e[4] == 3 or e[4] == 1)
                if self.lmb_held: self.last_mouse_x = e[1]; self.last_mouse_y = e[2]

        if self.lmb_held and (dx or dy):
            self.yaw -= dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            self.pitch = max(-89, min(89, self.pitch))

        # FPS MOVEMENT - "W" moves Forward along camera yaw
        move_speed = self.speed * (3.0 if keys.get(p.B3G_SHIFT, 0) else 1.0)
        
        # Calculate Forward/Right vectors based on YAW only (movement remains flat usually)
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
# MAIN
# ============================================================================
info_text_id = None
current_density = 2

def update_info_display(stats):
    global info_text_id
    info_text = f"Buildings: {stats['buildings']} | Houses: {stats['houses']} | Apts: {stats['apts']} | Towers: {stats['towers']} | Lamps: {stats['lamps']}"
    if info_text_id: p.removeUserDebugItem(info_text_id)
    info_text_id = p.addUserDebugText(info_text, [100, 100, 80], [1, 1, 0], 1.5, 0)

def regenerate_city_with_settings(target_area, lamp_freq, city_type, density=2, max_block_size=80, difficulty=1, road_width=10.0, color_mode=1):
    """Regenerate city with user settings. color_mode: 1=Original, 2=Partial 25%, 3=Full Color."""
    global spawned_bodies, current_density
    current_density = density
    
    # HARD MODE OVERRIDE: Ignore slider settings, use fixed challenging values
    if difficulty == 3:
        target_area = 400       # Minimum block area
        max_block_size = 1000   # Maximum block area
        city_type = 3           # All towers
        print(f"🔴 HARD MODE ACTIVE! Overriding settings: min=400, max=1000, towers only")
    
    # Strict Math: road_scale is derived from width (10m base)
    road_scale = road_width / 10.0
    configure_templates()
    
    # FULL RESET
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    spawned_bodies.clear()
    ASSET_CACHE.clear() # Clear cache because resetSimulation destroys existing shape IDs
    
    start_time = time.time()
    
    # OPTIMIZATION: Disable Rendering during generation for SPEED
    print("🚀 Turbo Mode: Rendering Disabled temporarily...")
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    
    print("\n" + "="*50)
    print("🌆 GENERATING CITY...")
    print("="*50)
    try:
        # strict strict strict
        city_data = city_gen.generate_city(seed=int(time.time()), target_area=target_area, city_type=city_type, max_block_size=max_block_size, difficulty=difficulty, road_scale=road_scale)
    except Exception as e:
        print(f"ERROR: city_gen failed: {e}")
        return {'buildings':0, 'houses':0, 'apts':0, 'towers':0, 'lamps':0}

    # Grass
    for block in city_data['blocks']:
        spawn_grass_block(block.rect.x, block.rect.y, block.rect.w, block.rect.h)
    
    # Roads - strict grid placement
    effective_tile = road_width  # Exact width
    for tile in city_data['roads']:
        # Center is strictly tile + half width
        cx = tile.x + effective_tile / 2
        cy = tile.y + effective_tile / 2
        
        # Strict Scaling with Seam Prevention
        # Strict Scaling (No Overlap for 3D Meshes to avoid Z-Fighting/Seams)
        # 1.0 = Exact fit.
        visual_scale = (road_width / 10.0) * 1.00
        
        # FIX: Road assets are XZ plane (Y-up) but rotated 90deg X.
        # So Local X -> World X. Local Z -> World Y. Local Y -> World Z (Height).
        # Kenney Roads are 1x1 unit base, while legacy assets were 2x2.
        # We need to DOUBLE the scale to fill the same 10m tile.
        scale_mod = 2.0
        # Reduce scale for Roundabout to fit the 10m grid better without overlapping neighbors
        if tile.type == 'roundabout':
            # 3x3 Space cleared in city_gen.py -> Use FULL scale!
            # It needs to span 3 tiles roughly. 
            # Actually, keep it standard 2.0x (10m) first, see if it fits the gap.
            # If the model is huge, it will fill the void.
            scale_mod = 2.0
            
        road_scale_vec = [scale_mod * SCALE_FACTOR * visual_scale, SCALE_FACTOR, scale_mod * SCALE_FACTOR * visual_scale]
        path = get_asset_path(tile.type)
        if path:
            # Z-Fighting Prevention: Not needed for 3D meshes with exact fit
            z_offset = 0.00
            if tile.type == 'roundabout': z_offset = 0.00
            
            # Kenney Roads need +90 degrees Z rotation to align with grid
            spawn_asset_exact(path, cx, cy, z_offset, tile.rotation + 90, road_scale_vec)


    # ZONE-AWARE BUILDING PLACEMENT
    # Assign building TYPE to each block based on its zone (not individual buildings)
    block_styles = {}
    block_types = {}  # New: store building type per block
    
    for block in city_data['blocks']:
        block_id = (block.rect.x, block.rect.y)
        
        # Calculate block center
        block_cx = block.rect.x + block.rect.w / 2
        block_cy = block.rect.y + block.rect.h / 2
        
        # Get zone for this block
        zone = get_building_zone(block_cx, block_cy)
        
        if difficulty == 3:
            # HARD MODE: All blocks are towers only
            block_types[block_id] = 'tower'
            block_styles[block_id] = 'commercial'
        elif city_type == 1:  # Residential
            block_types[block_id] = 'house'
            block_styles[block_id] = 'suburban'
        elif city_type in [2, 3]:  # Mixed or Urban
            # Get building type for this zone
            building_type = get_zone_building_type(zone, city_type)
            block_types[block_id] = building_type
            # Set style based on building type
            if building_type == 'house':
                block_styles[block_id] = 'suburban'
            else:
                block_styles[block_id] = 'commercial'
    
    building_count = 0
    houses_count = 0
    apts_count = 0
    towers_count = 0
    for b in city_data['buildings']:
        rect = b.rect
        
        # Handle PARKS
        if b.type == 'park':
            # Green boundary
            colors = [0, 1, 0]
            z = 0.5
            p.addUserDebugLine([rect.x, rect.y, z], [rect.x+rect.w, rect.y, z], colors, 2)
            p.addUserDebugLine([rect.x+rect.w, rect.y, z], [rect.x+rect.w, rect.y+rect.h, z], colors, 2)
            p.addUserDebugLine([rect.x+rect.w, rect.y+rect.h, z], [rect.x, rect.y+rect.h, z], colors, 2)
            p.addUserDebugLine([rect.x, rect.y+rect.h, z], [rect.x, rect.y, z], colors, 2)
            
            # Spawn Trees
            cx, cy = rect.x + rect.w/2, rect.y + rect.h/2
            area = rect.w * rect.h
            if area < 100:
                load_asset_simple('tree', cx, cy, 0, random.uniform(0, 360))
            else:
                # Scatter trees
                num_trees = int(area / 50) # 1 tree per 50m2
                for _ in range(num_trees):
                    tx = random.uniform(rect.x + 2, rect.x + rect.w - 2)
                    ty = random.uniform(rect.y + 2, rect.y + rect.h - 2)
                    load_asset_simple('tree', tx, ty, 0, random.uniform(0, 360))
            continue
        
        # ===== HARD MODE ABSOLUTE FILTER =====
        # In Hard Mode (difficulty=3), ONLY SPAWN TOWERS. Skip everything else.
        if difficulty == 3 and b.type != 'tower':
            continue
        
        # Determine block style and type for this building
        block_style = 'suburban'
        effective_type = b.type
        
        # Find which block this building belongs to
        cx_temp, cy_temp = rect.x + rect.w/2, rect.y + rect.h/2
        for block in city_data['blocks']:
            bx, by, bw, bh = block.rect.x, block.rect.y, block.rect.w, block.rect.h
            if bx <= cx_temp <= bx + bw and by <= cy_temp <= by + bh:
                block_id = (block.rect.x, block.rect.y)
                block_style = block_styles.get(block_id, 'suburban')
                # Use the pre-calculated block type for consistency
                if block_id in block_types:
                    effective_type = block_types[block_id]
                break
        
        # Use ZONE-AWARE selection with the block's type
        path = get_asset_for_zone(effective_type, rect.w, rect.h, city_type, block_style)
        
        # FALLBACK - DISABLED in Hard Mode (must be exact skyscraper match)
        if not path:
            if difficulty == 3:
                continue
            path = get_asset_path(b.type)
             
        if not path: 
            print(f"ERROR: No path found for {b.type} even after fallback!")
            continue
        
        scale_mod = 1.0 if density == 3 else (0.9 if density == 2 else 0.8)
        final_scale = SCALE_FACTOR * scale_mod
        
        cx = rect.x + rect.w/2
        cy = rect.y + rect.h/2
        
        # COLOR MODE LOGIC
        # Mode 1: Base texture only (no variations, no color override)
        # Mode 2: Texture variations (colormap, variation-a, variation-b)
        # Mode 3: Variations + random RGBA colors
        
        # Pleasant, non-dark color palette for Mode 3
        PLEASANT_COLORS = [
            [0.85, 0.75, 0.65, 1],  # Warm Sand
            [0.70, 0.80, 0.85, 1],  # Light Sky Blue
            [0.80, 0.85, 0.75, 1],  # Soft Mint
            [0.90, 0.85, 0.75, 1],  # Cream
            [0.75, 0.75, 0.85, 1],  # Soft Lavender
            [0.85, 0.80, 0.70, 1],  # Soft Peach
            [0.80, 0.90, 0.90, 1],  # Light Cyan
            [0.95, 0.90, 0.80, 1],  # Light Ivory
        ]
        
        color_override = None
        use_variations = False
        
        if color_mode == 1:
            # Mode 1: Base texture only
            color_override = None
            use_variations = False
        elif color_mode == 2:
            # Mode 2: Texture variations
            color_override = None
            use_variations = True
        elif color_mode == 3:
            # Mode 3: 50% variations, 50% random RGBA
            if random.random() < 0.5:
                use_variations = True
            else:
                color_override = random.choice(PLEASANT_COLORS)
        
        # Spawn based on mode
        if color_override:
            bid = spawn_asset_exact(path, cx, cy, 0, b.facing + 180, [final_scale, final_scale, final_scale], rgbaColor=color_override)
        elif use_variations:
            bid = spawn_asset_with_random_color(path, cx, cy, 0, b.facing + 180, [final_scale, final_scale, final_scale], warm_mode=False)
        else:
            bid = spawn_asset_exact(path, cx, cy, 0, b.facing + 180, [final_scale, final_scale, final_scale])
        
        if bid is not None:
            building_count += 1
            if effective_type == 'house': houses_count += 1
            elif effective_type == 'apt': apts_count += 1
            elif effective_type == 'tower': towers_count += 1
        else:
            print(f"ERROR: Spawn failed for {path}")
    # Lamps - fixed positions at road edges
    effective_tile = 10 * road_scale
    half_tile = effective_tile / 2
    # User requested 4.575m offset for 10m road (5m half). Margin = 0.425.
    lamp_offset = half_tile - 0.425
    
    lamp_count = 0
    road_idx = 0
    for tile in city_data['roads']:
        if tile.type == 'intersection': continue
        road_idx += 1
        if road_idx % lamp_freq != 0: continue
        
        # Center of tile adjusted for road_scale
        cx = tile.x + half_tile
        cy = tile.y + half_tile
        
        if tile.type == 'straight_v':
            load_asset_simple('streetlight', cx - lamp_offset, cy, 0, 180)
            load_asset_simple('streetlight', cx + lamp_offset, cy, 0, 0)
            lamp_count += 2
        elif tile.type == 'straight_h':
            load_asset_simple('streetlight', cx, cy - lamp_offset, 0, 270)
            load_asset_simple('streetlight', cx, cy + lamp_offset, 0, 90)
            lamp_count += 2
            
    # Traffic Lights - Pedestrian Crossings ONLY
    # Per user request: "If we have a pedestrian signal, put it with the crossing"
    # We will use the standard light (trafficlight_A) to simulate a crossing signal.
    tl_count = 0
    
    for tile in city_data['roads']:
        if tile.type == 'crossing':
            # Crossing is a straight road converted.
            # We need lights at the STOP line for cars.
            # Rotation of tile: 
            # 0 (Vertical road): Crossing happens at Y. Lights face North and South.
            # 90 (Horizontal): Lights face East and West.
            
            effective_tile = 10 * road_scale
            half_tile = effective_tile / 2
            offset = half_tile - 0.425 # Edge of pavement
            
            cx = tile.x + half_tile
            cy = tile.y + half_tile
            
            # Position lights on the RIGHT side of the road for oncoming traffic.
            # UPDATED: Place them "In the Middle", aligned with the crossing strip.
            
            if tile.rotation == 0: # Vertical Road
                # Right side (East): Faces West (270)
                load_asset_simple('traffic_light', cx + offset, cy, 0, 270) 
                
                # Left side (West): Faces East (90)
                load_asset_simple('traffic_light', cx - offset, cy, 0, 90)
                tl_count += 2
                
            elif tile.rotation == 90: # Horizontal Road
                # Right side (South): Faces North (0)
                load_asset_simple('traffic_light', cx, cy - offset, 0, 0)
                
                # Top side (North): Faces South (180)
                load_asset_simple('traffic_light', cx, cy + offset, 0, 180)
                tl_count += 2

    car_assets = ['sedan', 'taxi', 'police', 'suv', 'truck', 'van', 'hatchback-sports']
    cars_spawned = 0
    lane_offset_factor = 0.18 # Reduced to 0.18 (Closer to center line)
    
    # Define constants locally for this scope
    TILE_SIZE = city_gen.TILE_SIZE
    ROAD_SCALE = road_width / TILE_SIZE
    
    for tile in city_data['roads']:
        if tile.type in ['straight_v', 'straight_h']:
            # 35% Chance to spawn a car on THIS tile
            if random.random() < 0.35:
                car_model = random.choice(car_assets)
                
                # Determine Lane (Right Hand Traffic)
                # Randomly choose Direction 1 or Direction 2
                direction = random.choice([1, -1]) 
                
                offset = TILE_SIZE * ROAD_SCALE * lane_offset_factor
                
                cx, cy = tile.x + (TILE_SIZE * ROAD_SCALE) / 2, tile.y + (TILE_SIZE * ROAD_SCALE) / 2
                
                car_x, car_y = cx, cy
                car_rot = 0
                
                if tile.type == 'straight_v':
                    # Vertical Road (0 deg)
                    if direction == 1: # Heading North (+Y)
                        # Right side = East (+X)
                        car_x = cx + offset
                        car_rot = 0 + 180 # Face South (Flip 180)
                    else: # Heading South (-Y)
                        # Right side = West (-X)
                        car_x = cx - offset
                        car_rot = 180 + 180 # Face North (Flip 180) -> 360/0
                        
                elif tile.type == 'straight_h':
                    # Horizontal Road (90 deg)
                    if direction == 1: # Heading East (+X)
                        # Right side = South (-Y)
                        car_y = cy - offset
                        car_rot = 270 + 180 # Face West (Flip 180) -> 450 -> 90
                    else: # Heading West (-X)
                         # Right side = North (+Y)
                         car_y = cy + offset
                         car_rot = 90 + 180 # Face East (Flip 180) -> 270
                        
                load_asset_simple(car_model, car_x, car_y, 0, car_rot)
                cars_spawned += 1
                        
                load_asset_simple(car_model, car_x, car_y, 0, car_rot)
                cars_spawned += 1

    total_time = time.time() - start_time
    
    # RESTORE RENDERING
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    
    # FINAL SUMMARY REPORT
    print("\n" + "="*50)
    print(f"✨ CITY GENERATION COMPLETE! ✨")
    print("="*50)
    
    # TIME STANDALONE
    print(f"\n⏱️  EXECUTION TIME : {total_time:.2f} seconds\n")
    
    print("-" * 50)
    print("📊 STATISTICS:")
    print(f"    🏙️  Buildings  : {building_count}")
    print(f"        🏠 Houses  : {houses_count}")
    print(f"        🏢 Apts    : {apts_count}")
    print(f"        🌆 Towers  : {towers_count}")
    print(f"    🛣️  Road Tiles : {len(city_data['roads'])}")
    print(f"    💡 Lamps      : {lamp_count}")
    print(f"    🚦 Signals    : {tl_count}")
    print(f"    🚗 Cars       : {cars_spawned}")
    print("="*50 + "\n")

    return {'buildings': building_count, 'houses': houses_count, 
            'apts': apts_count, 
            'towers': towers_count, 
            'lamps': lamp_count, 'traffic_lights': tl_count, 'cars': cars_spawned}

def run_simulation():
    p.connect(p.GUI); p.setAdditionalSearchPath(pybullet_data.getDataPath()); p.setGravity(0, 0, -9.8)
    cam = CameraController()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    # DISABLE DEFAULT KEYBOARD SHORTCUTS (Prevents W = Wireframe)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    
    btn_gen = p.addUserDebugParameter("[ REGENERATE ]", 1, 0, 0); last_gen = 0
    sl_size = p.addUserDebugParameter("Min Block Area", 400, 2000, 400)
    sl_maxblk = p.addUserDebugParameter("Max Block Area", 1000, 5000, 2000)
    sl_lamp = p.addUserDebugParameter("Lamps", 1, 3, 2)
    sl_type = p.addUserDebugParameter("Type (1=Res, 2=Mix, 3=Urb)", 1, 3, 2)
    sl_dens = p.addUserDebugParameter("Density (1=Lo, 3=Hi)", 1, 3, 2)
    sl_diff = p.addUserDebugParameter("Difficulty (1=Easy, 3=Hard)", 1, 3, 1)
    sl_width = p.addUserDebugParameter("Road Width (m)", 6.0, 12.0, 10.0)
    sl_color = p.addUserDebugParameter("Color Mode (1=Orig, 2=Partial, 3=Full)", 1, 3, 1)
    
    # Initial city generation
    target_area = int(p.readUserDebugParameter(sl_size))
    max_block = int(p.readUserDebugParameter(sl_maxblk))
    lamp_freq = int(p.readUserDebugParameter(sl_lamp))
    city_type = int(p.readUserDebugParameter(sl_type))
    density = int(p.readUserDebugParameter(sl_dens))
    difficulty = int(round(p.readUserDebugParameter(sl_diff)))
    road_width = int(round(p.readUserDebugParameter(sl_width)))
    color_mode = int(round(p.readUserDebugParameter(sl_color)))
    # Hard Mode Override: Force 6m width AND Urban Type
    if difficulty == 3:
        road_width = 6
        city_type = 3
    
    regenerate_city_with_settings(target_area, lamp_freq, city_type, density, max_block_size=max_block, difficulty=difficulty, road_width=road_width, color_mode=color_mode)
    
    wireframe_on = False
    
    while p.isConnected():
        # INPUT HANDLING FIX: Fetch keys ONCE
        keys = p.getKeyboardEvents()
        
        # Update camera with same keys
        cam.update(keys)
        
        # Check for Manual Keybinds
        # Toggle Wireframe with 'P'
        if keys.get(112, 0) & p.KEY_WAS_TRIGGERED:
            wireframe_on = not wireframe_on
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1 if wireframe_on else 0)
            
        curr_gen = p.readUserDebugParameter(btn_gen)
        if curr_gen > last_gen:
            last_gen = curr_gen
            difficulty = int(round(p.readUserDebugParameter(sl_diff)))
            r_width = int(round(p.readUserDebugParameter(sl_width)))
            c_type = int(p.readUserDebugParameter(sl_type))
            
            # Hard Mode Override: Force 6m width AND Urban Type
            if difficulty == 3:
                r_width = 6
                c_type = 3
                
            regenerate_city_with_settings(
                p.readUserDebugParameter(sl_size), 
                int(p.readUserDebugParameter(sl_lamp)), 
                c_type, 
                int(p.readUserDebugParameter(sl_dens)),
                max_block_size=int(p.readUserDebugParameter(sl_maxblk)),
                difficulty=difficulty,
                road_width=r_width,
                color_mode=int(round(p.readUserDebugParameter(sl_color)))
            )

        p.stepSimulation(); time.sleep(1./60.)

if __name__ == "__main__":
    try:
        run_simulation()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n❌ FATAL ERROR: {e}")
    finally:
        input("\nPress Enter to exit...")


