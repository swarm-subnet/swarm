"""
🏙️ City Generator (Python Port)
Core logic for procedural city generation.
Ports the verified 2D grid/block algorithms from the JS prototype.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

# ============================================================================
# CONSTANTS
# ============================================================================
MAP_SIZE = 200          # 200m x 200m world
TILE_SIZE = 10          # Road width (10m)
DEFAULT_SEED = 42

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    
    @property
    def x2(self): return self.x + self.w
    @property
    def y2(self): return self.y + self.h
    @property
    def area(self): return self.w * self.h

@dataclass
class Building:
    rect: Rect
    type: str  # 'house', 'apt', 'tower'
    facing: int = 0  # Rotation in degrees (0=south, 90=west, 180=north, 270=east)
    color: List[float] = field(default_factory=lambda: [1,1,1])

@dataclass
class Block:
    id: int
    rect: Rect
    too_small: bool

@dataclass
class RoadTile:
    x: float
    y: float
    type: str  # 'intersection', 'straight_v', 'straight_h', 'corner', 't_junction'
    rotation: int = 0  # Rotation in degrees (0, 90, 180, 270)
    debug_label: str = ""  # Debug info

# ============================================================================
# RNG
# ============================================================================
class SeededRNG:
    def __init__(self, seed):
        self._seed = seed if seed is not None else random.randint(0, 999999)
        self._rng = random.Random(self._seed)

    def range(self, min_val, max_val):
        return self._rng.uniform(min_val, max_val)

    def rand_int(self, min_val, max_val):
        return self._rng.randint(min_val, max_val)

    def next_float(self):
        return self._rng.random()

    def choice(self, seq):
        """Select a random element from a non-empty sequence."""
        return self._rng.choice(seq)

# ============================================================================
# LOGIC
# ============================================================================

def generate_road_positions(rng: SeededRNG, min_spacing: int, target_area: int, tile_size: float = None):
    """Generate road positions using specified tile_size for grid spacing."""
    if tile_size is None:
        tile_size = TILE_SIZE  # Default to constant
    
    min_side_from_area = math.sqrt(target_area)
    needed_gap = max(min_spacing, min_side_from_area)
    
    # Step = Gap + RoadWidth
    raw_min_step = needed_gap + tile_size
    min_step = math.ceil(raw_min_step / tile_size) * tile_size
    max_step = math.ceil((min_step * 1.5) / tile_size) * tile_size

    # Calculate map size: Snap MAP_SIZE (200) to nearest multiple of tile_size
    # We DO NOT scale the map size with road_scale, we just fit the grid into ~200m
    num_tiles_f = round(MAP_SIZE / tile_size)
    if num_tiles_f < 1: num_tiles_f = 1
    effective_map_size = num_tiles_f * tile_size

    def get_auto_positions():
        raw_positions = [0]
        current_pos = 0
        safety = 0
        
        while current_pos < effective_map_size - tile_size and safety < 100:
            safety += 1
            jump_raw = rng.range(min_step, max_step)
            jump = round(jump_raw / tile_size) * tile_size
            
            next_pos = current_pos + jump
            if next_pos <= effective_map_size - tile_size:
                raw_positions.append(next_pos)
                current_pos = next_pos
            else:
                break
        
        if raw_positions[-1] != effective_map_size - tile_size:
            raw_positions.append(effective_map_size - tile_size)

        valid_positions = [0]
        
        for i in range(1, len(raw_positions) - 1):
            prev = valid_positions[-1]
            curr = raw_positions[i]
            gap = curr - prev - tile_size
            if gap >= needed_gap:
                valid_positions.append(curr)
        
        end_pos = effective_map_size - tile_size
        prev = valid_positions[-1]
        last_gap = end_pos - prev - tile_size
        
        if last_gap < needed_gap:
            if len(valid_positions) > 1:
                valid_positions.pop()
        
        valid_positions.append(end_pos)
        return valid_positions

    return get_auto_positions()

def extract_blocks(v_pos, h_pos, min_area, removed_h_segments=None, removed_v_segments=None, added_h_segments=None, added_v_segments=None, effective_tile_size=None) -> List[Block]:
    """Extract blocks from road grid, merging where segments are removed, then splitting where new segments are added.
    effective_tile_size: The actual road width to use (for road scaling)."""
    if effective_tile_size is None:
        effective_tile_size = TILE_SIZE  # Default to constant
    if removed_h_segments is None: removed_h_segments = []
    if removed_v_segments is None: removed_v_segments = []
    if added_h_segments is None: added_h_segments = []
    if added_v_segments is None: added_v_segments = []
    
    # Convert to sets for O(1) lookup
    h_removed_set = set(removed_h_segments)
    v_removed_set = set(removed_v_segments)
    
    # Create a grid of cells (each cell is a potential block)
    # cells[i][j] represents the block between v_pos[i] and v_pos[i+1], h_pos[j] and h_pos[j+1]
    n_cols = len(v_pos) - 1
    n_rows = len(h_pos) - 1
    
    # Union-Find for merging blocks
    parent = {}
    for i in range(n_cols):
        for j in range(n_rows):
            parent[(i, j)] = (i, j)
    
    def find(cell):
        if parent[cell] != cell:
            parent[cell] = find(parent[cell])
        return parent[cell]
    
    def union(cell1, cell2):
        root1, root2 = find(cell1), find(cell2)
        if root1 != root2:
            parent[root1] = root2
    
    # Merge horizontally adjacent blocks if vertical segment between them is removed
    for i in range(n_cols - 1):
        for j in range(n_rows):
            # Vertical segment between columns i and i+1 at row j
            # This segment would be at x = v_pos[i+1], between h_pos[j] and h_pos[j+1]
            x = v_pos[i + 1]
            y1, y2 = h_pos[j], h_pos[j + 1]
            if (x, y1, y2) in v_removed_set:
                union((i, j), (i + 1, j))
    
    # Merge vertically adjacent blocks if horizontal segment between them is removed
    for i in range(n_cols):
        for j in range(n_rows - 1):
            # Horizontal segment between rows j and j+1 at column i
            # This segment is at y = h_pos[j+1]
            y = h_pos[j + 1]
            x1, x2 = v_pos[i], v_pos[i + 1]
            if (y, x1, x2) in h_removed_set:
                union((i, j), (i, j + 1))
    
    # Group cells by their root
    groups = {}
    for i in range(n_cols):
        for j in range(n_rows):
            root = find((i, j))
            if root not in groups:
                groups[root] = []
            groups[root].append((i, j))
    
    # Create initial merged blocks
    initial_blocks = []
    bid = 0
    
    for root, cells in groups.items():
        # Find bounding box of all cells in this group
        min_i = min(c[0] for c in cells)
        max_i = max(c[0] for c in cells)
        min_j = min(c[1] for c in cells)
        max_j = max(c[1] for c in cells)
        
        # Calculate block coordinates
        x1 = v_pos[min_i] + effective_tile_size
        x2 = v_pos[max_i + 1]
        y1 = h_pos[min_j] + effective_tile_size
        y2 = h_pos[max_j + 1]
        
        w = x2 - x1
        h = y2 - y1
        
        if w <= 1 or h <= 1:
            continue
        
        initial_blocks.append(Block(bid, Rect(x1, y1, w, h), False))
        bid += 1

    # Post-process: Split blocks based on added segments (Staggered Grid)
    final_blocks = []
    queue = list(initial_blocks)
    
    while queue:
        block = queue.pop(0)
        splitted = False
        
        # Check for Horizontal split
        for (y, sx1, sx2) in added_h_segments:
            # Check if segment cuts through the block
            # Since segment includes road width, the road is at [y, y+effective_tile_size]
            # Valid split if road is strictly inside block Y range
            if block.rect.y < y and (y + effective_tile_size) < (block.rect.y + block.rect.h):
                # Check X overlap (simplistic: if segment spans relevant portion)
                if sx1 < block.rect.x + block.rect.w and sx2 > block.rect.x:
                     # Calculate cut
                     # Top Block
                     h1 = y - block.rect.y
                     b1 = Block(bid, Rect(block.rect.x, block.rect.y, block.rect.w, h1), False)
                     bid += 1
                     
                     # Bottom Block
                     y2_new = y + effective_tile_size
                     h2 = (block.rect.y + block.rect.h) - y2_new
                     b2 = Block(bid, Rect(block.rect.x, y2_new, block.rect.w, h2), False)
                     bid += 1
                     
                     queue.append(b1)
                     queue.append(b2)
                     splitted = True
                     break
        
        if splitted: continue
        
        # Check for Vertical split
        for (x, sy1, sy2) in added_v_segments:
            if block.rect.x < x and (x + effective_tile_size) < (block.rect.x + block.rect.w):
                if sy1 < block.rect.y + block.rect.h and sy2 > block.rect.y:
                    # Left Block
                    w1 = x - block.rect.x
                    b1 = Block(bid, Rect(block.rect.x, block.rect.y, w1, block.rect.h), False)
                    bid += 1
                    
                    # Right Block
                    x2_new = x + effective_tile_size
                    w2 = (block.rect.x + block.rect.w) - x2_new
                    b2 = Block(bid, Rect(x2_new, block.rect.y, w2, block.rect.h), False)
                    bid += 1
                    
                    queue.append(b1)
                    queue.append(b2)
                    splitted = True
                    break
        
        if not splitted:
            # Check size constraints one last time
            area = block.rect.w * block.rect.h
            block.too_small = area < (min_area * 0.95)
            final_blocks.append(block)
            
    return final_blocks

def generate_road_tiles(v_pos, h_pos, rng: SeededRNG = None, max_block_size=60, tile_size: float = None, difficulty: int = 1) -> Tuple[List[RoadTile], set, list, list, list, list]:
    """Generate road tiles with random removed segments for T-intersections.
    Returns (tiles, blocked_positions, rm_h, rm_v, add_h, add_v).
    max_block_size: Maximum block dimension before T-junction is prevented.
    tile_size: The effective size of a road tile (default TILE_SIZE).
    difficulty: Impacts shift frequentcy (3 = chaos/more shifts)."""
    if tile_size is None:
        tile_size = TILE_SIZE

    # Effective map size (closest multiple of tile_size)
    num_tiles = round(MAP_SIZE / tile_size)
    effective_map_size = num_tiles * tile_size

    tiles = []
    occupied = set()
    v_set = set(v_pos)
    h_set = set(h_pos)
    h_pos_list = sorted(list(h_pos)) if not isinstance(h_pos, list) else h_pos
    v_pos_list = sorted(list(v_pos)) if not isinstance(v_pos, list) else v_pos
    
    if rng is None:
        rng = SeededRNG(42)
    
    # Step 1: Randomly select HORIZONTAL segments to remove
    # If block is too wide, remove BUT add a shifted segment (staggered)
    removed_h_segments = []
    added_h_segments = [] # List of (y, x1, x2)
    
    for y in h_pos_list:
        if y == h_pos_list[0] or y == h_pos_list[-1]: continue # Skip edges
        
        # Find prev/next y for bounds checking
        idx = h_pos_list.index(y)
        y_prev = h_pos_list[idx-1]
        y_next = h_pos_list[idx+1]
        
        for i in range(len(v_pos_list) - 1):
            x1, x2 = v_pos_list[i], v_pos_list[i+1]
            segment_width = x2 - x1
            
            is_too_wide = segment_width > max_block_size
            
            
            # Probability settings based on Difficulty
            shift_chance = 0.2
            if difficulty == 3: 
                shift_chance = 0.6 # High chaos in Hard Mode
            
            # Modify this segment
            if rng.next_float() < shift_chance:
                if not is_too_wide:
                    # In Hard Mode, we prefer SHIFTS over simple removals (to keep density high)
                    # But standard removal creates plazas.
                    removed_h_segments.append((y, x1, x2))
                else:
                    # "Smart Shift": Remove original, add offset segment
                    margin_check = 2 * tile_size
                    max_shift_up = (y - y_prev) - margin_check
                    max_shift_down = (y_next - y) - margin_check
                    
                    possible_shifts = []
                    # Allow shift if space allows
                    if max_shift_up >= tile_size: possible_shifts.append(-tile_size)
                    if max_shift_up >= 2 * tile_size: possible_shifts.append(-2 * tile_size) # More variance
                    if max_shift_down >= tile_size: possible_shifts.append(tile_size)
                    if max_shift_down >= 2 * tile_size: possible_shifts.append(2 * tile_size)
                    
                    # If grid is very spacious, maybe allow 20? But user wants smaller.
                    # Stick to +/- 10 for consistency and safety.
                    
                    if possible_shifts:
                        shift = rng.choice(possible_shifts)
                        branch_y = y + shift
                        # Add to removals AND additions
                        removed_h_segments.append((y, x1, x2))
                        added_h_segments.append((branch_y, x1, x2))
    
    # Step 2: Randomly select VERTICAL segments to remove
    removed_v_segments = []
    added_v_segments = [] # List of (x, y1, y2)
    
    # Build sets for quick lookup to prevent corners
    h_removed_at_intersection = set() 
    for (y, x1, x2) in removed_h_segments:
        h_removed_at_intersection.add((x1, y))
        h_removed_at_intersection.add((x2, y))
        
    for x in v_pos_list:
        if x == v_pos_list[0] or x == v_pos_list[-1]: continue
        
        idx = v_pos_list.index(x)
        x_prev = v_pos_list[idx-1]
        x_next = v_pos_list[idx+1]
        
        for i in range(len(h_pos_list) - 1):
            y1, y2 = h_pos_list[i], h_pos_list[i+1]
            segment_height = y2 - y1
            
            is_too_tall = segment_height > max_block_size
            
            would_create_corner = ((x, y1) in h_removed_at_intersection or 
                                   (x, y2) in h_removed_at_intersection)
            
            if not would_create_corner and rng.next_float() < shift_chance:
                if not is_too_tall:
                     removed_v_segments.append((x, y1, y2))
                else:
                    # Smart Shift
                    margin_check = 2 * tile_size
                    max_shift_left = (x - x_prev) - margin_check
                    max_shift_right = (x_next - x) - margin_check
                    
                    possible_shifts = []
                    if max_shift_left >= tile_size: possible_shifts.append(-tile_size)
                    if max_shift_left >= 2 * tile_size: possible_shifts.append(-2 * tile_size)
                    if max_shift_right >= tile_size: possible_shifts.append(tile_size)
                    if max_shift_right >= 2 * tile_size: possible_shifts.append(2 * tile_size)
                    
                    if possible_shifts:
                        shift = rng.choice(possible_shifts)
                        branch_x = x + shift
                        removed_v_segments.append((x, y1, y2))
                        added_v_segments.append((branch_x, y1, y2))
    
    # Helper for grid snapping keys
    def to_grid_key(x, y):
        return f"{int(round(x))},{int(round(y))}"

    # Step 3: Build set of blocked road tile positions
    blocked_positions = set()
    for (y, x1, x2) in removed_h_segments:
        # bx loop: range(x1 + tile_size, x2, tile_size)
        start_t = int(round((x1 + tile_size)/tile_size))
        end_t = int(round(x2/tile_size))
        for i in range(start_t, end_t):
            bx = i * tile_size
            blocked_positions.add(to_grid_key(bx, y))
    for (x, y1, y2) in removed_v_segments:
        # by loop: range(y1 + tile_size, y2, tile_size)
        start_t = int(round((y1 + tile_size)/tile_size))
        end_t = int(round(y2/tile_size))
        for i in range(start_t, end_t):
            by = i * tile_size
            blocked_positions.add(to_grid_key(x, by))

    # Helper to check if a point is an endpoint of an added segment
    added_intersections = {} # Key: (x,y), Value: 'T_N', 'T_S', 'T_E', 'T_W' direction of the new road
    
    for (y, x1, x2) in added_h_segments:
        # New road at y (x1..x2). Intersects v_roads at x1 and x2.
        # At x1: New road goes East. So intersection at (x1, y) needs 'E' added.
        added_intersections[(x1, y)] = added_intersections.get((x1, y), "") + "E"
        # At x2: New road goes West.
        added_intersections[(x2, y)] = added_intersections.get((x2, y), "") + "W"
        
    for (x, y1, y2) in added_v_segments:
        # At y1: New road goes South.
        added_intersections[(x, y1)] = added_intersections.get((x, y1), "") + "S"
        # At y2: New road goes North.
        added_intersections[(x, y2)] = added_intersections.get((x, y2), "") + "N"
    
    # Step 4: Build adjacency info for intersections
    def get_neighbors(x, y):
        neighbors = {'N': False, 'S': False, 'E': False, 'W': False}
        
        # Check vertical connectivity (roads going N/S at this x position)
        if x in v_set:
            # North: check if there's road going north
            if y > 0:  # Not at top edge
                # Find next h_road going north
                north_y = None
                for hy in reversed(h_pos_list):
                    if hy < y:
                        north_y = hy
                        break
                if north_y is not None:
                    # Check if segment is removed
                    if (x, north_y, y) not in removed_v_segments:
                        neighbors['N'] = True
                else:
                    # No h_road to north, but still connected to edge
                    neighbors['N'] = True
            
            # South: check if there's road going south  
            if y < effective_map_size - tile_size:  # Not at bottom edge
                south_y = None
                for hy in h_pos_list:
                    if hy > y:
                        south_y = hy
                        break
                if south_y is not None:
                    if (x, y, south_y) not in removed_v_segments:
                        neighbors['S'] = True
                else:
                    neighbors['S'] = True
        
        # Check horizontal connectivity (roads going E/W at this y position)
        if y in h_set:
            # West
            if x > 0:
                west_x = None
                for vx in reversed(v_pos_list):
                    if vx < x:
                        west_x = vx
                        break
                if west_x is not None:
                    if (y, west_x, x) not in removed_h_segments:
                        neighbors['W'] = True
                else:
                    neighbors['W'] = True
            
            # East
            if x < effective_map_size - tile_size:
                east_x = None
                for vx in v_pos_list:
                    if vx > x:
                        east_x = vx
                        break
                if east_x is not None:
                    if (y, x, east_x) not in removed_h_segments:
                        neighbors['E'] = True
                else:
                    neighbors['E'] = True
        
        # Merge with added intersections (from shifted roads)
        added = added_intersections.get((x, y), "")
        if "N" in added: neighbors['N'] = True
        if "S" in added: neighbors['S'] = True
        if "E" in added: neighbors['E'] = True
        if "W" in added: neighbors['W'] = True
        
        return neighbors
    
    def classify_intersection(neighbors):
        n, s, e, w = neighbors['N'], neighbors['S'], neighbors['E'], neighbors['W']
        neighbor_str = ""
        if n: neighbor_str += "N"
        if s: neighbor_str += "S"
        if e: neighbor_str += "E"
        if w: neighbor_str += "W"

        count = sum([n, s, e, w])
        if count == 4: return ('intersection', 0, neighbor_str)
        elif count == 3:
            if not w: return ('t_junction', 0, neighbor_str)
            elif not n: return ('t_junction', 90, neighbor_str)
            elif not e: return ('t_junction', 180, neighbor_str)
            else: return ('t_junction', 270, neighbor_str)
        elif count == 2:
            if (n and s): return ('straight_v', 0, neighbor_str)
            if (e and w): return ('straight_h', 90, neighbor_str)
            if n and w: return ('corner', 270, neighbor_str)
            elif n and e: return ('corner', 0, neighbor_str)
            elif e and s: return ('corner', 90, neighbor_str)
            else: return ('corner', 180, neighbor_str)
        elif count == 1:
            # Dead End - Determine rotation based on the ONLY neighbor
            # We want the "open" side to face the neighbor
            if n: return ('dead_end', 0, neighbor_str)    # Road goes North
            if s: return ('dead_end', 180, neighbor_str)  # Road goes South
            if e: return ('dead_end', 270, neighbor_str)  # Road goes East
            if w: return ('dead_end', 90, neighbor_str)   # Road goes West
            return ('straight_v', 0, neighbor_str) # Should not happen
        else:
            # 0 neighbors - Isolated tile?
            if n or s: return ('straight_v', 0, neighbor_str)
            else: return ('straight_h', 90, neighbor_str)
    
    # Use a dictionary to track tiles by position -> easy updates
    tile_map = {} 
    
    # Step 5: Generate standard intersection tiles
    for x in v_pos:
        for y in h_pos:
            neighbors = get_neighbors(x, y)
            tile_type, rotation, debug_lbl = classify_intersection(neighbors)
            k = to_grid_key(x, y)
            tile_map[k] = RoadTile(x, y, tile_type, rotation, debug_lbl)
            occupied.add(k)
    
    # Step 6: Generate straight road tiles (excluding blocked)
    # Iterate x in v_pos, y from 0 to map_size step tile_size
    num_map_tiles = int(round(effective_map_size / tile_size))

    for x in v_pos:
        for i in range(num_map_tiles):
            y = i * tile_size
            key = to_grid_key(x, y)
            if key not in occupied and key not in blocked_positions:
                tile_map[key] = RoadTile(x, y, 'straight_v', 0)
                occupied.add(key)
    
    for y in h_pos:
        for i in range(num_map_tiles):
            x = i * tile_size
            key = to_grid_key(x, y)
            if key not in occupied and key not in blocked_positions:
                tile_map[key] = RoadTile(x, y, 'straight_h', 90)
                occupied.add(key)

    # Step 7: Generate tiles for ADDED segments (shifted roads)
    for (y, x1, x2) in added_h_segments:
        # bx in range(x1 + tile_size, x2, step=tile_size)
        start_t = int(round((x1 + tile_size)/tile_size))
        end_t = int(round(x2/tile_size))
        for i in range(start_t, end_t):
            bx = i * tile_size
            key = to_grid_key(bx, y)
            # Only add if empty (should be empty as it's inside a block)
            if key not in tile_map:
                tile_map[key] = RoadTile(bx, y, 'straight_h', 90)
                occupied.add(key)
                
    for (x, y1, y2) in added_v_segments:
        # by in range(y1 + tile_size, y2, step=tile_size)
        start_t = int(round((y1 + tile_size)/tile_size))
        end_t = int(round(y2/tile_size))
        for i in range(start_t, end_t):
            by = i * tile_size
            key = to_grid_key(x, by)
            if key not in tile_map:
                tile_map[key] = RoadTile(x, by, 'straight_v', 0)
                occupied.add(key)
                
    # Step 8: Fixup intersections at connection points of added segments
    # These points are where the new roads meet the existing orthogonal networks
    for (x, y), added_dir in added_intersections.items():
        key = to_grid_key(x, y)
        if key in tile_map:
            # Upgrade existing tile
            # It's likely a straight tile. We need to recalculate its type
            # We can't use get_neighbors easily because it assumes grid logic
            # Instead, we just manually upgrade based on road direction
            existing_tile = tile_map[key]
            
            # Determine neighbors based on existing type + added_dir
            neighbors = {'N': False, 'S': False, 'E': False, 'W': False}
            if existing_tile.type == 'straight_v':
                neighbors['N'] = True; neighbors['S'] = True
            elif existing_tile.type == 'straight_h':
                neighbors['E'] = True; neighbors['W'] = True
            elif existing_tile.type == 'intersection':
                neighbors['N'] = True; neighbors['S'] = True; neighbors['E'] = True; neighbors['W'] = True
            elif existing_tile.type == 't_junction':
                # Already a T? Just add the missing one (making it 4-way)
                rot = existing_tile.rotation
                if rot == 0: neighbors.update({'N':True, 'S':True, 'E':True})
                elif rot == 90: neighbors.update({'S':True, 'E':True, 'W':True}) 
                elif rot == 180: neighbors.update({'N':True, 'S':True, 'W':True}) # 180 is T facing West (N,S,W)
                elif rot == 270: neighbors.update({'N':True, 'E':True, 'W':True})
                
            # Add the new connection
            if "N" in added_dir: neighbors['N'] = True
            if "S" in added_dir: neighbors['S'] = True
            if "E" in added_dir: neighbors['E'] = True
            if "W" in added_dir: neighbors['W'] = True
            
            new_type, new_rot, _ = classify_intersection(neighbors)
            tile_map[key] = RoadTile(x, y, new_type, new_rot, "FIXUP")

    # Step 9: Roundabout Upgrade (Post-Processing)
    # Safely convert some intersections to roundabouts if they have space
    roundabout_candidates = []
    
    # 1. Identify candidates (Must be 4-way intersections)
    for k, tile in tile_map.items():
        if tile.type == 'intersection':
            roundabout_candidates.append(tile)
            
    # 2. Shuffle to randomize
    rng.choice(roundabout_candidates) # Just shuffle implicitly or use rng
    # Since rng.choice picks one, we need a shuffle method. 
    # We can just iterate and use chance.
    
    roundabout_min_dist = 3 * tile_size # Minimum distance between roundabouts
    existing_roundabouts = []
    
    for tile in roundabout_candidates:
        # Chance to convert (e.g. 30%)
        if rng.next_float() > 0.3: continue
        
        # Safety Check 1: Neighbors must be STRAIGHT (Buffer Zone)
        # We need to look up neighbors in tile_map
        # Neighbors: (x, y-size), (x, y+size), (x-size, y), (x+size, y)
        ts = tile_size
        n_positions = [
            (tile.x, tile.y - ts), (tile.x, tile.y + ts), 
            (tile.x - ts, tile.y), (tile.x + ts, tile.y)
        ]
        
        safe = True
        for nx, ny in n_positions:
            nk = to_grid_key(nx, ny)
            if nk not in tile_map:
                safe = False # Edge of map or void
                break
            neighbor = tile_map[nk]
            # Neighbor MUST be a straight road. No corners, no T's, no other intersections close by.
            if neighbor.type not in ['straight_v', 'straight_h']:
                safe = False
                break
                
        # Safety Check 2: Distance from other roundabouts
        if safe:
            for ex_r in existing_roundabouts:
                dist = math.sqrt((tile.x - ex_r.x)**2 + (tile.y - ex_r.y)**2)
                if dist < roundabout_min_dist:
                    safe = False
                    break
        
        if safe:
            tile.type = 'roundabout'
            existing_roundabouts.append(tile)
            
            # EXPANSION: Remove the 4 neighbor tiles to make space for the large roundabout model (3x3 area)
            # This prevents overlap. We verified they are straight roads, so safe to remove.
            # However, removing them might isolate other tiles? 
            # No, 'dead_end' logic runs BEFORE this. But visual continuity is fine.
            # Better approach: We KEEP them in logic but mark them as "consumed" so they don't render?
            # Or just delete them from tile_map. If we delete, they won't be in the final list.
            
            for nx, ny in n_positions:
                nk = to_grid_key(nx, ny)
                if nk in tile_map:
                    # instead of deleting (which makes space "buildable"), mark as reserved road
                    tile_map[nk].type = 'roundabout_arm'
                    # tile_map[nk].rotation stays same, irrelevant as it won't render

    tiles = list(tile_map.values())
    
    # Step 10: Pedestrian Crossings (Post-Processing)
    # Convert some straight roads to crossings if they are far from junctions
    crossing_candidates = [t for t in tiles if t.type in ['straight_v', 'straight_h']]
    rng.choice(crossing_candidates) # Shuffle
    
    for tile in crossing_candidates:
        # Chance (e.g. 15%)
        if rng.next_float() > 0.15: continue
        
        # Safety Check: Neighbors must NOT be junctions
        # We want at least 1 tile buffer from any intersection/corner/roundabout
        ts = tile_size
        n_positions = [
            (tile.x, tile.y - ts), (tile.x, tile.y + ts), 
            (tile.x - ts, tile.y), (tile.x + ts, tile.y)
        ]
        
        safe = True
        for nx, ny in n_positions:
            nk = to_grid_key(nx, ny)
            if nk in tile_map:
                neighbor_type = tile_map[nk].type
                # If neighbor is anything other than straight, it's unsafe
                if neighbor_type not in ['straight_v', 'straight_h']:
                    safe = False
                    break
        
        if safe:
            tile.type = 'crossing'

    tiles = list(tile_map.values())
    return tiles, blocked_positions, removed_h_segments, removed_v_segments, added_h_segments, added_v_segments



# ============================================================================
# BUILDING GENERATION
# ============================================================================
TEMPLATES = {
    # STRICT SIZE RANGES - NO OVERLAP!
    # House: 12-15m
    'house': [
        {'w': 12, 'd': 12}, {'w': 13, 'd': 13}, {'w': 14, 'd': 14}, {'w': 15, 'd': 15}
    ],
    # Apt: 18-24m (Distinct from House and Tower)
    'apt': [
        {'w': 18, 'd': 18}, {'w': 20, 'd': 20}, {'w': 22, 'd': 22}, {'w': 24, 'd': 24}
    ],
    # Tower: 28m+ (Strictly large footprint)
    'tower': [
        {'w': 28, 'd': 28}, {'w': 32, 'd': 32}, 
        {'w': 36, 'd': 36}, {'w': 40, 'd': 40}
    ]
}

def generate_buildings(blocks: List[Block], rng: SeededRNG, city_type: int = 2, b_margin=4, difficulty: int = 1) -> List[Building]:
    """
    Generate buildings with zoning logic and JUSTIFIED spacing.
    city_type: 1=Residential (Houses), 2=Mixed (Houses/Apts), 3=Downtown (Apts/Towers).
    difficulty: 1=Easy (normal), 2=Medium (tighter, more towers), 3=Hard (no margins, towers only).
    Adaptive Margin: Tries comfortable settings first, then squeezes.
    """
    buildings = []
    occupied_rects = []
    print(f"🔧 generate_buildings called with difficulty={difficulty}, city_type={city_type}")
    min_gap = 1.5  # Increased from 0.5 to prevent touching/clipping (Safety Margin)
    if difficulty == 3:
        min_gap = 1.0 # Still tight for Hard Mode, but safe from clipping
    
    # DIFFICULTY-BASED SETTINGS (Safe margins, NO center gap)
    if difficulty == 1:  # Easy - Safe margins
        base_road_margin = 2   # Reduced from 4 -> Closer to street
        base_center_gap = 0    # No gap between rows
        squeeze_road_margin = 1.5
        squeeze_center_gap = 0
        ultra_road_margin = 1.5
        ultra_center_gap = 0
    elif difficulty == 2:  # Medium - Tighter
        base_road_margin = 3
        base_center_gap = 0
        squeeze_road_margin = 3
        squeeze_center_gap = 0
        ultra_road_margin = 3
        ultra_center_gap = 0
    else:  # Hard (difficulty == 3)
        base_road_margin = 3
        base_center_gap = 0
        squeeze_road_margin = 3
        squeeze_center_gap = 0
        ultra_road_margin = 3
        ultra_center_gap = 0
    
    def rects_overlap(r1: Rect, r2: Rect) -> bool:
        return not (r1.x2 <= r2.x or r2.x2 <= r1.x or r1.y2 <= r2.y or r2.y2 <= r1.y)
    
    def can_place(new_rect: Rect, current_occupied: List[Rect]) -> bool:
        for existing in current_occupied:
            if rects_overlap(new_rect, existing):
                return False
        return True
    
    def get_category_for_block(block_h, rng_inst, c_type, diff=1, block_tier=None):
        """
        Select ONE building category for entire block - NO MIXING!
        block_tier: 'house', 'apt', or 'tower' - all buildings in this block will be this type.
        """
        # HARD mode: Always towers!
        if diff == 3:
            return 'tower'
        
        # USE block_tier DIRECTLY - it's already 'house', 'apt', or 'tower'
        if block_tier in ['house', 'apt', 'tower']:
            return block_tier
        
        # FALLBACK: Random selection based on city_type
        rand = rng_inst.next_float()
        
        if c_type == 1:  # RESIDENTIAL: Mostly houses
            if rand < 0.7: return 'house'
            return 'apt'
        elif c_type == 3:  # URBAN: Mostly towers
            if rand < 0.5: return 'tower'
            return 'apt'
        else:  # MIXED
            if rand < 0.33: return 'house'
            elif rand < 0.66: return 'apt'
            return 'tower'

    def fill_row_justified(start_val, end_val, fixed_val, depth, axis, category, facing, rng_inst, current_builds, current_occupied):
        length = end_val - start_val
        if length < 5: return
        
        candidates = []
        current_used = 0
        
        valid_tmpls = [t for t in TEMPLATES.get(category, TEMPLATES['house']) if t['d'] == depth]
        if not valid_tmpls: 
            # In Hard Mode (difficulty=3), DO NOT FALLBACK to House/Apt.
            if difficulty == 3: return
            valid_tmpls = TEMPLATES.get(category, TEMPLATES['house'])
        
        # 1. Selection
        # Safety break to avoid infinite loops
        safety = 0
        while safety < 50:
            safety += 1
            tmpl = rng_inst.choice(valid_tmpls)
            if current_used + tmpl['w'] > length: break
            candidates.append(tmpl)
            current_used += tmpl['w'] + min_gap
        
        if not candidates: return

        # 2. Tight Packing (NOT justified: no wasted gaps)
        # Buildings placed side-by-side with minimal gap
        # Slack remains at the end, not distributed
        final_gap = min_gap  # Constant tight gap (2.0m)
            
        # 3. Placement (start from edge, not centered)
        curr_pos = start_val + min_gap  # Small padding from edge
        
        for tmpl in candidates:
            if axis == 'x':
                r = Rect(curr_pos, fixed_val, tmpl['w'], depth)
            else:
                r = Rect(fixed_val, curr_pos, depth, tmpl['w'])
            
            if can_place(r, current_occupied):
                current_builds.append(Building(r, category, facing=facing))
                current_occupied.append(r)
            
            curr_pos += tmpl['w'] + final_gap

    def process_block(block, road_margin=4, center_gap=1.5, block_tier='commercial'):
        """Returns (buildings_list, score). Score: 2 for double row, 1 for single.
        road_margin: Fixed distance from road to buildings.
        center_gap: Gap between North and South rows (can be reduced for squeeze).
        block_tier: 'residential' (house+apt) or 'commercial' (apt+tower)
        """
        local_builds = []
        local_occupied = [] # Local tracking for this trial
        
        # We need to check against GLOBAL occupied_rects too? 
        # Ideally blocks are disjoint so we don't need to, but to be safe we could pass it.
        # But for 'score' comparison, we just simulate.
        # Actually, global overlap is rare due to road gaps. Let's assume disjoint blocks.
        
        valid_rect = Rect(
            block.rect.x + road_margin,
            block.rect.y + road_margin,
            max(0, block.rect.w - road_margin*2),
            max(0, block.rect.h - road_margin*2)
        )
        
        if valid_rect.w < 1 or valid_rect.h < 1:
            # print(f"DEBUG: Block too small after margin: {valid_rect.w:.1f}x{valid_rect.h:.1f}")
            return [], 0
        
        category = get_category_for_block(valid_rect.h, rng, city_type, difficulty, block_tier=block_tier)
        
        # --- PASS 1: NORTH ---
        ref_tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES['house']))
        north_depth = ref_tmpl['d']
        
        fill_row_justified(
            valid_rect.x, valid_rect.x2 - 0.1, 
            valid_rect.y, north_depth, 'x', category, 0, rng, local_builds, local_occupied
        )
        
        has_north = len(local_builds) > 0
        
        # --- PASS 2: SOUTH ---
        south_depth = 0
        available_depth = valid_rect.h - north_depth - center_gap
        has_south = False
        
        if available_depth > 8:
            # SINGLE CATEGORY per block (no mixing)
            cat_chain = [category]  # Only use the block's assigned category
            
            selected_cat = None
            selected_ref = None
            for cat in cat_chain:
                tmpls = TEMPLATES.get(cat, [])
                valid_tmpls = [t for t in tmpls if t['d'] <= available_depth]
                if valid_tmpls:
                    selected_cat = cat
                    selected_ref = rng.choice(valid_tmpls)
                    break
            
            if selected_cat and selected_ref:
                south_depth = selected_ref['d']
                y_pos = valid_rect.y2 - south_depth
                prev_count = len(local_builds)
                fill_row_justified(
                    valid_rect.x, valid_rect.x2 - 0.1, 
                    y_pos, south_depth, 'x', selected_cat, 180, rng, local_builds, local_occupied
                )
                if len(local_builds) > prev_count: has_south = True

        # --- PASS 3 & 4: WEST/EAST ---
        start_y = valid_rect.y + north_depth + center_gap
        end_y = valid_rect.y2 - south_depth - center_gap
        
        west_depth = 0
        east_depth = 0
        if end_y - start_y > 4:
            # West (use block's category, not hardcoded 'house')
            ref_tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES['house']))
            west_depth = ref_tmpl['d']
            fill_row_justified(start_y, end_y, valid_rect.x, ref_tmpl['d'], 'y', category, 270, rng, local_builds, local_occupied)
            # East (use block's category)
            ref_tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES['house']))
            east_depth = ref_tmpl['d']
            x_pos = valid_rect.x2 - ref_tmpl['d']
            fill_row_justified(start_y, end_y, x_pos, ref_tmpl['d'], 'y', category, 90, rng, local_builds, local_occupied)

        # --- CENTER intentionally left EMPTY ---
        # Buildings only on edges, facing streets (backs toward each other)
        # This is realistic - no buildings in block center

        score = 0
        if has_north: score += 1
        if has_south: score += 1
        
        return local_builds, score

    def process_block_centered(block, road_margin=4, block_tier='commercial'):
        """Try to place ONE row of buildings in the CENTER of the block."""
        local_builds = []
        valid_rect = Rect(block.rect.x + road_margin, block.rect.y + road_margin, block.rect.w - 2*road_margin, block.rect.h - 2*road_margin)
        if valid_rect.w <= 0 or valid_rect.h <= 0: return []
        
        category = get_category_for_block(valid_rect.h, rng, city_type, difficulty, block_tier=block_tier)        

        # Determine orientation
        is_horizontal = valid_rect.w > valid_rect.h
        local_occupied = []

        if is_horizontal:
             # Center Y
             cy = valid_rect.y + valid_rect.h/2
             # Use a template to determine depth
             tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES['house']))
             depth = tmpl['d']
             y_pos = cy - depth/2
             # Fill Row
             fill_row_justified(valid_rect.x, valid_rect.x2, y_pos, depth, 'x', category, 0, rng, local_builds, local_occupied)
        else:
             # Vertical - Center X
             cx = valid_rect.x + valid_rect.w/2
             tmpl = rng.choice(TEMPLATES.get(category, TEMPLATES['house']))
             depth = tmpl['d']
             x_pos = cx - depth/2
             fill_row_justified(valid_rect.y, valid_rect.y2, x_pos, depth, 'y', category, 90, rng, local_builds, local_occupied)
             
        return local_builds

    def process_block_chaotic(block):
        """HARD MODE: Random chaotic fill - no rows, no gaps, pure chaos!"""
        local_builds = []
        local_occupied = []
        
        # No margins in chaos mode!
        valid_rect = Rect(block.rect.x, block.rect.y, block.rect.w, block.rect.h)
        if valid_rect.w < 10 or valid_rect.h < 10: return []
        
        # Always use tower templates for hard mode
        tower_tmpls = TEMPLATES.get('tower', [{'w': 25, 'd': 25}])
        
        # Try to fill the ENTIRE block with random placements
        attempts = 0
        max_attempts = 100  # Safety limit
        
        while attempts < max_attempts:
            attempts += 1
            
            # Pick random template
            tmpl = rng.choice(tower_tmpls)
            tw, td = tmpl['w'], tmpl['d']
            
            # Random position within block
            if valid_rect.w - tw < 1 or valid_rect.h - td < 1:
                continue
            
            rx = valid_rect.x + rng.next_float() * (valid_rect.w - tw)
            ry = valid_rect.y + rng.next_float() * (valid_rect.h - td)
            
            # Create rect for this placement
            new_rect = Rect(rx, ry, tw, td)
            
            # Check if it overlaps with existing buildings
            if can_place(new_rect, local_occupied):
                # Random rotation (0, 90, 180, 270)
                facing = rng.choice([0, 90, 180, 270])
                local_builds.append(Building(new_rect, 'tower', facing=facing))
                local_occupied.append(new_rect)
        
        return local_builds

    # MAIN LOOP
    for block in blocks:
        # if block.too_small: logic removed per user request

        # ASSIGN PURE CATEGORY FOR ENTIRE BLOCK - NO MIXING!
        # Each block gets ONE type: 'house', 'apt', or 'tower'
        tier_rand = rng.next_float()
        if city_type == 1:  # RESIDENTIAL
            block_tier = 'house' if tier_rand < 0.7 else 'apt'
        elif city_type == 3:  # URBAN
            block_tier = 'tower' if tier_rand < 0.5 else 'apt'
        else:  # MIXED
            if tier_rand < 0.33: block_tier = 'house'
            elif tier_rand < 0.66: block_tier = 'apt'
            else: block_tier = 'tower'
        
        # For HARD mode, override city_type to force towers
        effective_city_type = 3 if difficulty == 3 else city_type
        
        # HARD MODE: Use chaotic random fill instead of edge rows!
        if difficulty == 3:
            final_builds = process_block_chaotic(block)
            buildings.extend(final_builds)
            for b in final_builds:
                occupied_rects.append(b.rect)
            continue  # Skip normal processing
        
        # 1. Try with base settings (depends on difficulty)
        builds_4, score_4 = process_block(block, road_margin=base_road_margin, center_gap=base_center_gap, block_tier=block_tier)
        
        # 2. Decide if we should squeeze CENTER GAP
        # Squeeze if we failed to get a double row (score < 2)
        final_builds = builds_4
        
        if score_4 < 2:
            # Try squeeze settings
            builds_2, score_2 = process_block(block, road_margin=squeeze_road_margin, center_gap=squeeze_center_gap, block_tier=block_tier)
            # Use squeeze if it gets us 2 rows OR more buildings
            if score_2 >= 2 or len(builds_2) > len(builds_4):
                final_builds = builds_2
                score_4 = score_2  # Update score for next check
            
            # 3. ULTRA SQUEEZE - Try even tighter if still not 2 rows
            if score_4 < 2:
                builds_1, score_1 = process_block(block, road_margin=ultra_road_margin, center_gap=ultra_center_gap, block_tier=block_tier)
                if score_1 >= 2 or len(builds_1) > len(final_builds):
                    final_builds = builds_1
            
            # 4. CENTER STRATEGY - Use when only ONE row fits (score < 2)
            # This centers buildings in narrow blocks instead of placing on edge
            if score_4 < 2 or len(final_builds) == 0:
                 builds_center = process_block_centered(block, road_margin=base_road_margin, block_tier=block_tier)
                 if len(builds_center) > 0:
                     # Use centered layout for narrow blocks
                     final_builds = builds_center
        
        # Fallback to Park REMOVED as per user request ("No trees")
        # if not final_builds: ...


        
        # Add to main list
        # Note: We need to register occupied_rects globally to use can_place globally if needed?
        # But process_block uses local_occupied. 
        # Overlaps between blocks are handled by road width (10m).
        # So it's safe to just extend.
        buildings.extend(final_builds)
        for b in final_builds:
            occupied_rects.append(b.rect)

    return buildings

# ============================================================================
# MAIN INTERFACE
# ============================================================================
def generate_city(seed=42, min_spacing=20, target_area=1000, city_type=2, max_block_size=2000, difficulty=1, road_scale=1.0) -> Dict:
    """Generate a city with road tiles.
    max_block_size: Maximum block area in sq meters before T-junction is prevented.
    difficulty: 1=Easy, 2=Medium, 3=Hard - affects margins and building types.
    road_scale: Scale factor for road width (0.5=narrow, 1.0=normal, 1.5=wide)."""
    rng = SeededRNG(seed)
    
    # Effective tile size based on road scale
    effective_tile_size = TILE_SIZE * road_scale
    
    # Calculate max dimension from max area (assuming roughly square blocks)
    import math
    max_block_dimension = int(math.sqrt(max_block_size))
    
    # Grid - generated with effective_tile_size to match road scale
    v_pos = generate_road_positions(rng, min_spacing, target_area, tile_size=effective_tile_size)
    h_pos = generate_road_positions(rng, min_spacing, target_area, tile_size=effective_tile_size)
    
    # Objects - road tiles with random T-intersections
    tiles, blocked_positions, rm_h, rm_v, add_h, add_v = generate_road_tiles(v_pos, h_pos, rng, max_block_dimension, tile_size=effective_tile_size, difficulty=difficulty)
    blocks = extract_blocks(v_pos, h_pos, 400, rm_h, rm_v, add_h, add_v, effective_tile_size=effective_tile_size)
    buildings = generate_buildings(blocks, rng, city_type=city_type, difficulty=difficulty)
    
    # POST-BUILDING FILTER: Remove buildings that crash into ANY Road (Roundabouts, Arms, Intersections)
    # This acts as a final safety layer to prevent clipping.
    
    safe_buildings = []
    
    # Combined Filter: AABB (All Roads) + Diamond (Roundabouts)
    
    # Pre-calculate road rects
    road_rects = []
    margin = 0.5 
    for t in tiles:
        road_rects.append((t.x - margin, t.y - margin, t.x + effective_tile_size + margin, t.y + effective_tile_size + margin))
        
    # Pre-calculate Roundabout Centers
    roundabout_centers = []
    for t in tiles:
        if t.type == 'roundabout':
            cx = t.x + effective_tile_size / 2
            cy = t.y + effective_tile_size / 2
            roundabout_centers.append((cx, cy))
            
    # Diamond Radius (Manhattan Distance Threshold)
    # The user suggested connecting the ends of the arms.
    # Arms extend 1 tile out (reserved). Center to Edge of neighbor is 1.5 tiles.
    # Let's be safe and use 1.9 tiles (19m) to clean the corners.
    diamond_limit = effective_tile_size * 1.9

    for b in buildings:
        # Building corners
        corners = [
            (b.rect.x, b.rect.y),
            (b.rect.x + b.rect.w, b.rect.y),
            (b.rect.x, b.rect.y + b.rect.h),
            (b.rect.x + b.rect.w, b.rect.y + b.rect.h)
        ]
        
        is_safe = True
        
        # 1. Diamond Check (Manhattan Distance) - The "Rotated Square"
        # |x - cx| + |y - cy| < Limit
        for (rcx, rcy) in roundabout_centers:
            for (bx, by) in corners:
                manhattan_dist = abs(bx - rcx) + abs(by - rcy)
                if manhattan_dist < diamond_limit:
                    is_safe = False
                    break
            if not is_safe: break
        if not is_safe: continue

        # 2. AABB Check (All Roads)
        bx1, by1 = b.rect.x, b.rect.y
        bx2, by2 = b.rect.x + b.rect.w, b.rect.y + b.rect.h
        
        for (rx1, ry1, rx2, ry2) in road_rects:
            if not (bx1 >= rx2 or bx2 <= rx1 or by1 >= ry2 or by2 <= ry1):
                is_safe = False
                break
        
        if is_safe:
            safe_buildings.append(b)
            
    buildings = safe_buildings
    
    return {
        'seed': seed,
        'roads': tiles,
        'blocks': blocks,
        'buildings': buildings
    }
