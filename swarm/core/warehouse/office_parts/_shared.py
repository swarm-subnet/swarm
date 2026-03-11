"""
Embedded office room for the warehouse map.
Self-contained 12x12m room placed inside the OFFICE zone with walls, furniture,
workstations, filing shelves, services area, and center meeting table.
Uses Kenney furniture-kit OBJ models loaded via temporary URDFs.
"""

import math
import os
import random
import tempfile
from collections import OrderedDict

import pybullet as p

try:
    from PIL import Image, ImageChops, ImageDraw, ImageOps
except Exception:
    Image = None
    ImageDraw = None
    ImageOps = None
    ImageChops = None

from ..constants import (
    ASSETS_DIR,
    DOCK_INWARD_NUDGE,
    EMBEDDED_OFFICE_SEED_OFFSET,
    ENABLE_EMBEDDED_OFFICE_MAP,
    FURNITURE_KIT_OBJ_DIR,
    MESH_UP_FIX_RPY,
    WALL_SLOTS,
)
from ..helpers import slot_point

_SWARM_ASSETS_DIR = os.path.normpath(os.path.join(ASSETS_DIR, os.pardir))

FLOOR_SIZE = 12.0
UNIFORM_SCALE = 2.0
ROOM_CENTER = (0.0, 0.0)
SCREEN_BRANDING_ENABLED = True
SCREEN_BRANDING_LABEL = "SWARM"
SCREEN_TEXTURE_ROTATE_DEG = 180

ASSET_PATH = FURNITURE_KIT_OBJ_DIR
TEMP_URDF_DIR = os.path.join(tempfile.gettempdir(), "swarm_warehouse_office_urdfs")
SCREEN_LOGO_CANDIDATES = (
    os.path.join(_SWARM_ASSETS_DIR, "Swarm.png"),
    os.path.join(_SWARM_ASSETS_DIR, "Swarm_2.png"),
)

ASSETS = {
    "floor_tile": "floorFull.obj",
    "wall": "wall.obj",
    "wall_door": "wallDoorway.obj",
    "wall_corner": "wallCorner.obj",
    "doorway": "doorwayOpen.obj",
    "meeting_table": "tableCrossCloth.obj",
    "meeting_chair": "chairDesk.obj",
    "entry_coat_rack": "coatRackStanding.obj",
    "entry_plant": "pottedPlant.obj",
    "desk": "desk.obj",
    "desk_corner": "deskCorner.obj",
    "desk_chair": "chairDesk.obj",
    "monitor": "computerScreen.obj",
    "keyboard": "computerKeyboard.obj",
    "mouse": "computerMouse.obj",
    "bookcase_open": "bookcaseOpen.obj",
    "bookcase_open_low": "bookcaseOpenLow.obj",
    "bookcase_closed": "bookcaseClosed.obj",
    "bookcase_closed_doors": "bookcaseClosedDoors.obj",
    "bookcase_wide": "bookcaseClosedWide.obj",
    "books": "books.obj",
    "box": "cardboardBoxClosed.obj",
    "box_open": "cardboardBoxOpen.obj",
    "fridge": "kitchenFridgeSmall.obj",
    "fridge_tall": "kitchenFridge.obj",
    "fridge_large": "kitchenFridgeLarge.obj",
    "cabinet": "kitchenCabinet.obj",
    "cabinet_tv_doors": "cabinetTelevisionDoors.obj",
    "coffee_machine": "kitchenCoffeeMachine.obj",
    "trashcan": "trashcan.obj",
    "side_table": "sideTable.obj",
    "plant_small": "plantSmall1.obj",
    "plant_small_2": "plantSmall2.obj",
    "plant_small_3": "plantSmall3.obj",
    "table_coffee": "tableCoffee.obj",
    "table_coffee_square": "tableCoffeeSquare.obj",
    "lamp_table_round": "lampRoundTable.obj",
    "lamp_table_square": "lampSquareTable.obj",
    "lamp_floor": "lampRoundFloor.obj",
}

WALL_ROLES = ("entry", "workstations", "files", "services")
ENABLE_PERIMETER_WALL_MESHES = True
ENABLE_PERIMETER_WALL_CORNERS = True
ENTRY_WALL_OPENING_MODE = "door_segment"
ENTRY_WALL_OPENING_ALONG = 0.0
PERIMETER_WALL_ALONG_SCALE = 1.0
PERIMETER_WALL_CORNER_OUTWARD_EPS = 0.0
PERIMETER_WALL_CORNER_JOIN_GAP_M = 0.0
OFFICE_WALL_FORCE_FLAT_COLOR = True
OFFICE_WALL_FLAT_RGBA = (0.72, 0.74, 0.78, 1.0)
OFFICE_FLOOR_FORCE_FLAT_COLOR = True
OFFICE_FLOOR_FLAT_RGBA = (0.64, 0.66, 0.70, 1.0)

ASSET_FRONT_DEG = {
    "chairDesk.obj": 90.0,
    "desk.obj": 90.0,
    "deskCorner.obj": 90.0,
    "computerScreen.obj": 90.0,
    "computerKeyboard.obj": 90.0,
    "computerMouse.obj": 90.0,
    "bookcaseOpen.obj": 90.0,
    "bookcaseClosed.obj": 90.0,
    "bookcaseClosedDoors.obj": 90.0,
    "bookcaseClosedWide.obj": 90.0,
    "kitchenFridgeSmall.obj": 90.0,
    "kitchenFridge.obj": 90.0,
    "kitchenFridgeLarge.obj": 90.0,
    "kitchenCabinet.obj": 90.0,
    "cabinetTelevisionDoors.obj": 90.0,
    "kitchenCoffeeMachine.obj": 90.0,
    "trashcan.obj": 90.0,
    "pottedPlant.obj": 90.0,
    "plantSmall1.obj": 90.0,
    "plantSmall2.obj": 90.0,
    "plantSmall3.obj": 90.0,
    "coatRackStanding.obj": 90.0,
    "sideTable.obj": 90.0,
    "tableCoffee.obj": 90.0,
    "tableCoffeeSquare.obj": 90.0,
    "lampRoundTable.obj": 90.0,
    "lampSquareTable.obj": 90.0,
    "lampRoundFloor.obj": 90.0,
    "cardboardBoxClosed.obj": 90.0,
    "cardboardBoxOpen.obj": 90.0,
    "bookcaseOpenLow.obj": 90.0,
    "books.obj": 90.0,
}

__all__ = [name for name in globals() if not name.startswith("__")]
