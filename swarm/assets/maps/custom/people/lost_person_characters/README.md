# Lost-person character assets

This package contains 32 character targets intended for Search and Rescue
tasks and lost-person simulation scenarios in Swarm.

Each numbered character directory preserves the supplied package structure:

- `asset_files/` contains the OBJ, MTL, textures, and any supplied GLB files.
- `previews/` contains front, side, and back reference renders.
- `README.md` contains the supplied visual description and scale.

Use the scale recorded for each character in `manifest.json`. Keep each OBJ
beside its MTL and referenced textures so material paths continue to resolve.

## Origin and license status

The manifest uses two origin categories:

- `AI-generated/modified`: generated with AI, with some assets subsequently
  modified. These are grouped together because the package does not identify
  the amount of modification for each individual asset.
- `open-source`: characters sourced from Quaternius under CC0 1.0. The
  applicable characters are identified in `manifest.json`; see
  `SOURCE.txt` for the source notice.

The collection is mixed-origin and is not attributed as a whole to any
individual. Original third-party licenses and attribution requirements remain
applicable to their respective assets.
