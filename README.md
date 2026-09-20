# bpp-sun

3D sun and shadow simulation of Beckenham Place Park, for finding the bench that is actually in the sun.

https://radiosilence.github.io/bpp-sun/

Drag the map, scrub the date and time, and the shadows move. Benches in direct sun glow; the list ranks them by share of daylight spent in sun. Selecting a bench shows its whole year. With location allowed, it finds the nearest bench that is sunny now and will stay so for half an hour.

## Why it is built this way

**Shadows come from LIDAR, not from modelled trees.** The Environment Agency's 1m first-return surface model contains every real crown and roof, so a ray marched across it gives shadows for the trees that exist rather than for cones placed at guessed positions. The tree meshes in the 3D view are decoration, seeded from canopy peaks; they cast nothing.

**One march, two implementations.** The ground shadows are a GLSL pass over the surface model, rerun when the sun moves. Bench statistics use the same march in JS (`shadeAt`), because 57 benches × 288 five-minute steps is trivial on the CPU and avoids GPU readback. The two must stay in step; both live next to a comment saying so.

**No precomputed results.** Any date is computed in the browser on demand, so the shipped data is just three height layers (~5MB gzipped) and a scene JSON.

**Crowns have an underside.** A plain 2.5D surface treats a tree as a solid column to the ground, which wrongly shades every bench beside a trunk at low sun. Each canopy cell carries an estimated crown-base height (40% of the local canopy height), and rays may travel beneath it.

**No build step.** ES modules and an import map; three.js comes from a CDN. `site/` is deployed as-is.

## Known lies

- LIDAR is flown leaf-off. Crowns are morphologically closed to approximate summer leaf, so **winter tree shade is overstated** — most of this canopy is bare from November to March. Tree shade is reported separately from building/terrain shade for that reason.
- A ray may pass under canopy for 30m before it is assumed blocked by trunks and understorey. That number is a judgement, not a measurement.
- Building heights are the 90th percentile of LIDAR returns inside the OSM footprint, drawn with flat roofs. This affects the picture only; shadows use the raw surface.
- Bench positions and orientation come from OSM. Unmapped benches don't exist here, and benches without a `direction` tag are drawn facing the nearest path.
- Direct-beam geometry only: no cloud, no diffuse light.
- Solar position is the low-precision Meeus series, good to a fraction of a degree.

## Data

```
mise run data    # fetch LIDAR + OSM into pipeline/raw, write site/data
mise run serve   # http://localhost:8765
```

`pipeline/build.py` is a PEP 723 script; `uv` resolves its dependencies. Raw downloads are cached in `pipeline/raw/` and not committed. To cover a different park, change the extent and `PARK_WAY` at the top of the script.

Coordinates are local metres on the British National Grid from the extent's south-west corner. Solar azimuths are rotated by the grid convergence (−1.55° here) before use. GPS fixes are placed with an affine fit emitted by the pipeline (≤20cm error over the extent), which saves shipping a projection library.

## Attribution

Contains public sector information licensed under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) — LIDAR Composite DSM/DTM © Environment Agency. Map data © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors, ODbL.
