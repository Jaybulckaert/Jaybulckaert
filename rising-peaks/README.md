# Rising Peaks

One page, two photos, a switcher. Each photo is split into plates so that only the background mountains scale.

- **Banff Avenue** (1600 : 1, one mountain plate): plates live in `../scale-bench/`.
- **Recreation Grounds** (1400 : 1, two mountain plates): plates live in `../park-peaks/`.

Each mountain plate scales about its own base line under its summit, from the photo's reading up to 5000 : 1 against your height. Back and front plates never transform. The page remembers the last photo you had open.

## Runway renders

`renders/` holds image-to-video passes generated on Runway, anchored on the untouched photo as the start frame and on the 5000 : 1 composite as the end frame, re-encoded to 1280 × 960 H.264 for the web. The **Runway render** button on the page swaps the interactive canvas for the clip. `recreation-grounds-runway.mp4` is in; the Banff Avenue clip slot is wired and waits on the file.

Open `index.html` in a browser from this folder so the relative paths to the sibling folders resolve. No build step.
