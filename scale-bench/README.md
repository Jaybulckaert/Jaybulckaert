# Cascade Grows

One photo of Cascade Mountain from Banff Avenue, split into three plates so that only the mountain scales.

- **Back plate** `banff-cascade.jpg`: the untouched photograph.
- **Mountain plate** `mountain.webp`: Cascade and its treed flanks, cut along the ridge against the sky, with rock texture reconstructed behind the spruce, roofs and town treeline so no hole opens when it moves.
- **Front plate** `foreground.webp`: everything nearer than the mountain: the big spruce, streetlights, houses, street.

`index.html` composites the plates on a canvas. The mountain plate is scaled about its base under the summit, from 1600 : 1 to 5000 : 1 against your height (magnification `5000 / 1600 = 3.125`). Back and front plates never transform.

`pipeline/build_layers.py` is the OpenCV script that produced the plates: GrabCut seeded with hand-traced polygons for the two boundaries, then ridge-relative texture cloning plus gradient-domain blending for the hidden flank. It expects the source photo path at the top of the file.

Open `index.html` in a browser. No build step.
