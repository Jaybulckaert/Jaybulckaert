# Two Peaks Rising

A selfie on the Banff recreation grounds, split into four plates so that only the two background mountains scale.

- **Back plate** `park-base.jpg`: the photograph at half resolution (2448 × 1836).
- **Mountain plates** `mountain-left.webp`, `mountain-right.webp`: the range on the left and the peak on the right, each cut along its ridge against the sky, with slopes reconstructed behind the trees.
- **Front plate** `foreground.webp`: every tree, the kiosk, cars, house, lawn and the person.

`index.html` composites the plates on a canvas. Each mountain plate scales about its own base line under its summit, both driven by one slider from 1400 : 1 to 5000 : 1 against your height (magnification `5000 / 1400 = 3.571`). Back and front plates never transform.

`pipeline/build_layers.py` is the OpenCV script that produced the plates. It expects the source photo as `park.jpg` next to it.
