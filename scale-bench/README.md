# Banff Scale Bench

A scrollable side-by-side viewer for one photo of Cascade Mountain from Banff Avenue.

- Left frame: the original photo, read at **1600 : 1** against your height.
- Right frame: the same photo magnified by `5000 / 1600 = 3.125` so the mountain stands **5000 : 1**. Your pixel height is held constant; the world grows.
- A ruler in multiples of your height is anchored where the rock face meets the treeline. The summit lands on the 1600 tick on the left and on the comparison tick on the right.
- Scrolling either frame moves the other to the same point in the photo. The comparison scale slides from 1600 to 10000.

Open `index.html` in a browser. No build step, no dependencies beyond Google Fonts.
