# 2026-11-24 — YouTube icon added to email signature

**Request:** Add YouTube to the social icons in the Gmail signature (jay@rated-n.ca), matching the Instagram and LinkedIn icons and linked to the Rated-N channel.

**Status:** Done. The new signature, "Jay with Youtube", was saved in Gmail and set as the default by the human.

## What was done
- Read the existing signature HTML through a temporary Gmail draft, which was then deleted.
- Measured the existing icons: outline style from the Lucide icon set (stroke 2 on a 24-unit grid), green `#3CFFA6`, transparent background, shown at 20×20 from 40px files.
- Built the matching YouTube icon: `assets/brand/social/youtube-40.png` (also a `youtube-80.png`).
- Added it as a new table cell after LinkedIn: same 12px spacer, 20×20 size, linked to https://www.youtube.com/@Rated-N.
- Full signature HTML saved as `signature.html` in this repo.
- Delivered it through a private copy-and-paste page: https://claude.ai/artifact/3rDNYxSnPe84R1jbXCD4LL

## Limits encountered
- The Gmail connector can't write signature settings, so the human pasted the signature in manually.
- Gmail drafts created through the API drop external images, so a draft couldn't be used to carry the signature.
- The source for rated-n.ca (Netlify project `rated-n-v10-test`) isn't in any reachable GitHub repo, so the icon couldn't go on the site. A deploy without the source would have overwritten the live site.

## Open item
The YouTube icon is served from:
`https://raw.githubusercontent.com/Jaybulckaert/Jaybulckaert/c19f54f43074bb34025d127bb1fd3c7ad360d9b6/assets/brand/social/youtube-40.png`
That commit is only on branch `claude/beautiful-hopper-sr38ay`. To make it permanent, do one of these:
1. Upload `youtube-40.png` to rated-n.ca at `/assets/brand/social/` and change that one image link in the Gmail signature (preferred; all three icons then come from your own domain), or
2. Merge this branch into `main`.

Until one of these is done, don't delete the branch.
