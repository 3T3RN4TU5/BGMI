# Status (paused 2026-11-05)

The project was paused because the laptop is short of disk space. Plan for
about 15 GB of fixed space (Python + PyTorch CUDA ~6 GB, pip cache ~3 GB, later
OCR/YOLO + training data ~5 GB, dashboard ~0.5 GB) plus ~4.5 GB per 6 h video.
An external drive for `videos_root` is the easiest way back.

## Where it stopped

- PRD build phase 0 (setup, catalog): done and merged (PR #1).
- PRD build phase 1 (scene classifier, match segmentation, zone, team table):
  code on branch `phase1-match15` (PR #2). Match 15 of BMSD 2026 Finals Day 3
  (`analyze 8023 --start 2:05:00 --end 2:50:00`):
  - match 15 / Erangel, number read from the screen in 63/64 frames
  - team tag, FIN and alive read on ~99.9% of table rows; 126/128 rows right in
    a hand check of 8 frames (the other 2 blank, never wrong)
  - zone found on 94.6% of map frames before the last zone fix (Hough
    fallback + previous ring as a hint, commit 61dd00f). That fix is tested on
    synthetic frames only.

## Next steps when resuming

1. Re-run the command above and check the 24 former zone misses
   (2:17:39 ... 2:24:06), and that no Hough circle is a false ring.
2. Match start: the real start is the split-screen drop view (~2:09:54), not
   the first full-map frame (2:09:57). Add probes for that layout.
3. Remaining phase 1 gates: `cli.py scenes` 200-frame check (≥98%), a full
   Day 3 run (6 matches within ±30 s), a manual alive check at 10 timestamps.

## Not in git (by design)

Videos, extracted frames, debug images and the team/digit templates
(`data_root/calib/<overlay>/glyphs.npz`) are cut from the broadcast, so they
are never committed. Re-create templates with `cli.py calibrate` on labelled
frames (labels for three frames are in the PR #2 discussion and below).

- Match 15 ~2:24 frame: `BLEG V1BA GODL 8BIT GENS TAG GDR NINZ NBE OR SOUL RNTX K9 WELT WF MYTH`,
  FIN `04 04 02 02 01 01 01 01 00 00 00 00 00 00 00 02`
- Match 15 later frame: `OR BLEG V1BA 8BIT GENS GODL GDR K9 RNTX SOUL NINZ NBE TAG WELT WF MYTH`,
  FIN `06 05 05 03 02 02 02 02 01 02 04 03 02 00 00 02`
- Match 16 frame: `TAG WELT BLEG GDR GODL NINZ V1BA OR SOUL WF K9 GENS MYTH RNTX NBE 8BIT`,
  FIN `07 06 02 02 01 06 06 02 05 03 00 03 02 01 04 02`

Team names (BMSD 2026 Grand Finals, from Liquipedia; only OR confirmed):
SOUL Team SouL, GDR Gladiators Esports, 8BIT 8Bit, K9 K9 Esports, BLEG Blitz
Epigrotive, OR Team Outrage, V1BA Blue Army, WELT Welt Esports, NINZ Meta
Ninza, TAG Team Apex Gaming, RNTX Revenant XSpark, MYTH Myth Official, WF Wyld
Fangs, NBE Nebula Esports, GODL GodLike Esports, GENS Genesis Esports.
