# Physical Plausibility Atlas

A visual research atlas for evaluating generated robot-manipulation videos using three complementary metrics:

- **Contact** — FuturePair36 contact-response plausibility.
- **Penetration** — Type A boundary crossing, experimental backward-trajectory diagnostics, Type B limitations, and Type C truncation evaluation.
- **Consistency** — duplication and target-conditioned object birth/death/persistence.

The project includes a browsable video gallery with human ground-truth labels, per-metric results, and method flowcharts. Case IDs without generated model scores remain explicitly **not tested / no score**.

## Video hosting

The published `videos/*.mp4` clips are compressed H.264 copies of the source clips. The `videos/manifest.json` maps evaluated case IDs to relative URLs and is used by `index.html` in GitHub Pages. This static publication does not use the local Python server or private Mac absolute paths.

**All published videos are public.** Originals are left untouched on the user's Mac. Only videos matching the 141 evaluated case IDs are considered.

## Publish / refresh

Use the accompanying publishing bundle (`publish.sh`, `publish_videos.py`, `case_ids.json`) and a Mac with GitHub CLI and ffmpeg installed. After publishing, future updates can be made with the same script; it performs incremental `git push` updates.

GitHub Pages is configured for the `main` branch at `/` (repository root). If the automatic Pages setup is denied, enable it in repository Settings → Pages.
