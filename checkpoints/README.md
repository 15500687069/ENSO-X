# ENSO-X Checkpoints

This directory documents the checkpoint set used by the released ENSO-X package.

## Kept Checkpoints

- `final_24_complete`
  Final released ENSO-X checkpoint. This is the main checkpoint for evaluation and
  inference.
- `seed_24_run`
  The direct seed checkpoint used to reproduce the final 24-month run.
- `lead23_baseline`
  Earlier high-lead baseline kept for comparison and ablation reference.

## Deployment Note

In the deployed server package, this directory also contains symlinks with the same names:

- `final_24_complete -> /root/autodl-tmp/enso_x_ckpts/enso_x_24_complete_20260425_145549`
- `seed_24_run -> /root/autodl-tmp/enso_x_ckpts/enso_x_seed24_20260425_144651`
- `lead23_baseline -> /root/autodl-tmp/enso_x_ckpts/enso_x_lead23_baseline_20260425_014522`

The local repository keeps the manifest and documentation so that the package remains
portable even when the large binary checkpoint files are stored outside version control.

## Recommended Usage

- Training reproduction: initialize from `seed_24_run/best_frontier.ckpt`
- Main evaluation: load `final_24_complete/best.ckpt`
- Historical comparison: load `lead23_baseline/best_lead.ckpt`

See `MANIFEST.json` in the same directory for the exact checkpoint names, metrics, and
server paths.
