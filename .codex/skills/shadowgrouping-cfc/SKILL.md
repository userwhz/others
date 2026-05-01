---
name: shadowgrouping-cfc
description: Use when working on the shadowgrouping BeH2/H2O timing jobs on cfc, especially when launching or checking tmux runs, editing the remote shadowgrouping code, or measuring grouping and 10-shot simulation timing.
---

# Shadowgrouping CFC Workflow

When working on the cfc copy of this project:

- SSH target: `cfc`.
- Remote user is expected to be `biankaiming`.
- Reuse the existing tmux session `beh2_jupyter`.
- Work in the existing remote folder `~/shadowgrouping_beh2`; do not create a parallel project folder for this task.
- For a new run, create a new tmux window inside `beh2_jupyter` rather than starting a separate session.

Typical launch pattern:

```bash
tmux new-window -t beh2_jupyter -n <short-name> "cd ~/shadowgrouping_beh2 && <command>"
```

Timing runs should write logs under `~/shadowgrouping_beh2/logs/` and use `SG_TIMING_FILE` for JSONL timing output.
