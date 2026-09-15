# STATUS: voice-toy

> Last updated: 2026-09-15 15:50

## Current State
Spec drafted and awaiting user review. No code yet. Git repo initialised.

## In Progress
- SPEC.md written for review (per user's process: review before scaffolding).

## Recently Completed
- Project dir created at `/home/rob/projects/voice-toy`, git initialised, `.gitignore` added.
- SPEC.md, TASKS.md, STATUS.md, DECISIONS.md, NOTES.md drafted.

## Blockers
- Awaiting user sign-off on SPEC.md before T001 (scaffold).

## Next Action
- Present SPEC.md to user for review. On approval, start T001 (scaffold).

## Resume Notes
- User's process: SPEC.md must be reviewed before scaffolding/implementation.
- Pithagoras voice pipeline is the design reference (speculative STT, sentence-chunked TTS, barge-in) but its code is GPU-locked — this is a fresh CPU build.
- Open unknowns to verify early: llama.cpp URL reachable from container (U1), enable_thinking support on the live model (U2), mic over plain HTTP (U4/A4).
