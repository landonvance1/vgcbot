# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

vgcbot is an RL project aiming to build the strongest possible competitive Pokemon doubles (VGC) AI, targeting the **Pokemon Champions** ruleset. See [README.md](README.md) for the full pitch.

## Core thesis — read before suggesting design changes

We are extending [VGC-Bench](https://github.com/cameronangliss/VGC-Bench) (UT Austin, [arXiv:2506.10326](https://arxiv.org/abs/2506.10326); our summary at `research/vgc-bench-summary.md`). VGC-Bench reached expert-level play for a single team but degrades as the team set grows. Our bet is that their action space is too noisy and too large; we plan to:

1. Pre-calculate potential damage for damaging moves (remove damage-roll variance from the agent's problem).
2. Genericize damaging moves into abstracted archetypes (shrink the action cardinality).

When proposing architecture, observations, or training details, preserve this thesis. Do not reflexively mirror VGC-Bench in places where the simplification changes the tradeoff — in particular, move embeddings, target selection, and the action head will likely diverge from the paper.

## Ruleset caveat

**Pokemon Champions is not Scarlet/Violet VGC.** Team size, legal Pokemon, Tera mechanics, items, and format details may differ. Before locking in anything format-specific (action space size, legal action masking, team preview shape), confirm the current Pokemon Champions rules rather than copying from VGC-Bench's Gen 9 VGC assumptions.

## Repo layout

- `research/vgc-bench-summary.md` — our summary of the primary reference paper (arXiv:2506.10326).
- `LICENSE` — MIT.

The repo is currently pre-implementation; the RL stack, environment wrapper, and training code have not been written yet.

## Working conventions

- Keep commits focused and meaningful; the repo is new so history is still establishing conventions.
- When referencing the paper in code comments or docs, cite it as VGC-Bench (Angliss et al., 2025) rather than pasting large excerpts.
