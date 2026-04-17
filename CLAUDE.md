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
- `vgc_bench/` — training infrastructure adapted from VGC-Bench (MIT, Cameron Angliss 2025). See attribution note below.
  - `src/env.py` — Gymnasium environment wrapping poke-env's `DoublesEnv`. **`embed_battle` is our primary replacement target.**
  - `src/policy.py` — Actor-critic policy network. **`AttentionExtractor` is our primary replacement target.**
  - `src/policy_player.py` — Player that runs the policy; contains the full `embed_battle` implementation.
  - `src/teams.py` — Team pool management (loads `.txt` files from `teams/`).
  - `src/utils.py` — Shared constants: observation/action dimensions, move/item/ability index lists.
  - `src/callback.py` — SB3 training callback: evaluation, checkpointing, fictitious play opponent sampling.
  - `train.py` — PPO training entry point; supports self-play, fictitious play, double oracle, exploiter.
- `data/` — `abilities.json`, `items.json`, `moves.json` index lists (used by `utils.py` at import time).
- `teams/` — Competitive team files organised by regulation (`reg_a/` through `reg_j/`).
- `LICENSE` — MIT (covers our code). VGC-Bench original copyright retained per their MIT license.

## Environment setup

- **Python environment:** `.venv/` at repo root (`python3 -m venv .venv`). Activate with `source .venv/bin/activate`. All deps in `pyproject.toml`.
- **Showdown server:** cloned at `../pokemon-showdown/`. Start with `node ../pokemon-showdown/pokemon-showdown start --no-security`. Runs on port 8000.
- **poke-env:** using the official PyPI package (`poke-env`), not VGC-Bench's fork. Doubles support (`DoublesEnv`) is present in the official package.

## Known issues

- **`PURE_SELF_PLAY` mode is broken** (landonvance1/vgcbot#2): supersuit's `concat_vec_env` is incompatible with the official poke-env's action space shape. Use `FICTITIOUS_PLAY` mode (which uses `SingleAgentWrapper`) for all development until resolved. The obs/action pipeline itself is fine — the issue is only in supersuit's vectorization layer.
- **`FICTITIOUS_PLAY` requires a checkpoint:** the opponent `PolicyPlayer` has no policy until `set_opp_policy` is called by the callback. `env.reset()` will time out if called before any checkpoint exists. Use `SimpleHeuristicsPlayer` as a stand-in opponent during early development.

## VGC-Bench attribution

`vgc_bench/` is adapted from [VGC-Bench](https://github.com/cameronangliss/VGC-Bench) (MIT License, Copyright 2025 Cameron Angliss). Their license is retained in `LICENSE`. We keep their training infrastructure (PPO loop, fictitious play, callback) and replace the observation featurizer and policy architecture per our thesis.

We intentionally **omit** from VGC-Bench: `llm.py` (Llama baseline), `eval.py` (cross-eval suite, add later), `pretrain.py` (BC pretraining, add later), `scrape_*.py` / `logs2trajs.py` (data pipeline, add later).

## Working conventions

- Keep commits focused and meaningful.
- When referencing the paper in code comments or docs, cite it as VGC-Bench (Angliss et al., 2025) rather than pasting large excerpts.
- The `.venv/` directory is gitignored. New contributors run `python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"`.
- Do not use `PURE_SELF_PLAY` mode until landonvance1/vgcbot#2 is resolved.
