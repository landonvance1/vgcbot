# VGC-Bench — Summary

**Paper:** Angliss, Cameron L and Cui, Jiaxun and Hu, Jiaheng and Rahman, Arrasy and Stone, Peter. *VGC-Bench: Towards Mastering Diverse Team Strategies in Competitive Pokemon.* arXiv:2506.10326v2, June 2025.

**Repo:** https://github.com/cameronangliss/VGC-Bench

**PDF:** https://arxiv.org/abs/2506.10326

This is a summary in our own words of the parts relevant to vgcbot. See the paper for full detail, math, and tables.

## TL;DR

VGC-Bench is a benchmark + baseline suite for AI agents playing competitive Pokemon doubles. Built on [poke-env](https://github.com/hsahovic/poke-env) + [PettingZoo](https://pettingzoo.farama.org/), it adds VGC/doubles support, a 330k+ game human replay dataset, and 11 baseline agents (heuristics, an LLM agent, behavior cloning, and three population-based RL methods). Their strongest agent — behavior cloning initialization fine-tuned with fictitious play (BCFP) — beats a VGC-World-Championships-qualified player in single-team matches, but performance degrades sharply as the training team pool grows to 3, 10, and 30 teams. Generalization across teams is the headline open problem.

## Why VGC is hard

Four properties stack up to make VGC harder than prior benchmarked games (Chess, Go, Poker, Dota, StarCraft):

1. **Team configuration space ~10^139** — species, moves, items, abilities, natures, EVs, IVs, Tera types all combine. Orders of magnitude larger than Dota's ~10^17.
2. **Stochastic battle mechanics** — ~16 damage rolls per damaging move, crits, secondary-effect procs, accuracy. Per-turn branching factor ~10^12.
3. **Partial observability** — even with Open Team Sheets (OTS), exact stat spreads are hidden. Information-set size ~10^58.
4. **Simultaneous, multi-agent actions** — both players commit to actions for two active Pokemon each, per turn. Credit assignment is noisy (e.g. if two Pokemon attack the same target and one is immune, the observation after the turn can't tell you which dealt the KO).
5. **Team preview** — each match, players select 4 of 6 Pokemon before play. That alone is (6 choose 2)·(4 choose 2) = 90 team-preview decisions per side.

They formalize the problem as a two-player zero-sum partially-observable stochastic game (POSG) with team configurations sampled per episode.

## Infrastructure contributions

- Integrated poke-env with PettingZoo so multi-agent / population-based methods can run against it cleanly.
- Added VGC + doubles support to poke-env.
- Training toggles: skip team-preview phase; disable mirror matches (but they note that disabling mirrors can cause reward hacking where the agent throws one side to win the other).
- Parallelized scraper for Gen 9 OTS-enabled Pokemon Showdown replays → **330,000+ games** of human play data reconstructed into approximate state/action trajectories.
- Pool of competitive teams sourced from VGCPastes.

## Observation and action space

- **Observation:** 12 Pokemon (6 ours, 6 opponent) × features composed of global features (weather, etc.), side features (screens, etc.), and per-Pokemon features. Optional frame stacking adds a time axis.
- **Action space:** per-Pokemon, 107 discrete actions covering: pass, forfeit, default, 6 switches, and 4 moves × up to 3 targets, each optionally with Mega / Z-Move / Dynamax / Terastallize. Turn action is the joint of two Pokemon's actions. Team preview is modeled as two joint switch-ins.

## Architecture

- Transformer encoder over the 12 Pokemon tokens (3 layers) + learned embeddings for moves, items, abilities.
- When frame-stacked: a second Transformer along the time axis with causal masking; only the final frame's logits are used.
- Linear head to action-space logits, with invalid-action masking at −∞ and interdependent-action constraints (e.g. two Pokemon can't switch to the same slot).
- Actor-critic with **no parameter sharing** between policy and value nets (shared architecture though).

## Baselines

Three heuristic agents (Random, Max Base Power, Simple Heuristics — the last extended by the authors to doubles), an LLM agent (Llama 3.1 8B Instruct, prompt-engineered), a behavior cloning agent trained on replays from players rated ≥1200, and three population-based RL methods on PPO: self-play (SP), fictitious play (FP), and double oracle (DO). Each RL variant is also run from a BC-initialized policy: **BCSP, BCFP, BCDO**.

## Results

- **Single team (the restricted setting):** BC-initialized RL variants dominate. ELO ranking roughly: BCFP > BCSP > FP > SP > BCDO > SH > DO > MBP > BC > R > LLM. In a cyclic rock-paper-scissors pattern the game is non-transitive at one team (FP > BCFP > DO > FP).
- **Human matches (single team, BCFP):** beat an intermediate player 5/5, beat an advanced player 2/5, and won some games against an expert who qualified for the VGC World Championships. Feedback: the agent is very strong on first-play but has predictable dips; strong humans can adapt across many games.
- **Generalization (3, 10, 30 teams):** In-distribution win rate drops monotonically as the training team pool grows (Table 3 in the paper: from 92 when one-team agent fights thirty-team agent down to 8 when thirty-team agent fights one-team agent, on the one-team agent's home teams). Out-of-distribution, the thirty-team agent does modestly better than single-team agents (Table 4) — so some generalization, but not enough to hold up in-distribution strength.
- **Exploitability:** for all four agents (1, 3, 10, 30 teams), a BC-initialized counter-policy reached ~100% win rate after 3M training steps. So these agents are far from Nash-optimal.

## Hyperparameters (single training run)

PPO, learning rate 1e-5, γ=1.0, GAE λ=0.95, clip 0.2, entropy coef 0.001, value coef 0.5, max grad norm 0.5, 24·128 steps per update, batch 64, 10 epochs, ~5M total timesteps. 8× A40 GPUs.

## Relevance to vgcbot

What they **did not** do — and what our project is betting on:

- They do **not** pre-compute damage or abstract damaging moves. Their architecture goes the other way: learn move/item/ability embeddings from scratch via Transformer, letting the network discover structure.
- Their acknowledged simplifications (disable team preview, disable mirror matches) are orthogonal to the move/damage axis — they don't touch action-space cardinality or damage-roll variance.
- The stochastic branching factor (~10^12/turn) is flagged as a core challenge but not attacked directly; it's treated as part of the environment.

Our hypothesis is that pre-computing expected damage and collapsing damaging moves into a small archetype set will give the learner a cleaner signal on the strategically interesting axes (switch, target, status, Tera timing, team preview), and should help with cross-team generalization. The paper gives us a clear baseline to compare against in the single-team setting and a clear open problem (multi-team generalization) to aim at.

**Ruleset note:** the paper is on Scarlet/Violet Gen 9 VGC. vgcbot targets the new **Pokemon Champions** ruleset, which may differ in team size, legal Pokemon, Tera/gimmick mechanics, and items. Don't assume parity — check Pokemon Champions rules before mirroring their action-space shape or team-preview logic.
