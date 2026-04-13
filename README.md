# vgcbot

Reinforcement learning agent for competitive Pokemon doubles (VGC), targeting the **Pokemon Champions** ruleset.

## Goal

Train the strongest possible VGC battle AI. We build on [VGC-Bench](https://github.com/cameronangliss/VGC-Bench) (Angliss et al., UT Austin, [arXiv:2506.10326](https://arxiv.org/abs/2506.10326)), which demonstrated that PPO with behavior cloning + fictitious play can beat expert human players in single-team VGC, but struggles to generalize across teams.

## Hypothesis

VGC's learning signal is diluted by the stochastic, combinatorially vast action space — per-turn branching factor reaches ~10^12, and each Pokemon has 107 possible actions. We believe generalization can be improved by **simplifying the action space** before handing it to the RL agent:

1. **Pre-calculate potential damage** for damaging moves, so the policy sees a deterministic expected-damage value instead of learning the damage roll distribution.
2. **Genericize damaging moves** into a smaller abstracted set, so the policy chooses among damage archetypes rather than every unique move in the game.

The expectation is that reducing variance and cardinality on the damaging-move axis lets the RL agent focus capacity on the strategically interesting decisions (switches, protects, status, target selection, positioning, Tera timing, team preview).

## Status

Early scaffolding. See [`research/vgc-bench-summary.md`](research/vgc-bench-summary.md) for a summary of the paper we're building on.

## License

MIT — see [LICENSE](LICENSE).
