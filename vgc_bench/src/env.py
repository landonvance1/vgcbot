"""
Gymnasium environment module for VGC-Bench.

Provides a custom Gymnasium environment wrapping poke-env's DoublesEnv for
training reinforcement learning agents on Pokemon VGC battles.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from gymnasium import Env
from gymnasium.spaces import Box
from poke_env.battle import AbstractBattle
from poke_env.environment import DoublesEnv, SingleAgentWrapper
from poke_env.ps_client import ServerConfiguration
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv as SB3VecEnv

from vgc_bench.src.policy_player import PolicyPlayer
from vgc_bench.src.teams import RandomTeamBuilder, TeamToggle, get_available_regs
from vgc_bench.src.utils import LearningStyle, chunk_obs_len, format_map, moves


class SelfPlayVecEnv(SB3VecEnv):
    """
    VecEnv for PURE_SELF_PLAY: wraps N DoublesEnv (PettingZoo parallel) instances
    and exposes both agents as separate slots, giving total num_envs = 2 * N.

    Replaces the broken supersuit pettingzoo_env_to_vec_env_v1 + concat_vec_envs_v1
    pipeline, which fails because official poke-env uses MultiDiscrete([107, 107])
    per agent and supersuit's ConcatVecEnv only registers an iterate handler for
    Discrete spaces, causing a shape mismatch on action concatenation.
    """

    def __init__(self, env_fns: List[Callable[[], "ShowdownEnv"]]) -> None:
        self._envs = [fn() for fn in env_fns]
        sample = self._envs[0]
        self._agents = sample.possible_agents
        obs_space = sample.observation_space(self._agents[0])
        act_space = sample.action_space(self._agents[0])
        super().__init__(2 * len(self._envs), obs_space, act_space)
        self._actions: Optional[np.ndarray] = None

    @staticmethod
    def _stack_obs(obs_list):
        if isinstance(obs_list[0], dict):
            return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}
        return np.stack(obs_list)

    def reset(self):
        all_obs = []
        for env in self._envs:
            obs_dict, _ = env.reset()
            for agent in self._agents:
                all_obs.append(obs_dict[agent])
        return self._stack_obs(all_obs)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        assert self._actions is not None
        all_obs, all_rews, all_dones, all_infos = [], [], [], []
        for i, env in enumerate(self._envs):
            act_dict = {
                self._agents[0]: self._actions[2 * i],
                self._agents[1]: self._actions[2 * i + 1],
            }
            obs_dict, rew_dict, term_dict, trunc_dict, info_dict = env.step(act_dict)
            done = any(term_dict.values()) or any(trunc_dict.values())
            if done:
                for agent, obs in obs_dict.items():
                    info_dict[agent]["terminal_observation"] = obs
                obs_dict, _ = env.reset()
            for agent in self._agents:
                all_obs.append(obs_dict[agent])
                all_rews.append(rew_dict.get(agent, 0.0))
                all_dones.append(done)
                all_infos.append(info_dict.get(agent, {}))
        self._actions = None
        return (
            self._stack_obs(all_obs),
            np.array(all_rews, dtype=np.float32),
            np.array(all_dones, dtype=bool),
            all_infos,
        )

    def close(self) -> None:
        for env in self._envs:
            env.close()

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [getattr(env, attr_name) for env in self._select_envs(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        for env in self._select_envs(indices):
            setattr(env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        return [
            getattr(env, method_name)(*method_args, **method_kwargs)
            for env in self._select_envs(indices)
        ]

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False] * (len(self._select_envs(indices)) * 2)  # 2 agent slots per env

    def _select_envs(self, indices):
        if indices is None:
            return self._envs
        game_indices = sorted({i // 2 for i in indices})
        return [self._envs[i] for i in game_indices]


class ShowdownEnv(DoublesEnv):
    """
    Gymnasium environment for Pokemon VGC doubles battles.

    Extends poke-env's DoublesEnv with custom observation embedding,
    reward calculation, and support for various training paradigms.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Initialize the ShowdownEnv.
        """
        super().__init__(*args, **kwargs)
        self.observation_spaces = {
            agent: Box(-1, len(moves), shape=(12 * chunk_obs_len,), dtype=np.float32)
            for agent in self.possible_agents
        }

    @classmethod
    def create_env(
        cls,
        reg: str | None,
        run_id: int,
        num_teams: int | None,
        num_envs: int,
        log_level: int,
        port: int,
        learning_style: LearningStyle,
        allow_mirror_match: bool,
        choose_on_teampreview: bool,
        team_paths: list[Path] | None = None,
    ) -> Env:
        """
        Factory method to create a properly wrapped training environment.

        Creates the base ShowdownEnv and applies appropriate wrappers based
        on the learning style (vectorization for self-play, single-agent
        wrapper for other paradigms).

        Args:
            reg: VGC regulation letter (e.g. 'g', 'h', 'i'), or None for all.
            run_id: Training run identifier.
            num_teams: Number of teams to train with, or None for all.
            num_envs: Number of parallel environments.
            log_level: Logging verbosity for Showdown clients.
            port: Port for the Pokemon Showdown server.
            learning_style: Training paradigm to use.
            allow_mirror_match: Whether to allow same-team matchups.
            choose_on_teampreview: Whether policy controls teampreview.
            team_paths: Optional list of team file paths for matchup solving.

        Returns:
            Wrapped Gymnasium environment ready for training.
        """
        toggle = None if allow_mirror_match else TeamToggle()
        if reg is None:
            battle_format = format_map[get_available_regs()[0]]
        else:
            battle_format = format_map[reg]
        if learning_style == LearningStyle.PURE_SELF_PLAY:
            def _make_env():
                team_toggle = None if allow_mirror_match else TeamToggle()
                return cls(
                    server_configuration=ServerConfiguration(
                        f"ws://localhost:{port}/showdown/websocket",
                        "https://play.pokemonshowdown.com/action.php?",
                    ),
                    battle_format=battle_format,
                    log_level=log_level,
                    accept_open_team_sheet=True,
                    open_timeout=None,
                    team=RandomTeamBuilder(run_id, num_teams, reg, team_paths, team_toggle),
                    choose_on_teampreview=choose_on_teampreview,
                )
            return SelfPlayVecEnv([_make_env for _ in range(num_envs)])
        else:
            env = cls(
                server_configuration=ServerConfiguration(
                    f"ws://localhost:{port}/showdown/websocket",
                    "https://play.pokemonshowdown.com/action.php?",
                ),
                battle_format=battle_format,
                log_level=log_level,
                accept_open_team_sheet=True,
                open_timeout=None,
                team=RandomTeamBuilder(run_id, num_teams, reg, team_paths, toggle),
                choose_on_teampreview=choose_on_teampreview,
            )
            opponent = PolicyPlayer(start_listening=False)
            env = SingleAgentWrapper(env, opponent)
            env = Monitor(env)
            return env

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Reset the environment, updating battle format if multi-reg."""
        assert isinstance(self._team, RandomTeamBuilder)
        if self._team.available_regs is not None:
            assert self._team.current_reg is not None
            self._team.pick_reg()
            fmt = format_map[self._team.current_reg]
            self.agent1._format = fmt
            self.agent2._format = fmt
        # Showdown server retains battle state across reconnects when the username is
        # the same (deterministic seed → same username every run). On reconnect the
        # server resumes the old unfinished battle, which lands in agent._battles and
        # causes reset_battles() to raise "Can not reset while battles are running".
        # Force-clear any unfinished battles that aren't the current live ones.
        for agent in (self.agent1, self.agent2):
            stale = [tag for tag, b in agent._battles.items()
                     if not b.finished and b is not self.battle1 and b is not self.battle2]
            if stale:
                for tag in stale:
                    del agent._battles[tag]
        return super().reset(seed=seed, options=options)

    def calc_reward(self, battle: AbstractBattle) -> float:
        """
        Calculate reward for the current battle state.

        Returns:
            1 if won, -1 if lost, 0 otherwise.
        """
        if not battle.finished:
            return 0
        elif battle.won:
            return 1
        elif battle.lost:
            return -1
        else:
            return 0

    @staticmethod
    def get_action_mask(battle) -> list:
        """
        Override to fix poke-env's teampreview mask bug.

        DoublesEnv.get_action_mask_individual builds the teampreview mask by
        enumerating all team members where selected_in_teampreview is False,
        enabling switch actions 1–6 for a 6-Pokemon team. But
        DoubleBattle.valid_orders (used by action_to_order to validate the
        submitted action) is built from available_switches, which excludes
        Pokemon that are active=True. Showdown marks the first 2 team slots as
        active in the teampreview request JSON (default leads), so those 2 are
        absent from valid_orders but present in the base mask — causing a
        ValueError when the policy selects one of them.

        Fix: for teampreview, restrict valid switch actions to Pokemon that are
        both in available_switches and not yet selected_in_teampreview,
        matching the set that valid_orders will accept.
        """
        if not battle.teampreview:
            return DoublesEnv.get_action_mask(battle)
        avail_base = [
            {p.base_species for p in battle.available_switches[pos]}
            for pos in range(2)
        ]
        size = DoublesEnv.get_action_space_size(battle.gen)

        def _mask_for_pos(pos):
            actions = [
                i + 1
                for i, pokemon in enumerate(battle.team.values())
                if pokemon.base_species in avail_base[pos]
                and not pokemon.selected_in_teampreview
            ] or [0]
            return [int(i in actions) for i in range(size)]

        return _mask_for_pos(0) + _mask_for_pos(1)

    def embed_battle(self, battle: AbstractBattle) -> npt.NDArray[np.float32]:
        """
        Convert the battle state to a feature vector observation.

        Args:
            battle: Current battle state.

        Returns:
            Numpy array observation for the policy network.
        """
        return PolicyPlayer.embed_battle(battle, fake_rating=2000)
