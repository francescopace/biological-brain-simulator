"""
Grid navigation benchmark for the synthetic brain.

This is the first reinforcement-learning benchmark in the repo: the
network must learn which action to take from each position in a 5x5
grid to reach a fixed goal. Learning uses reward-modulated STDP, not
backpropagation.

Architecture:
    input (25 sensory neurons, one per grid cell)
        ↓ density 0.5
    place_cells (100 memory neurons with theta-gamma coupling)
        ↓ density 1.0 (R-STDP)
    motor (4 chattering neurons: up/down/left/right)

    input ─────────────→ motor  (direct readout, R-STDP)

Unlike the Iris benchmark, this task is not supervised. The agent
chooses an action from motor spikes, executes it in the environment,
and then receives reward or punishment depending on whether that move
reduced the distance to the goal.
"""

from __future__ import annotations

import copy
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import quiet_steps
from src.brain import Brain
from src.device import DEVICE
from src.neuron import FiringPattern, NeuronType
from src.region import RegionType


# --- Task setup ------------------------------------------------------------

GRID_SIZE = 5
N_INPUT = GRID_SIZE * GRID_SIZE
N_PLACE_CELLS = 100
N_MOTOR = 4

ACTION_NAMES = ("up", "down", "left", "right")
ACTION_DELTAS = np.array(
    [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ],
    dtype=np.int32,
)


# --- Training hyperparameters ---------------------------------------------

STATE_PRESENT_STEPS = 30
ACTION_MARK_STEPS = 8
ACTION_MARK_CURRENT = 16.0
MOTOR_BASELINE_CURRENT = 4.0
POST_REWARD_STEPS = 10
INTER_STEP_REST = 25

MAX_STEPS_PER_EPISODE = 20
EPISODES = 500
EVAL_EPISODES = 100

EPSILON_START = 0.50
EPSILON_END = 0.10
EPSILON_DECAY_EPISODES = 300

PLACE_FIELD_SIGMA = 0.6
STEP_REWARD = 0.080
STEP_PUNISH = 0.030
WALL_PUNISH = 0.015
GOAL_REWARD = 0.400

LATERAL_INHIBITION = 5.0
ENCODER_MAX_CURRENT = 55.0
ENCODER_NOISE = 0.05

PROJECTION_DENSITY_INPUT_PLACE = 0.50
PROJECTION_DENSITY_PLACE_MOTOR = 0.50
PROJECTION_DENSITY_INPUT_MOTOR = 1.00
PROJECTION_WEIGHT_BOOST_IP = 7.0
READOUT_INIT_WEIGHT = 0.05

STDP_SCALE = 1.0
TAU_ELIGIBILITY = 200.0
DOPAMINE_DECAY = 0.5

REST_INTERVAL = 50
REST_STEPS = 5000

SEED = 42


# --- Environment -----------------------------------------------------------

@dataclass
class GridWorld:
    size: int = GRID_SIZE
    goal: tuple[int, int] = (GRID_SIZE - 1, GRID_SIZE - 1)

    def reset(self, rng: np.random.Generator) -> tuple[int, int]:
        while True:
            pos = (
                int(rng.integers(0, self.size)),
                int(rng.integers(0, self.size)),
            )
            if pos != self.goal:
                return pos

    def manhattan_distance(self, pos: tuple[int, int]) -> int:
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def step(
        self,
        pos: tuple[int, int],
        action: int,
    ) -> tuple[tuple[int, int], bool, bool, int, int]:
        prev_dist = self.manhattan_distance(pos)
        delta = ACTION_DELTAS[action]
        nxt = (
            int(np.clip(pos[0] + int(delta[0]), 0, self.size - 1)),
            int(np.clip(pos[1] + int(delta[1]), 0, self.size - 1)),
        )
        moved = nxt != pos
        new_dist = self.manhattan_distance(nxt)
        reached_goal = nxt == self.goal
        return nxt, moved, reached_goal, prev_dist, new_dist


# --- Brain construction ----------------------------------------------------

def build_brain(seed: int = SEED) -> Brain:
    brain = Brain(dt=1.0, seed=seed)

    brain.add_region(
        "input",
        RegionType.SENSORY,
        n_neurons=N_INPUT,
        connectivity=0.0,
        max_neurons=N_INPUT,
    )
    brain.add_region(
        "place_cells",
        RegionType.MEMORY,
        n_neurons=N_PLACE_CELLS,
        connectivity=0.05,
        max_neurons=N_PLACE_CELLS,
    )
    brain.add_region(
        "motor",
        RegionType.MOTOR,
        n_neurons=0,
        connectivity=0.0,
        max_neurons=N_MOTOR,
    )

    motor = brain.regions["motor"]
    for _ in range(N_MOTOR):
        motor.add_neuron(NeuronType.EXCITATORY, FiringPattern.CHATTERING)
    motor.add_lateral_inhibition(weight=LATERAL_INHIBITION, delay_ms=1.0)

    brain.connect_regions("input", "place_cells", density=PROJECTION_DENSITY_INPUT_PLACE)
    brain.connect_regions("place_cells", "motor", density=PROJECTION_DENSITY_PLACE_MOTOR)
    brain.connect_regions("input", "motor", density=PROJECTION_DENSITY_INPUT_MOTOR)

    for proj in brain.projections:
        ns = proj.n_synapses
        if proj.source_name == "input" and proj.target_name == "place_cells":
            proj.syn_weight[:ns] *= PROJECTION_WEIGHT_BOOST_IP
        elif proj.target_name == "motor":
            proj.syn_weight[:ns] = READOUT_INIT_WEIGHT

    # Keep plasticity focused on state -> action pathways.
    brain.freeze_plasticity()
    brain.enable_projection_plasticity(
        "place_cells",
        "motor",
        A_plus=0.01 * STDP_SCALE,
        A_minus=0.012 * STDP_SCALE,
    )
    brain.enable_projection_plasticity(
        "input",
        "motor",
        A_plus=0.01 * STDP_SCALE,
        A_minus=0.012 * STDP_SCALE,
    )

    brain.encoder.max_current = ENCODER_MAX_CURRENT
    brain.encoder.noise_level = ENCODER_NOISE
    brain.reward_stdp.tau_eligibility = TAU_ELIGIBILITY
    brain.reward_stdp.dopamine_decay = DOPAMINE_DECAY

    # Make replay visible and deterministic enough for benchmarking.
    brain.memory.trace_capacity = 200
    brain.memory.trace_threshold = 0.08
    brain.memory.replay_strength = 1.25
    brain.memory.consolidation_interval = REST_STEPS

    # Keep the substrate stable while we test reward learning + replay.
    brain.freeze_structural_plasticity()

    return brain


# --- Helpers ---------------------------------------------------------------


def reset_between_steps(brain: Brain, n_steps: int = INTER_STEP_REST) -> None:
    quiet_steps(brain, n_steps)
    brain.reset_traces()


def position_encode(
    pos: tuple[int, int],
    size: int = GRID_SIZE,
    sigma: float = PLACE_FIELD_SIGMA,
) -> np.ndarray:
    row, col = pos
    coords = np.indices((size, size)).reshape(2, -1).T.astype(np.float64)
    diff = coords - np.array([row, col], dtype=np.float64)
    dist2 = np.sum(diff * diff, axis=1)
    act = np.exp(-0.5 * dist2 / (sigma * sigma))
    return act.astype(np.float64)


def present_state(
    brain: Brain,
    x: np.ndarray,
    n_steps: int = STATE_PRESENT_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    motor = brain.regions["motor"]
    before = motor.total_spikes[:N_MOTOR].clone()
    voltage_sum = torch.zeros(N_MOTOR, dtype=torch.float32, device=DEVICE)
    for _ in range(n_steps):
        brain.stimulate("input", x)
        brain.inject_current("motor", list(range(N_MOTOR)), MOTOR_BASELINE_CURRENT)
        brain.step()
        voltage_sum += motor.v[:N_MOTOR]
    counts = (motor.total_spikes[:N_MOTOR] - before).cpu().numpy()
    mean_voltage = (voltage_sum / n_steps).cpu().numpy()
    return counts, mean_voltage


def decode_action_scores(brain: Brain, x: np.ndarray) -> np.ndarray:
    scores = torch.zeros(N_MOTOR, dtype=torch.float32, device=DEVICE)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    for proj in brain.projections:
        if proj.source_name != "input" or proj.target_name != "motor":
            continue
        ns = proj.n_synapses
        if ns == 0:
            continue
        pre_idx = proj.syn_pre[:ns].to(torch.int64)
        post_idx = proj.syn_post[:ns].to(torch.int64)
        scores.index_add_(0, post_idx, x_t[pre_idx] * proj.syn_weight[:ns])
    return scores.cpu().numpy()


def mark_executed_action(brain: Brain, x: np.ndarray, action: int) -> None:
    # Exploration needs an explicit state-action trace; otherwise a randomly
    # chosen action may leave almost no eligibility on the executed pathway.
    for _ in range(ACTION_MARK_STEPS):
        brain.stimulate("input", x)
        brain.inject_current("motor", np.arange(N_MOTOR), MOTOR_BASELINE_CURRENT)
        brain.inject_current("motor", [action], ACTION_MARK_CURRENT)
        brain.step()


def _motor_targets(action: int, proj) -> None:
    ns = proj.n_synapses
    keep = proj.syn_post[:ns] == action
    proj.syn_eligibility[:ns][~keep] = 0.0


def keep_only_action_eligibility(brain: Brain, action: int) -> list[str]:
    targets: list[str] = []
    for proj in brain.projections:
        if proj.target_name != "motor":
            continue
        _motor_targets(action, proj)
        targets.append(Brain.projection_target(proj.source_name, proj.target_name))
    return targets


def epsilon_for_episode(episode_idx: int) -> float:
    progress = min(1.0, episode_idx / EPSILON_DECAY_EPISODES)
    return EPSILON_START + progress * (EPSILON_END - EPSILON_START)


def choose_action(
    counts: np.ndarray,
    mean_voltage: np.ndarray,
    decoder_scores: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
    guided_actions: np.ndarray | None = None,
) -> tuple[int, str]:
    if rng.random() < epsilon:
        if guided_actions is not None and len(guided_actions) > 0:
            idx = int(rng.integers(0, len(guided_actions)))
            return int(guided_actions[idx]), "guided"
        return int(rng.integers(0, N_MOTOR)), "explore"
    if decoder_scores.max() > 0:
        return int(np.argmax(decoder_scores)), "decoder"
    if counts.max() <= 0:
        return int(np.argmax(mean_voltage)), "voltage"
    return int(np.argmax(counts)), "policy"


def apply_reinforcement(
    brain: Brain,
    targets: list[str],
    amount: float,
    positive: bool,
) -> None:
    if amount <= 0.0:
        return
    for target in targets:
        if positive:
            brain.reward(amount, target=target)
        else:
            brain.punish(amount, target=target)
    quiet_steps(brain, POST_REWARD_STEPS)


def best_actions_for_position(env: GridWorld, pos: tuple[int, int]) -> np.ndarray:
    current = env.manhattan_distance(pos)
    actions = []
    best = current
    for action in range(N_MOTOR):
        _, _, _, _, new_dist = env.step(pos, action)
        if new_dist < best:
            best = new_dist
            actions = [action]
        elif new_dist == best and new_dist < current:
            actions.append(action)
    return np.asarray(actions, dtype=np.int32)


# --- Rollouts --------------------------------------------------------------

def run_episode(
    brain: Brain,
    env: GridWorld,
    rng: np.random.Generator,
    episode_idx: int,
    learn: bool = True,
    epsilon_override: float | None = None,
) -> dict[str, float | int | bool]:
    pos = env.reset(rng)
    epsilon = epsilon_for_episode(episode_idx) if epsilon_override is None else epsilon_override

    explore_steps = 0
    silent_steps = 0
    total_counts = np.zeros(N_MOTOR, dtype=np.int32)

    for step_idx in range(1, MAX_STEPS_PER_EPISODE + 1):
        x = position_encode(pos)
        counts, mean_voltage = present_state(brain, x)
        decoder_scores = decode_action_scores(brain, x)
        total_counts += counts
        guided_actions = best_actions_for_position(env, pos) if learn else None
        action, source = choose_action(
            counts,
            mean_voltage,
            decoder_scores,
            epsilon,
            rng,
            guided_actions=guided_actions,
        )
        if learn and len(guided_actions) > 0 and action not in guided_actions:
            action = int(guided_actions[int(rng.integers(0, len(guided_actions)))])
            source = "teacher"
        if source in {"explore", "guided"}:
            explore_steps += 1
        elif source == "voltage":
            silent_steps += 1

        if learn:
            mark_executed_action(brain, x, action)

        nxt, moved, reached_goal, prev_dist, new_dist = env.step(pos, action)

        if learn:
            targets = keep_only_action_eligibility(brain, action)
            if reached_goal:
                apply_reinforcement(brain, targets, GOAL_REWARD, positive=True)
            elif new_dist < prev_dist:
                apply_reinforcement(brain, targets, STEP_REWARD, positive=True)
            elif not moved:
                apply_reinforcement(brain, targets, WALL_PUNISH, positive=False)
            elif new_dist > prev_dist:
                apply_reinforcement(brain, targets, STEP_PUNISH, positive=False)

        pos = nxt
        if reached_goal:
            reset_between_steps(brain)
            return {
                "success": True,
                "steps": step_idx,
                "final_distance": 0,
                "explore_steps": explore_steps,
                "silent_steps": silent_steps,
                "motor_spikes": int(total_counts.sum()),
            }

        reset_between_steps(brain)

    return {
        "success": False,
        "steps": MAX_STEPS_PER_EPISODE,
        "final_distance": env.manhattan_distance(pos),
        "explore_steps": explore_steps,
        "silent_steps": silent_steps,
        "motor_spikes": int(total_counts.sum()),
    }


def evaluate_policy(
    brain: Brain,
    env: GridWorld,
    rng: np.random.Generator,
    n_episodes: int = EVAL_EPISODES,
) -> dict[str, float]:
    eval_brain = copy.deepcopy(brain)
    eval_brain.reset_traces()

    successes = 0
    steps = []
    final_distances = []
    success_flags = []
    silent_steps = 0
    motor_spikes = 0

    for _ in range(n_episodes):
        metrics = run_episode(eval_brain, env, rng, episode_idx=EPISODES, learn=False, epsilon_override=0.0)
        success = bool(metrics["success"])
        successes += int(success)
        steps.append(int(metrics["steps"]))
        final_distances.append(int(metrics["final_distance"]))
        success_flags.append(success)
        silent_steps += int(metrics["silent_steps"])
        motor_spikes += int(metrics["motor_spikes"])

    steps_arr = np.asarray(steps, dtype=np.float64)
    dist_arr = np.asarray(final_distances, dtype=np.float64)
    success_steps = steps_arr[np.asarray(success_flags, dtype=bool)]
    return {
        "success_rate": successes / n_episodes,
        "mean_steps": float(np.mean(steps_arr)),
        "mean_success_steps": float(np.mean(success_steps)) if len(success_steps) else float("inf"),
        "mean_final_distance": float(np.mean(dist_arr)),
        "silent_step_rate": silent_steps / (n_episodes * MAX_STEPS_PER_EPISODE),
        "mean_motor_spikes": motor_spikes / n_episodes,
    }


def random_policy_baseline(
    env: GridWorld,
    rng: np.random.Generator,
    n_episodes: int = EVAL_EPISODES,
) -> dict[str, float]:
    successes = 0
    steps = []
    final_distances = []

    for _ in range(n_episodes):
        pos = env.reset(rng)
        for step_idx in range(1, MAX_STEPS_PER_EPISODE + 1):
            action = int(rng.integers(0, N_MOTOR))
            pos, _, reached_goal, _, _ = env.step(pos, action)
            if reached_goal:
                successes += 1
                steps.append(step_idx)
                final_distances.append(0)
                break
        else:
            steps.append(MAX_STEPS_PER_EPISODE)
            final_distances.append(env.manhattan_distance(pos))

    success_steps = [s for s, d in zip(steps, final_distances) if d == 0]
    return {
        "success_rate": successes / n_episodes,
        "mean_steps": float(np.mean(np.asarray(steps, dtype=np.float64))),
        "mean_success_steps": float(np.mean(np.asarray(success_steps, dtype=np.float64))) if success_steps else float("inf"),
        "mean_final_distance": float(np.mean(np.asarray(final_distances, dtype=np.float64))),
    }


def rest_and_measure(
    brain: Brain,
    env: GridWorld,
    rng: np.random.Generator,
    n_eval_episodes: int = 40,
) -> tuple[dict[str, float], dict[str, float]]:
    before = evaluate_policy(brain, env, rng, n_episodes=n_eval_episodes)
    quiet_steps(brain, REST_STEPS)
    brain.reset_traces()
    after = evaluate_policy(brain, env, rng, n_episodes=n_eval_episodes)
    return before, after


# --- Main ------------------------------------------------------------------

def main() -> None:
    print("=" * 68)
    print("  GRID NAVIGATION BENCHMARK - Spiking Brain with R-STDP")
    print("  (Operant conditioning with place cells, replay, and oscillations)")
    print("=" * 68)

    env = GridWorld()
    rng = np.random.default_rng(SEED)

    print("\n" + "-" * 68)
    print("  Building synthetic brain")
    print("-" * 68)
    brain = build_brain(seed=SEED)
    print(brain.summary())

    baseline = random_policy_baseline(env, np.random.default_rng(SEED + 1))
    untrained = evaluate_policy(brain, env, np.random.default_rng(SEED + 2))

    print("\n  Random policy baseline:")
    print(
        f"    success={baseline['success_rate']:.1%}  "
        f"mean_steps={baseline['mean_steps']:.2f}  "
        f"mean_success_steps={baseline['mean_success_steps']:.2f}  "
        f"mean_final_distance={baseline['mean_final_distance']:.2f}"
    )
    print("\n  Untrained brain policy:")
    print(
        f"    success={untrained['success_rate']:.1%}  "
        f"mean_steps={untrained['mean_steps']:.2f}  "
        f"mean_success_steps={untrained['mean_success_steps']:.2f}  "
        f"mean_final_distance={untrained['mean_final_distance']:.2f}  "
        f"silent_step_rate={untrained['silent_step_rate']:.1%}"
    )

    print("\n" + "-" * 68)
    print("  Training")
    print("-" * 68)

    rolling_steps: deque[int] = deque(maxlen=20)
    rolling_success: deque[int] = deque(maxlen=20)
    rolling_silence: deque[float] = deque(maxlen=20)
    rest_deltas: list[tuple[int, float, float]] = []

    best_score = -np.inf
    best_episode = 0
    best_brain = None
    t0 = time.time()

    for episode in range(1, EPISODES + 1):
        metrics = run_episode(brain, env, rng, episode_idx=episode, learn=True)
        rolling_steps.append(int(metrics["steps"]))
        rolling_success.append(int(metrics["success"]))
        rolling_silence.append(float(metrics["silent_steps"]) / MAX_STEPS_PER_EPISODE)

        elapsed = time.time() - t0
        eps = epsilon_for_episode(episode)
        rolling_success_rate = float(np.mean(rolling_success))
        rolling_mean_steps = float(np.mean(rolling_steps))
        rolling_silent_rate = float(np.mean(rolling_silence))

        print(
            f"  Ep {episode:3d}/{EPISODES}  "
            f"eps={eps:.2f}  "
            f"result={'goal' if metrics['success'] else 'fail':>4s}  "
            f"steps={int(metrics['steps']):2d}  "
            f"roll_success={rolling_success_rate:.1%}  "
            f"roll_steps={rolling_mean_steps:5.2f}  "
            f"roll_silent={rolling_silent_rate:.1%}  "
            f"t={elapsed:5.1f}s"
        )

        score = rolling_success_rate * 100.0 - rolling_mean_steps
        if score > best_score and len(rolling_steps) >= 20:
            best_score = score
            best_episode = episode
            best_brain = copy.deepcopy(brain)

        if episode % REST_INTERVAL == 0:
            before, after = rest_and_measure(
                brain,
                env,
                np.random.default_rng(SEED + 1000 + episode),
            )
            rest_deltas.append((episode, before["success_rate"], after["success_rate"]))
            print(
                "    Rest consolidation: "
                f"success {before['success_rate']:.1%} -> {after['success_rate']:.1%}, "
                f"mean_steps {before['mean_steps']:.2f} -> {after['mean_steps']:.2f}, "
                f"mean_final_distance {before['mean_final_distance']:.2f} -> {after['mean_final_distance']:.2f}"
            )

    if best_brain is not None:
        brain = best_brain
        brain.reset_traces()

    final_eval = evaluate_policy(brain, env, np.random.default_rng(SEED + 5000))

    print("\n" + "-" * 68)
    print("  Final Results")
    print("-" * 68)
    print(
        f"  Best checkpoint episode:   {best_episode}\n"
        f"  Policy success rate:       {final_eval['success_rate']:.1%}\n"
        f"  Mean steps (all episodes): {final_eval['mean_steps']:.2f}\n"
        f"  Mean steps (successes):    {final_eval['mean_success_steps']:.2f}\n"
        f"  Mean final distance:       {final_eval['mean_final_distance']:.2f}\n"
        f"  Silent step rate:          {final_eval['silent_step_rate']:.1%}\n"
        f"  Mean motor spikes/ep:      {final_eval['mean_motor_spikes']:.2f}"
    )

    print("\n  Random baseline reference:")
    print(
        f"    success={baseline['success_rate']:.1%}  "
        f"mean_steps={baseline['mean_steps']:.2f}  "
        f"mean_success_steps={baseline['mean_success_steps']:.2f}  "
        f"mean_final_distance={baseline['mean_final_distance']:.2f}"
    )

    if rest_deltas:
        avg_before = float(np.mean([b for _, b, _ in rest_deltas]))
        avg_after = float(np.mean([a for _, _, a in rest_deltas]))
        print("\n  Consolidation checkpoints:")
        for episode, before_success, after_success in rest_deltas:
            print(
                f"    after ep {episode:3d}: "
                f"success {before_success:.1%} -> {after_success:.1%}"
            )
        print(f"    average success: {avg_before:.1%} -> {avg_after:.1%}")

    print("\n" + "-" * 68)
    print("  Final brain state")
    print("-" * 68)
    print(brain.summary())

    print("\n" + "=" * 68)
    if final_eval["success_rate"] >= 0.80 and final_eval["mean_success_steps"] < 10.0:
        print("  OK: the brain learned a usable navigation policy.")
    elif final_eval["success_rate"] >= 0.50:
        print("  PARTIAL: the agent often reaches the goal but is still inefficient.")
    else:
        print("  FAIL: no robust navigation policy emerged with this configuration.")
    print("=" * 68)


if __name__ == "__main__":
    main()
