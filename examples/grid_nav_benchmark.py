"""
Grid navigation benchmark for the synthetic brain.

This is the first reinforcement-learning benchmark in the repo: the
network must learn which action to take from each position in a 5x5
grid to reach a fixed goal. Learning uses reward-modulated STDP, not
backpropagation.

Architecture:
    input (25 sensory neurons, one per grid cell)
        ↓ density 1.0 (R-STDP)
    motor (4 chattering neurons: up/down/left/right)

The baseline deliberately disables replay, oscillations and synaptic
homeostatic scaling. Those mechanisms should be reintroduced only as
separate ablations after direct state-action learning is established.

Unlike the Iris benchmark, this task is not supervised. The agent
chooses an action from motor spikes, executes it in the environment,
and then receives reward or punishment depending on whether that move
reduced the distance to the goal.

Evaluation freezes the network and resets transient state between episodes,
while retaining neural history within each path. Starts are fixed before any
rollout so a policy's path length cannot change the next starting position.
Repeated deterministic starts may reuse complete episode metrics. --output
NEW_DIR records starts, results and immutable checkpoints; CLI resume is not
implemented. The final exhaustive check covers all 24 non-goal cells.

Success is reported separately for paths driven entirely by spikes and paths
that used voltage fallback at least once. --selection-metric can select either
total or spike-only success (score: 100 * rate - mean steps). The default is
spike-only success. The calibrated readout uses transmission gain 512 and no
tonic motor current. The former baseline remains available with gain 1, motor
baseline current 4 and selection metric success_rate.
New recorded runs include a source_snapshot directory alongside source hashes.
"""

from __future__ import annotations

import copy
import argparse
import hashlib
import math
import operator
import os
import sys
import time
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples._utils import (quiet_steps, inference_brain, reset_independent_state,
                             can_reuse_independent_inference)
from examples.training_checkpoint import atomic_json, save_training_checkpoint
from src.brain import Brain
from src.device import DEVICE
from src.neuron import FiringPattern, NeuronType
from src.region import RegionType


# --- Task setup ------------------------------------------------------------

GRID_SIZE = 5
N_INPUT = GRID_SIZE * GRID_SIZE
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
MOTOR_BASELINE_CURRENT = 0.0
POST_REWARD_STEPS = 10
INTER_STEP_REST = 25

MAX_STEPS_PER_EPISODE = 20
EPISODES = 500
EVAL_EPISODES = 100
CHECKPOINT_INTERVAL = 50
CHECKPOINT_EVAL_EPISODES = 40

EPSILON_START = 0.50
EPSILON_END = 0.10
EPSILON_DECAY_EPISODES = 300

STEP_REWARD = 0.080
STEP_PUNISH = 0.030
WALL_PUNISH = 0.015
GOAL_REWARD = 0.400
REWARD_BASELINE_ALPHA = 0.05

LATERAL_INHIBITION = 5.0
ENCODER_MAX_CURRENT = 55.0
ENCODER_NOISE = 0.05

PROJECTION_DENSITY_INPUT_MOTOR = 1.00
READOUT_INIT_WEIGHT = 0.05
READOUT_TRANSMISSION_GAIN = 512.0
SELECTION_METRIC = "spike_only_success_rate"

STDP_SCALE = 1.0
TAU_ELIGIBILITY = 200.0
DOPAMINE_DECAY = 0.5

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

def build_brain(seed: int = SEED, *, transmission_gain: float | None = None) -> Brain:
    transmission_gain = READOUT_TRANSMISSION_GAIN if transmission_gain is None else float(transmission_gain)
    if not math.isfinite(transmission_gain) or transmission_gain < 0:
        raise ValueError("Transmission gain must be finite and nonnegative")
    brain = Brain(dt=1.0, seed=seed)

    input_region = brain.add_region(
        "input",
        RegionType.SENSORY,
        n_neurons=0,
        connectivity=0.0,
        max_neurons=N_INPUT,
    )
    # Each one-hot state must have a pathway to every motor neuron.
    # Long-range projections originate only from excitatory neurons.
    for _ in range(N_INPUT):
        input_region.add_neuron(NeuronType.EXCITATORY, FiringPattern.REGULAR_SPIKING)
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

    brain.connect_regions("input", "motor", density=PROJECTION_DENSITY_INPUT_MOTOR)

    for proj in brain.projections:
        ns = proj.n_synapses
        if proj.target_name == "motor":
            proj.syn_weight[:ns] = READOUT_INIT_WEIGHT
            if transmission_gain != 1.0:
                # Scale transmitted current, not plasticity amplitudes or
                # weight bounds. Gain one retains the original substrate.
                proj.syn_modulation[:ns].mul_(transmission_gain)

    # Keep plasticity focused on state -> action pathways.
    brain.freeze_plasticity()
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

    # Establish the minimal reward-learning baseline first.
    brain.freeze_structural_plasticity()
    brain.freeze_homeostatic_scaling()
    brain.disable_memory()
    brain.disable_oscillations()

    return brain


# --- Helpers ---------------------------------------------------------------


def reset_between_steps(brain: Brain, n_steps: int | None = None) -> None:
    quiet_steps(brain, INTER_STEP_REST if n_steps is None else n_steps)
    brain.reset_traces()


def position_encode(
    pos: tuple[int, int],
    size: int = GRID_SIZE,
) -> np.ndarray:
    row, col = pos
    encoded = np.zeros(size * size, dtype=np.float64)
    encoded[row * size + col] = 1.0
    return encoded


def present_state(
    brain: Brain,
    x: np.ndarray,
    n_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = STATE_PRESENT_STEPS if n_steps is None else n_steps
    if n_steps < 1:
        raise ValueError("State presentation steps must be positive")
    stimulus = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    motor_indices = torch.arange(N_MOTOR, dtype=torch.int64, device=DEVICE)
    motor = brain.regions["motor"]
    before = motor.total_spikes[:N_MOTOR].clone()
    voltage_sum = torch.zeros(N_MOTOR, dtype=torch.float32, device=DEVICE)
    for _ in range(n_steps):
        brain.stimulate("input", stimulus)
        brain.inject_current("motor", motor_indices, MOTOR_BASELINE_CURRENT)
        brain.step()
        voltage_sum += motor.v[:N_MOTOR]
    counts = (motor.total_spikes[:N_MOTOR] - before).cpu().numpy()
    mean_voltage = (voltage_sum / n_steps).cpu().numpy()
    return counts, mean_voltage


def mark_executed_action(brain: Brain, x: np.ndarray, action: int) -> None:
    # Exploration needs an explicit state-action trace; otherwise a randomly
    # chosen action may leave almost no eligibility on the executed pathway.
    stimulus = torch.as_tensor(x, dtype=torch.float32, device=DEVICE)
    motor_indices = torch.arange(N_MOTOR, dtype=torch.int64, device=DEVICE)
    action_indices = torch.tensor([action], dtype=torch.int64, device=DEVICE)
    for _ in range(ACTION_MARK_STEPS):
        brain.stimulate("input", stimulus)
        brain.inject_current("motor", motor_indices, MOTOR_BASELINE_CURRENT)
        brain.inject_current("motor", action_indices, ACTION_MARK_CURRENT)
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
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[int, str]:
    if rng.random() < epsilon:
        return int(rng.integers(0, N_MOTOR)), "explore"
    if counts.max() > 0:
        return int(np.argmax(counts)), "policy"
    return int(np.argmax(mean_voltage)), "voltage"


def apply_reinforcement(
    brain: Brain,
    targets: list[str],
    amount: float,
    positive: bool,
    action: int,
) -> dict[str, float]:
    if amount <= 0.0:
        return {
            "eligibility_mean": 0.0,
            "eligibility_positive_fraction": 0.0,
            "mean_abs_weight_change": 0.0,
            "saturated_fraction": 0.0,
        }

    selected = []
    eligibility_parts = []
    for proj in brain.projections:
        target = Brain.projection_target(proj.source_name, proj.target_name)
        if target not in targets:
            continue
        ns = proj.n_synapses
        mask = (proj.syn_post[:ns] == action) & proj.syn_alive[:ns]
        selected.append((proj, mask, proj.syn_weight[:ns][mask].clone()))
        if torch.any(mask):
            eligibility_parts.append(proj.syn_eligibility[:ns][mask].clone())

    eligibility = (
        torch.cat(eligibility_parts)
        if eligibility_parts else torch.empty(0, device=DEVICE)
    )
    # New spikes may arrive while dopamine is active. Keep credit restricted
    # throughout the pulse, including those newly accumulated traces.
    with ExitStack() as credit:
        for target in targets:
            credit.enter_context(
                brain.reward_stdp.restrict_to_posts(target, [action])
            )
            if positive:
                brain.reward(amount, target=target)
            else:
                brain.punish(amount, target=target)
        quiet_steps(brain, POST_REWARD_STEPS)

    weight_changes = []
    saturated = 0
    n_selected = 0
    for proj, mask, before in selected:
        ns = proj.n_synapses
        after = proj.syn_weight[:ns][mask]
        weight_changes.append(torch.abs(after - before))
        mins = proj.syn_min_weight[:ns][mask]
        maxs = proj.syn_max_weight[:ns][mask]
        saturated += int(
            ((after <= mins + 1e-6) | (after >= maxs - 1e-6)).sum().item()
        )
        n_selected += len(after)

    changes = (
        torch.cat(weight_changes)
        if weight_changes else torch.empty(0, device=DEVICE)
    )
    return {
        "eligibility_mean": float(eligibility.mean().item()) if len(eligibility) else 0.0,
        "eligibility_positive_fraction": (
            float((eligibility > 0).to(torch.float32).mean().item())
            if len(eligibility) else 0.0
        ),
        "mean_abs_weight_change": float(changes.mean().item()) if len(changes) else 0.0,
        "saturated_fraction": saturated / max(n_selected, 1),
    }


# --- Rollouts --------------------------------------------------------------

def run_episode(
    brain: Brain,
    env: GridWorld,
    rng: np.random.Generator,
    episode_idx: int,
    learn: bool = True,
    epsilon_override: float | None = None,
    reward_baseline: np.ndarray | None = None,
    *, start_pos: tuple[int, int] | None = None,
) -> dict[str, float | int | bool]:
    pos = env.reset(rng) if start_pos is None else _validate_start(env, start_pos)
    epsilon = epsilon_for_episode(episode_idx) if epsilon_override is None else epsilon_override

    explore_steps = 0
    silent_steps = 0
    total_counts = np.zeros(N_MOTOR, dtype=np.int32)
    reinforcement_stats: list[dict[str, float]] = []

    for step_idx in range(1, MAX_STEPS_PER_EPISODE + 1):
        x = position_encode(pos)
        counts, mean_voltage = present_state(brain, x)
        total_counts += counts
        action, source = choose_action(
            counts,
            mean_voltage,
            epsilon,
            rng,
        )
        if source == "explore":
            explore_steps += 1
        if counts.max() == 0:
            silent_steps += 1

        if learn:
            mark_executed_action(brain, x, action)

        nxt, moved, reached_goal, prev_dist, new_dist = env.step(pos, action)

        if learn:
            targets = keep_only_action_eligibility(brain, action)
            signed_reward = 0.0
            if reached_goal:
                signed_reward = GOAL_REWARD
            elif new_dist < prev_dist:
                signed_reward = STEP_REWARD
            elif not moved:
                signed_reward = -WALL_PUNISH
            elif new_dist > prev_dist:
                signed_reward = -STEP_PUNISH

            if reward_baseline is not None:
                state_idx = pos[0] * GRID_SIZE + pos[1]
                advantage = signed_reward - reward_baseline[state_idx]
                reward_baseline[state_idx] += REWARD_BASELINE_ALPHA * advantage
            else:
                advantage = signed_reward
            reinforcement_stats.append(apply_reinforcement(
                brain,
                targets,
                abs(float(advantage)),
                positive=advantage >= 0.0,
                action=action,
            ))

        pos = nxt
        if reached_goal:
            reset_between_steps(brain)
            result = {
                "success": True,
                "steps": step_idx,
                "final_distance": 0,
                "explore_steps": explore_steps,
                "silent_steps": silent_steps,
                "motor_spikes": int(total_counts.sum()),
            }
            return _attach_reinforcement_summary(result, reinforcement_stats)

        reset_between_steps(brain)

    result = {
        "success": False,
        "steps": MAX_STEPS_PER_EPISODE,
        "final_distance": env.manhattan_distance(pos),
        "explore_steps": explore_steps,
        "silent_steps": silent_steps,
        "motor_spikes": int(total_counts.sum()),
    }
    return _attach_reinforcement_summary(result, reinforcement_stats)


def _attach_reinforcement_summary(
    result: dict[str, float | int | bool],
    stats: list[dict[str, float]],
) -> dict[str, float | int | bool]:
    for key in (
        "eligibility_mean",
        "eligibility_positive_fraction",
        "mean_abs_weight_change",
        "saturated_fraction",
    ):
        result[key] = float(np.mean([item[key] for item in stats])) if stats else 0.0
    return result


def _validate_start(env, position):
    try:
        pos = tuple(operator.index(coordinate) for coordinate in position)
    except (TypeError, ValueError) as error:
        raise ValueError("Start position must contain two integer coordinates") from error
    if len(pos) != 2 or pos == env.goal or any(not 0 <= value < env.size for value in pos):
        raise ValueError("Start position must be inside the grid and outside the goal")
    return pos


def evaluation_starts(env, rng, n_episodes=None, start_positions=None):
    if n_episodes is None:
        n_episodes = EVAL_EPISODES if start_positions is None else len(start_positions)
    n_episodes = operator.index(n_episodes)
    if n_episodes < 1:
        raise ValueError("Evaluation episode count must be positive")
    if start_positions is None:
        return [env.reset(rng) for _ in range(n_episodes)]
    if len(start_positions) != n_episodes:
        raise ValueError("Start positions must match the requested episode count")
    return [_validate_start(env, position) for position in start_positions]


def evaluate_policy(
    brain: Brain,
    env: GridWorld,
    rng: np.random.Generator,
    n_episodes: int | None = None,
    *, start_positions=None, independent: bool = True, fast: bool = True,
) -> dict[str, float]:
    if independent:
        eval_brain = inference_brain(brain)
        positions = evaluation_starts(env, rng, n_episodes, start_positions)
        # Draw every start before the rollout. Its length must not alter the
        # next initial position, nor should memoization advance the caller RNG.
        rollout_rng = copy.deepcopy(rng)
    else:
        eval_brain = copy.deepcopy(brain)
        eval_brain.reset_traces()
        eval_brain.freeze_adaptive_thresholds()
        eval_brain.encoder.noise_level = 0.0
        if start_positions is not None:
            positions = evaluation_starts(env, rng, n_episodes, start_positions)
        else:
            n_episodes = EVAL_EPISODES if n_episodes is None else operator.index(n_episodes)
            if n_episodes < 1:
                raise ValueError("Evaluation episode count must be positive")
            positions = [None] * n_episodes
        rollout_rng = rng
    n_episodes = len(positions)
    reuse = independent and fast and can_reuse_independent_inference(eval_brain)
    cached = {}

    successes = 0
    spike_only_successes = 0
    steps = []
    final_distances = []
    success_flags = []
    silent_steps = 0
    motor_spikes = 0

    for position in positions:
        if reuse and position in cached:
            metrics = cached[position]
        else:
            if independent:
                reset_independent_state(eval_brain)
            metrics = run_episode(eval_brain, env, rollout_rng, episode_idx=EPISODES,
                                  learn=False, epsilon_override=0.0, start_pos=position)
            if reuse:
                cached[position] = metrics
        success = bool(metrics["success"])
        successes += int(success)
        spike_only_successes += int(success and int(metrics["silent_steps"]) == 0)
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
        "spike_only_success_rate": spike_only_successes / n_episodes,
        "fallback_assisted_success_rate": (successes - spike_only_successes) / n_episodes,
        "mean_steps": float(np.mean(steps_arr)),
        "mean_success_steps": float(np.mean(success_steps)) if len(success_steps) else float("inf"),
        "mean_final_distance": float(np.mean(dist_arr)),
        "silent_step_rate": silent_steps / sum(steps),
        "mean_motor_spikes": motor_spikes / n_episodes,
    }


def random_policy_baseline(
    env: GridWorld,
    rng: np.random.Generator,
    n_episodes: int | None = None,
    *, start_positions=None,
) -> dict[str, float]:
    successes = 0
    steps = []
    final_distances = []

    positions = evaluation_starts(env, rng, n_episodes, start_positions)
    n_episodes = len(positions)
    for pos in positions:
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


def policy_action_margin(brain: Brain, env: GridWorld) -> dict[str, float]:
    """Measure action separation when each state starts with empty transients."""
    eval_brain = inference_brain(brain)
    margins = []
    for row in range(env.size):
        for col in range(env.size):
            if (row, col) == env.goal:
                continue
            reset_independent_state(eval_brain)
            counts, voltage = present_state(eval_brain, position_encode((row, col)))
            scores = counts.astype(np.float64) if counts.max() > 0 else voltage
            ordered = np.sort(scores)
            margins.append(float(ordered[-1] - ordered[-2]))
            reset_between_steps(eval_brain)
    return {
        "mean": float(np.mean(margins)),
        "minimum": float(np.min(margins)),
    }


# --- Main ------------------------------------------------------------------

def json_safe(value):
    """Represent unavailable statistics as null, never nonstandard JSON Infinity."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main(*, output: Path | None = None):
    if SELECTION_METRIC not in ("success_rate", "spike_only_success_rate"):
        raise ValueError("Unknown checkpoint selection metric")
    if any(operator.index(value) < 1 for value in (
            EPISODES, CHECKPOINT_INTERVAL, EVAL_EPISODES, CHECKPOINT_EVAL_EPISODES)):
        raise ValueError("Episode and checkpoint counts must be positive")
    if not math.isfinite(MOTOR_BASELINE_CURRENT) or MOTOR_BASELINE_CURRENT < 0:
        raise ValueError("Motor baseline current must be finite and nonnegative")
    if not math.isfinite(READOUT_TRANSMISSION_GAIN) or READOUT_TRANSMISSION_GAIN < 0:
        raise ValueError("Transmission gain must be finite and nonnegative")
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/_utils.py", root / "examples/training_checkpoint.py",
               *sorted((root / "src").glob("*.py"))]
    source_bytes = {str(p.relative_to(root)): p.read_bytes() for p in sources}
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in source_bytes.items()}
    config = {name: value for name, value in globals().items()
              if name.isupper() and not name.startswith("_") and isinstance(value, (int, float, str, bool))}
    payload = {"complete": False, "pid": os.getpid(), "started_at": time.time(),
               "source_sha256": hashes, "config": config, "torch": torch.__version__,
               "numpy": np.__version__, "episodes": [], "checkpoints": [],
               "inference": "independent frozen episodes; within-episode neural history retained; duplicate starts reused"}
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        for name, content in source_bytes.items():
            destination = output / "source_snapshot" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)

    def publish():
        if output is not None:
            atomic_json(output / "summary.json", json_safe(payload))

    def save_checkpoint(label, model, progress):
        if output is not None:
            save_training_checkpoint(model, output / "checkpoints" / label,
                json_safe({"protocol": {"config": config, "source_sha256": hashes,
                                        "evaluation": payload["evaluation"]}, **progress}))

    publish()
    print("=" * 68)
    print("  GRID NAVIGATION BENCHMARK - Spiking Brain with R-STDP")
    print("  (Minimal teacher-free state-action baseline)")
    print(f"  Transmission gain={READOUT_TRANSMISSION_GAIN:g}, motor baseline={MOTOR_BASELINE_CURRENT:g}, "
          f"selection={SELECTION_METRIC}")
    print("=" * 68)

    env = GridWorld()
    rng = np.random.default_rng(SEED)
    final_starts = evaluation_starts(env, np.random.default_rng(SEED + 5000), EVAL_EPISODES)
    selection_starts = evaluation_starts(env, np.random.default_rng(SEED + 2000), CHECKPOINT_EVAL_EPISODES)
    payload["evaluation"] = {"size": env.size, "goal": list(env.goal),
                             "action_deltas": ACTION_DELTAS.tolist(),
                             "selection_starts": selection_starts, "final_starts": final_starts}
    publish()

    print("\n" + "-" * 68)
    print("  Building synthetic brain")
    print("-" * 68)
    brain = build_brain(seed=SEED)
    print(brain.summary())

    save_checkpoint("initial", brain, {"completed_episodes": 0})
    baseline = random_policy_baseline(env, np.random.default_rng(SEED + 1), start_positions=final_starts)
    untrained = evaluate_policy(brain, env, np.random.default_rng(SEED + 2), start_positions=final_starts)
    payload.update(random_baseline=baseline, initial_policy=untrained)
    publish()

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
        f"spike_only={untrained['spike_only_success_rate']:.1%}  "
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
    reward_baseline = np.zeros(N_INPUT, dtype=np.float64)
    checkpoint_history: list[dict[str, float]] = []

    best_score = -np.inf
    best_episode = 0
    best_brain = None
    t0 = time.time()

    for episode in range(1, EPISODES + 1):
        metrics = run_episode(
            brain,
            env,
            rng,
            episode_idx=episode,
            learn=True,
            reward_baseline=reward_baseline,
        )
        rolling_steps.append(int(metrics["steps"]))
        rolling_success.append(int(metrics["success"]))
        rolling_silence.append(float(metrics["silent_steps"]) / int(metrics["steps"]))
        payload["episodes"].append({"episode": episode, **metrics})

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
            f"elig+={float(metrics['eligibility_positive_fraction']):.1%}  "
            f"|dw|={float(metrics['mean_abs_weight_change']):.5f}  "
            f"sat={float(metrics['saturated_fraction']):.1%}  "
            f"t={elapsed:5.1f}s"
        )

        if episode % CHECKPOINT_INTERVAL == 0 or episode == EPISODES:
            checkpoint = evaluate_policy(
                brain,
                env,
                np.random.default_rng(SEED + 2000),
                n_episodes=CHECKPOINT_EVAL_EPISODES,
                start_positions=selection_starts,
            )
            margin = policy_action_margin(brain, env)
            checkpoint["episode"] = float(episode)
            checkpoint["mean_action_margin"] = margin["mean"]
            checkpoint["minimum_action_margin"] = margin["minimum"]
            checkpoint_history.append(checkpoint)
            score = checkpoint[SELECTION_METRIC] * 100.0 - checkpoint["mean_steps"]
            if score > best_score:
                best_score = score
                best_episode = episode
                best_brain = copy.deepcopy(brain)
            payload.update(checkpoints=checkpoint_history, best_episode=best_episode, best_score=best_score)
            save_checkpoint(f"episode_{episode:06d}", brain,
                {"completed_episodes": episode, "rollout_rng_state": rng.bit_generator.state,
                 "reward_baseline": reward_baseline.tolist(), "best_episode": best_episode,
                 "checkpoint_history": checkpoint_history, "rolling_steps": list(rolling_steps),
                 "rolling_success": list(rolling_success), "rolling_silence": list(rolling_silence)})
            print(
                "    Deterministic checkpoint: "
                f"success={checkpoint['success_rate']:.1%}  "
                f"spike_only={checkpoint['spike_only_success_rate']:.1%}  "
                f"mean_steps={checkpoint['mean_steps']:.2f}  "
                f"margin_mean={margin['mean']:.3f}  "
                f"margin_min={margin['minimum']:.3f}"
            )
        publish()

    if best_brain is not None:
        brain = best_brain
        brain.reset_traces()

    final_eval = evaluate_policy(brain, env, np.random.default_rng(SEED + 5000), start_positions=final_starts)
    all_starts = [(row, col) for row in range(env.size) for col in range(env.size) if (row, col) != env.goal]
    exhaustive = evaluate_policy(brain, env, np.random.default_rng(SEED + 6000), start_positions=all_starts)
    payload.update(final_policy=final_eval, exhaustive_policy=exhaustive, exhaustive_starts=all_starts)
    save_checkpoint("selected", brain, {"completed_episodes": best_episode, "selected_by": "checkpoint_starts"})

    print("\n" + "-" * 68)
    print("  Final Results")
    print("-" * 68)
    print(
        f"  Best checkpoint episode:   {best_episode}\n"
        f"  Policy success rate:       {final_eval['success_rate']:.1%}\n"
        f"  Spike-only success rate:   {final_eval['spike_only_success_rate']:.1%}\n"
        f"  Fallback-assisted success: {final_eval['fallback_assisted_success_rate']:.1%}\n"
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

    if checkpoint_history:
        print("\n  Deterministic checkpoint history:")
        for checkpoint in checkpoint_history:
            print(
                f"    ep {int(checkpoint['episode']):3d}: "
                f"success={checkpoint['success_rate']:.1%}  "
                f"spike_only={checkpoint['spike_only_success_rate']:.1%}  "
                f"mean_steps={checkpoint['mean_steps']:.2f}  "
                f"margin={checkpoint['mean_action_margin']:.3f}"
            )

    print("\n" + "-" * 68)
    print("  Final brain state")
    print("-" * 68)
    print(brain.summary())

    print("\n" + "=" * 68)
    if final_eval["spike_only_success_rate"] >= 0.80 and final_eval["mean_success_steps"] < 10.0:
        print("  OK: the brain learned a usable spike-driven navigation policy.")
    elif final_eval["success_rate"] >= 0.50:
        print("  PARTIAL: the agent often reaches the goal but is still inefficient.")
    else:
        print("  FAIL: no robust navigation policy emerged with this configuration.")
    print("=" * 68)
    if hashes != {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}:
        raise RuntimeError("Benchmark sources changed during the run")
    payload.update(complete=True, finished_at=time.time(), source_unchanged=True)
    publish()
    return json_safe(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New result/checkpoint directory; no CLI resume support")
    parser.add_argument("--transmission-gain", type=float, default=READOUT_TRANSMISSION_GAIN,
                        help="Experimental input-to-motor current gain; weight bounds and STDP amplitudes stay unchanged")
    parser.add_argument("--motor-baseline-current", type=float, default=MOTOR_BASELINE_CURRENT)
    parser.add_argument("--selection-metric", choices=("success_rate", "spike_only_success_rate"), default=SELECTION_METRIC)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    READOUT_TRANSMISSION_GAIN = args.transmission_gain
    MOTOR_BASELINE_CURRENT = args.motor_baseline_current
    SELECTION_METRIC = args.selection_metric
    EPISODES = args.episodes
    SEED = args.seed
    main(output=args.output)
