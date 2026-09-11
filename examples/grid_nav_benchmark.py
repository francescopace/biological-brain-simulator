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
"""

from __future__ import annotations

import copy
import sys
import time
from collections import deque
from contextlib import ExitStack
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

def build_brain(seed: int = SEED) -> Brain:
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


def reset_between_steps(brain: Brain, n_steps: int = INTER_STEP_REST) -> None:
    quiet_steps(brain, n_steps)
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
) -> dict[str, float | int | bool]:
    pos = env.reset(rng)
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
        elif source == "voltage":
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


def evaluate_policy(
    brain: Brain,
    env: GridWorld,
    rng: np.random.Generator,
    n_episodes: int = EVAL_EPISODES,
) -> dict[str, float]:
    eval_brain = copy.deepcopy(brain)
    eval_brain.reset_traces()
    eval_brain.freeze_adaptive_thresholds()
    eval_brain.encoder.noise_level = 0.0

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


def policy_action_margin(brain: Brain, env: GridWorld) -> dict[str, float]:
    """Measure how decisively the spiking policy separates its top actions."""
    eval_brain = copy.deepcopy(brain)
    eval_brain.reset_traces()
    eval_brain.freeze_adaptive_thresholds()
    eval_brain.encoder.noise_level = 0.0
    margins = []
    for row in range(env.size):
        for col in range(env.size):
            if (row, col) == env.goal:
                continue
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

def main() -> None:
    print("=" * 68)
    print("  GRID NAVIGATION BENCHMARK - Spiking Brain with R-STDP")
    print("  (Minimal teacher-free state-action baseline)")
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
            f"elig+={float(metrics['eligibility_positive_fraction']):.1%}  "
            f"|dw|={float(metrics['mean_abs_weight_change']):.5f}  "
            f"sat={float(metrics['saturated_fraction']):.1%}  "
            f"t={elapsed:5.1f}s"
        )

        if episode % CHECKPOINT_INTERVAL == 0:
            checkpoint = evaluate_policy(
                brain,
                env,
                np.random.default_rng(SEED + 2000),
                n_episodes=CHECKPOINT_EVAL_EPISODES,
            )
            margin = policy_action_margin(brain, env)
            checkpoint["episode"] = float(episode)
            checkpoint["mean_action_margin"] = margin["mean"]
            checkpoint["minimum_action_margin"] = margin["minimum"]
            checkpoint_history.append(checkpoint)
            score = checkpoint["success_rate"] * 100.0 - checkpoint["mean_steps"]
            if score > best_score:
                best_score = score
                best_episode = episode
                best_brain = copy.deepcopy(brain)
            print(
                "    Deterministic checkpoint: "
                f"success={checkpoint['success_rate']:.1%}  "
                f"mean_steps={checkpoint['mean_steps']:.2f}  "
                f"margin_mean={margin['mean']:.3f}  "
                f"margin_min={margin['minimum']:.3f}"
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

    if checkpoint_history:
        print("\n  Deterministic checkpoint history:")
        for checkpoint in checkpoint_history:
            print(
                f"    ep {int(checkpoint['episode']):3d}: "
                f"success={checkpoint['success_rate']:.1%}  "
                f"mean_steps={checkpoint['mean_steps']:.2f}  "
                f"margin={checkpoint['mean_action_margin']:.3f}"
            )

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
