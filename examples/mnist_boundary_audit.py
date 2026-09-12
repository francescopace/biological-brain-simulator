"""Measure pair-STDP updates that use spike history from before the current image.

Counterfactual updates start from each observed step's weights and spike state.
They censor old timestamps only for the comparison, without changing the actual
training trajectory. This audit does not measure classifier accuracy.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig, prepare_dataset
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import atomic_json, directory_digest
from src.persistence import load_brain


def censored_update(brain, before, image_start):
    """Replay this step's rule on private weights, ignoring pre-image history."""
    proj = brain.get_projection("input", "cortex")
    src, dst = brain.regions["input"], brain.regions["cortex"]
    ns = proj.n_synapses
    pre = src.last_spike_time[:src.n_neurons]
    post = dst.last_spike_time[:dst.n_neurons]
    weights = before.clone()
    brain.stdp.apply_event(
        fired_pre=src.fired[:src.n_neurons], fired_post=dst.fired[:dst.n_neurons],
        syn_pre=proj.syn_pre[:ns], syn_post=proj.syn_post[:ns],
        pre_last_spike_arr=torch.where(pre > image_start, pre, -float("inf")),
        post_last_spike_arr=torch.where(post > image_start, post, -float("inf")),
        A_plus=proj.syn_A_plus[:ns], A_minus=proj.syn_A_minus[:ns], alive=proj.syn_alive[:ns],
        weights=weights, min_weight=proj.syn_min_weight[:ns], max_weight=proj.syn_max_weight[:ns],
        current_time=brain.time, event_indices=(proj._pre_events, proj._post_events),
    )
    return weights


def audit(brain, images, *, train_steps, rest_steps, learning_rule="pair"):
    if learning_rule != "pair":
        raise ValueError("Timestamp-boundary audit supports only the pair learning rule")
    model = copy.deepcopy(brain)
    proj = model.get_projection("input", "cortex")
    weights = proj.syn_weight[:proj.n_synapses]
    norm_target = mn.compute_norm_target(model)
    rows = []
    for number, image in enumerate(images):
        start = model.time
        row = {"sample": number, "observed_update_l1": 0., "pre_image_history_difference_l1": 0.,
               "changed_weight_step_pairs": 0, "steps_with_history_effect": 0}
        for _ in range(train_steps):
            model.stimulate("input", image)
            model.step()
            before = weights.clone()
            mn.apply_feedforward_stdp(model)
            censored = censored_update(model, before, start)
            difference = weights - censored
            changed = int(torch.count_nonzero(difference))
            row["observed_update_l1"] += float((weights - before).abs().double().sum())
            row["pre_image_history_difference_l1"] += float(difference.abs().double().sum())
            row["changed_weight_step_pairs"] += changed
            row["steps_with_history_effect"] += bool(changed)
        mn.normalize_feedforward_weights(model, norm_target)
        mn.reset_brain_state(model, rest_steps=rest_steps)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        parser.error("Use a new output and positive sample count")
    study = json.loads((args.study / "summary.json").read_text())
    config = LearningConfig(**study["signature"]["config"])
    config.validate()
    if config.learning_rule != "pair":
        parser.error("Timestamp-boundary audit supports only the pair learning rule")
    root = Path(__file__).resolve().parent.parent
    sources = [Path(__file__), root / "examples/mnist_benchmark.py", *sorted((root / "src").glob("*.py"))]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    checkpoint = args.study / "seed_101/initial/checkpoints/sample_00000000/brain"
    checkpoint_hash = directory_digest(checkpoint)
    brain = load_brain(checkpoint)
    before = simulation_digest(brain)
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    del raw
    assert data.manifest == study["dataset"]
    order = np.random.default_rng(101).permutation(len(data.train_X))[:args.samples]
    rows = audit(brain, data.train_X[order], train_steps=config.train_steps, rest_steps=config.rest_steps,
                 learning_rule=config.learning_rule)
    totals = {k: sum(r[k] for r in rows) for k in rows[0] if k != "sample"}
    assert before == simulation_digest(brain)
    assert checkpoint_hash == directory_digest(checkpoint)
    assert hashes == {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    result = {"source_sha256": hashes, "checkpoint_sha256": checkpoint_hash, "checkpoint_unchanged": True,
              "train_ids": [data.manifest["train_ids"][i] for i in order], "rows": rows, "totals": totals,
              "interpretation": "One-step counterfactual along the observed training trajectory. "
              "The L1 difference is not an additive attribution or a classifier gain."}
    atomic_json(args.output, result)
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
