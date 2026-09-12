"""Measure the two pair-STDP arms along a saved model's private training trajectory.

Each step is replayed on private weights with LTP then LTD in the production
order. The combined result must match production bit for bit. Arm totals measure
realized, bounded updates; they do not predict an alternative rule's accuracy.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import examples.mnist_benchmark as mn
from examples.mnist_learning_check import LearningConfig, prepare_dataset, weight_delta_metrics
from examples.mnist_optimization_check import simulation_digest
from examples.training_checkpoint import atomic_json, directory_digest, load_training_checkpoint


def replay_arms(brain, before):
    """Return weights after LTP and after LTD without changing network state."""
    projection = brain.get_projection("input", "cortex")
    source, target = brain.regions["input"], brain.regions["cortex"]
    ns = projection.n_synapses
    weights = before.clone()
    if not projection.plasticity_enabled:
        return weights.clone(), weights
    kwargs = dict(
        syn_pre=projection.syn_pre[:ns], syn_post=projection.syn_post[:ns],
        pre_last_spike_arr=source.last_spike_time[:source.n_neurons],
        post_last_spike_arr=target.last_spike_time[:target.n_neurons],
        A_plus=projection.syn_A_plus[:ns], A_minus=projection.syn_A_minus[:ns],
        alive=projection.syn_alive[:ns], weights=weights,
        min_weight=projection.syn_min_weight[:ns], max_weight=projection.syn_max_weight[:ns],
        current_time=brain.time,
    )
    # The dense reference also cross-checks the production event-index path.
    pre, post = source.fired[:source.n_neurons], target.fired[:target.n_neurons]
    brain.stdp.apply_event(fired_pre=torch.zeros_like(pre), fired_post=post, **kwargs)
    after_ltp = weights.clone()
    brain.stdp.apply_event(fired_pre=pre, fired_post=torch.zeros_like(post), **kwargs)
    return after_ltp, weights


def audit(brain, images, *, train_steps, rest_steps, per_neuron=False, learning_rule="pair"):
    if learning_rule != "pair":
        raise ValueError("Separate-arm audit supports only the pair learning rule")
    model = copy.deepcopy(brain)
    projection = model.get_projection("input", "cortex")
    weights = projection.syn_weight[:projection.n_synapses]
    norm_target = mn.compute_norm_target(model, per_neuron=per_neuron)
    rows = []
    for number, image in enumerate(images):
        initial = weights.clone()
        row = {"sample": number, "ltp_l1": 0., "ltd_l1": 0., "net_step_l1": 0.,
               "ltp_weight_steps": 0, "ltd_weight_steps": 0, "both_weight_steps": 0}
        stimulus = torch.as_tensor(image, dtype=torch.float32, device=weights.device)
        for _ in range(train_steps):
            model.stimulate("input", stimulus)
            model.step()
            before = weights.clone()
            after_ltp, after_ltd = replay_arms(model, before)
            mn.apply_feedforward_stdp(model)
            if not torch.equal(weights.contiguous().view(torch.uint8), after_ltd.contiguous().view(torch.uint8)):
                raise AssertionError("Separated STDP arms differ from production")
            before64 = before.detach().cpu().double()
            ltp64 = after_ltp.detach().cpu().double()
            ltd64 = after_ltd.detach().cpu().double()
            ltp = ltp64 - before64
            ltd = ltd64 - ltp64
            row["ltp_l1"] += float(ltp.abs().sum())
            row["ltd_l1"] += float(ltd.abs().sum())
            row["net_step_l1"] += float((ltd64 - before64).abs().sum())
            row["ltp_weight_steps"] += int(torch.count_nonzero(ltp))
            row["ltd_weight_steps"] += int(torch.count_nonzero(ltd))
            row["both_weight_steps"] += int(torch.count_nonzero((ltp != 0) & (ltd != 0)))
        after_stdp = weights.clone()
        mn.normalize_feedforward_weights(model, norm_target)
        row["sample_updates"] = weight_delta_metrics(initial, after_stdp, weights)
        row["stdp_signed_sum"] = float((after_stdp.detach().cpu().double()
                                       - initial.detach().cpu().double()).sum())
        mn.reset_brain_state(model, rest_steps=rest_steps)
        rows.append(row)
    return rows, simulation_digest(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1:
        parser.error("Use a new output and positive sample count")
    study = json.loads((args.study / "summary.json").read_text())
    if not study["complete"] or args.seed not in study["signature"]["seeds"]:
        parser.error("Choose a completed source study and one of its seeds")
    config = LearningConfig(**study["signature"]["config"])
    config.validate()
    if config.learning_rule != "pair":
        parser.error("Separate-arm audit supports only the pair learning rule")
    root = Path(__file__).resolve().parent.parent
    # Include the benchmark's recorded source set plus this diagnostic.
    sources = sorted(set(study["signature"]["source_sha256"]) | {str(Path(__file__).resolve().relative_to(root))})
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    checkpoint = args.study / f"seed_{args.seed}/initial/checkpoints/sample_00000000"
    checkpoint_hash = directory_digest(checkpoint)
    protocol = json.loads((checkpoint / "progress.json").read_text())["progress"]["protocol"]
    brain, _ = load_training_checkpoint(checkpoint, protocol)
    before = simulation_digest(brain)
    row = next(r for r in study["results"] if r["seed"] == args.seed and r["condition"] == "initial")
    if before != row["final_sha256"]:
        raise ValueError("Initial state differs from the completed study")
    raw = mn.fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff",
                         data_home=str(root / ".sklearn_data"))
    data = prepare_dataset(raw.data, np.asarray(raw.target, dtype=np.int64), config)
    del raw
    if data.manifest != study["dataset"]:
        raise ValueError("Source study dataset mismatch")
    if args.samples > len(data.train_X):
        parser.error("The audit covers at most the first epoch")
    order = np.random.default_rng(args.seed).permutation(len(data.train_X))[:args.samples]
    rows, final = audit(brain, data.train_X[order], train_steps=config.train_steps,
                        rest_steps=config.rest_steps, per_neuron=config.normalization_policy == "initial_per_neuron",
                        learning_rule=config.learning_rule)
    training_checkpoint = args.study / f"seed_{args.seed}/stdp_normalized/checkpoints/sample_{args.samples:08d}"
    training_checkpoint_hash = None
    if training_checkpoint.exists():
        training_checkpoint_hash = directory_digest(training_checkpoint)
        training_protocol = json.loads((training_checkpoint / "progress.json").read_text())["progress"]["protocol"]
        trained, _ = load_training_checkpoint(training_checkpoint, training_protocol)
        if final != simulation_digest(trained):
            raise AssertionError("Audited trajectory differs from the source training checkpoint")
        assert training_checkpoint_hash == directory_digest(training_checkpoint)
    totals = {key: sum(row[key] for row in rows) for key in rows[0]
              if key not in ("sample", "sample_updates")}
    assert before == simulation_digest(brain)
    assert checkpoint_hash == directory_digest(checkpoint)
    assert hashes == {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources}
    result = {"source_sha256": hashes, "source_study_sha256": hashlib.sha256(
                  (args.study / "summary.json").read_bytes()).hexdigest(),
              "checkpoint_sha256": checkpoint_hash, "source_unchanged": True,
              "arms_bitwise_equal": True, "final_training_sha256": final,
              "matching_training_checkpoint_sha256": training_checkpoint_hash,
              "seed": args.seed, "config": study["signature"]["config"],
              "train_ids": [data.manifest["train_ids"][i] for i in order], "rows": rows, "totals": totals,
              "interpretation": "Realized LTP/LTD magnitudes in production order, not unconstrained "
              "updates or an alternative training result. No validation or test labels used."}
    atomic_json(args.output, result)
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
