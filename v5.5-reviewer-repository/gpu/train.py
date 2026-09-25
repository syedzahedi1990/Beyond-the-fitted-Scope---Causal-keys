import argparse
import hashlib
import json
import time
from pathlib import Path

from gpu.behavior_engine import BehaviorEngine, datasets, encode_pair, mapped, require, schedule


def pca_basis(matrix, seed):
    import numpy as np
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    rank = min(16, *centered.shape)
    require(rank == 16, "Insufficient observations or width for rank 16")
    sketch_rank = min(rank + 16, *centered.shape)
    omega = np.random.default_rng(seed).standard_normal((centered.shape[1], sketch_rank)).astype(np.float32)
    sketch = centered @ omega
    sketch, _ = np.linalg.qr(sketch, mode="reduced")
    for _ in range(2):
        sketch = centered @ (centered.T @ sketch)
        sketch, _ = np.linalg.qr(sketch, mode="reduced")
    _, _, vectors = np.linalg.svd(sketch.T @ centered, full_matrices=False)
    return vectors[:rank].astype(np.float32)


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def inventory(directory):
    result = {}
    for file in sorted(Path(directory).rglob("*")):
        if file.is_file() and file.name not in ("COMPLETE.json", "FAILED.json"):
            with file.open("rb") as stream:
                sha = hashlib.file_digest(stream, "sha256").hexdigest()
            result[file.relative_to(directory).as_posix()] = {"sha256": sha, "size_bytes": file.stat().st_size}
    return result


def train(args):
    from reproduce.integrity import ROOT, digest, verify
    from gpu.runtime.load import load_engine
    verify()
    output = Path(args.output)
    require(not output.exists(), "Use a new output directory; fits are never overwritten")
    output.mkdir(parents=True)
    (output / "bases").mkdir()
    started = time.monotonic()
    objectives = ("f_star", "m3") if args.cohort == "original_1000" else ("m1", "m3")
    specification = {"model": args.model, "cohort": args.cohort, "seeds": [101, 102, 103],
                     "objectives": objectives, "rank": 16, "block": 4,
                     "learning_rate": 0.001, "weight_decay": 0, "gradient_clip_norm": 1,
                     "updates_per_fit": 1000 if args.cohort == "original_1000" else 300,
                     "sampling": "one_shuffled_epoch" if args.cohort == "original_1000" else "with_replacement",
                     "interface": "original", "model_weights_optimized": False,
                     "code_identity": "derived_reviewer_implementation", "release_sha256": digest(ROOT / "RELEASE.json"),
                     "scientific_success_gate": None}
    write(output / "RUN.json", specification)
    try:
        native = load_engine(args.model, model_path=args.model_path, native_call_limit=15000,
                             journal_path=output / "native_loader.jsonl", deadline_seconds=43200)
        engine = BehaviorEngine(native, args.model, 1024 if args.model == "mistral" else None)
        engine.journal = lambda row: append(output / "calls.jsonl", row)
        torch, np = engine.torch, engine.np
        write(output / "ENVIRONMENT.json", engine.environment)
        weights = [(p, p._version, p.data_ptr()) for p in engine.model.parameters()]
        pairs = datasets()["training"]
        require(len(pairs) == 1000, "The original training pool must contain 1000 pairs")
        cache, deltas = [], []
        for i, pair in enumerate(pairs):
            base, source = encode_pair(engine.tok, pair, "original", args.model, engine.padding)
            _, hb, prefix = engine.forward(base)
            _, hs, _ = engine.forward(source)
            cache.append((base, hb.cpu(), hs.cpu(), prefix))
            deltas.append(hs.float().cpu().numpy() - hb.float().cpu().numpy())
            if (i + 1) % 50 == 0:
                print(json.dumps({"stage": "cache", "pairs": i + 1, "total": 1000}), flush=True)
        matrix = np.concatenate(deltas, axis=0).astype(np.float32)
        del deltas
        basis_records = []
        for seed in (101, 102, 103):
            raw_pca = pca_basis(matrix, seed + 16)
            raw_cpu = torch.linalg.qr(torch.tensor(raw_pca.T, dtype=torch.float32), mode="reduced")[0]
            initial_parameter = raw_cpu.to(engine.device)
            initial = torch.linalg.qr(initial_parameter.float(), mode="reduced")[0].T.detach().cpu().numpy().astype(np.float32)
            np.savez(output / "bases" / f"pca_ts{seed}.npz", rank_16=initial)
            np.savez(output / "bases" / f"raw_pca_ts{seed}.npz", rank_16=raw_pca)
            pca_file = output / "bases" / f"pca_ts{seed}.npz"
            basis_records.append({"objective": "pca", "seed": seed, "file": "bases/" + pca_file.name,
                                  "sha256": hashlib.sha256(pca_file.read_bytes()).hexdigest(),
                                  "tensor_sha256": hashlib.sha256(initial.tobytes()).hexdigest(), "updates": 0,
                                  "initializer_identity": "exact_saved_this_rerun"})
            order = schedule(seed, args.cohort)
            write(output / f"schedule_ts{seed}.json", order)
            for objective in objectives:
                parameter = torch.nn.Parameter(initial_parameter.clone())
                optimizer = torch.optim.AdamW([parameter], lr=0.001, weight_decay=0)
                fit_id = f"{objective}_ts{seed}"
                start_basis = torch.linalg.qr(parameter.float(), mode="reduced")[0].T
                require(np.array_equal(start_basis.detach().cpu().numpy(), initial), "Paired effective initializers differ")
                for probe in (0, 1, 2):
                    encoded, hb, hs, prefix = cache[probe]
                    hb, hs = hb.to(engine.device), hs.to(engine.device)
                    optimizer.zero_grad(set_to_none=True)
                    basis = torch.linalg.qr(parameter.float(), mode="reduced")[0].T
                    scores = engine.forward(encoded, basis=basis, source=hs, expected=hb, expected_prefix=prefix, gradients=True)
                    actual = scores.detach().clone()
                    fixed = engine.fixed(hb, hs, basis.detach()).detach().requires_grad_(True)
                    repeated = engine.forward(encoded, fixed=fixed, expected=hb, expected_prefix=prefix, gradients=True)
                    require(float((actual - repeated.detach()).abs().max()) <= 1e-4, "Gradient and materialized forward scores differ")
                    label = mapped(pairs[probe]["source"]["answer"], objective)
                    target = torch.tensor([encoded["choices"].index(label)], device=engine.device)
                    loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0).float(), target)
                    loss.backward()
                    require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), "Probe gradient missing or nonfinite")
                    append(output / "gradient_checks.jsonl", {"fit": fit_id, "pair": probe, "loss": float(loss.detach()),
                                                              "gradient_norm": float(parameter.grad.float().norm()),
                                                              "score_max_difference": float((actual-repeated.detach()).abs().max())})
                for step, index in enumerate(order, 1):
                    encoded, hb, hs, prefix = cache[index]
                    optimizer.zero_grad(set_to_none=True)
                    basis = torch.linalg.qr(parameter.float(), mode="reduced")[0].T
                    scores = engine.forward(encoded, basis=basis, source=hs.to(engine.device), expected=hb.to(engine.device),
                                            expected_prefix=prefix, gradients=True)
                    label = mapped(pairs[index]["source"]["answer"], objective)
                    target = torch.tensor([encoded["choices"].index(label)], device=engine.device)
                    loss = torch.nn.functional.cross_entropy(scores.unsqueeze(0).float(), target)
                    require(bool(torch.isfinite(loss)), "Nonfinite fitting loss")
                    loss.backward()
                    require(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), "Missing or nonfinite fitting gradient")
                    gradient_norm = torch.nn.utils.clip_grad_norm_([parameter], 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    require(bool(torch.isfinite(parameter).all()), "Nonfinite fitted parameter")
                    append(output / "updates.jsonl", {"fit": fit_id, "step": step, "pair": index,
                                                       "loss": float(loss.detach()), "gradient_norm_before_clip": float(gradient_norm)})
                basis = torch.linalg.qr(parameter.float(), mode="reduced")[0].T.detach().cpu().numpy().astype(np.float32)
                destination = output / "bases" / (fit_id + ".npz")
                np.savez(destination, rank_16=basis)
                basis_records.append({"objective": objective, "seed": seed, "file": "bases/" + destination.name,
                                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                                      "tensor_sha256": hashlib.sha256(basis.tobytes()).hexdigest(), "updates": len(order),
                                      "initializer": f"bases/pca_ts{seed}.npz", "initializer_identity": "exact_saved_this_rerun"})
                print(json.dumps({"stage": "fit_complete", "fit": fit_id, "updates": len(order)}), flush=True)
        require(all(not p.requires_grad and p.grad is None and p._version == v and p.data_ptr() == address for p, v, address in weights),
                "Frozen backbone parameters changed")
        write(output / "COMPLETE.json", {"status": "COMPLETE", "specification": specification, "bases": basis_records,
                                         "native_calls": engine.call_count, "seconds": time.monotonic() - started,
                                         "artifacts": inventory(output), "gradient_qualification_records": 18,
                                         "historical_identity_claimed": False})
    except BaseException as error:
        write(output / "FAILED.json", {"status": "FAILED", "exception": type(error).__name__, "message": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--cohort", choices=("original_1000", "mapping_300"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
