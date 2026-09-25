import argparse
import hashlib
import json
import time
from pathlib import Path

from gpu import consequence_task as task
from gpu.behavior_engine import BehaviorEngine, FAMILIES, datasets, encode_pair, mapped, require, same
from gpu.train import append, inventory, write


def bases_for(engine, model, directory=None):
    from gpu.runtime.load import load_basis, load_training_bases
    bases, descriptions = {}, {}
    if directory is not None:
        indexed = load_training_bases(directory, model, "original_1000")
    for seed in (101, 102, 103):
        for objective in ("f_star", "m3", "pca"):
            if directory is None:
                array, description = load_basis(model, "original_1000", objective, seed)
            else:
                array, description = indexed[objective, seed]
            require(array.shape == (16, engine.width) and engine.np.isfinite(array).all(), "Invalid fitted basis")
            require(engine.np.max(engine.np.abs(array.astype(float) @ array.astype(float).T - engine.np.eye(16))) < 1e-5,
                    "Fitted basis is not orthonormal")
            name = f"{objective}_ts{seed}"
            bases[name] = engine.torch.tensor(array, device=engine.device)
            descriptions[name] = description
    return bases, descriptions


def five_shift_pair(engine, pair, bases, index):
    base, source = encode_pair(engine.tok, pair, "original", engine.model_name, engine.padding)
    _, hb, prefix = engine.forward(base)
    _, hs, _ = engine.forward(source)
    patches = {name: engine.fixed(hb, hs, basis) for name, basis in bases.items()}
    patches["full_source"] = engine.fixed(hb, hs)
    hashes = {name: engine.tensor_hash(value) for name, value in patches.items()}
    record = {"pair_uid": pair.get("pair_uid", str(index)), "family": pair["family"],
              "base_answer": pair["base"]["answer"], "source_answer": pair["source"]["answer"],
              "targets": {o: mapped(pair["source"]["answer"], o) for o in ("f_star", "m3")},
              "patch_sha256": hashes, "formats": {}}
    for interface in ("original", "answer_prefill"):
        enc_base, enc_source = encode_pair(engine.tok, pair, interface, engine.model_name, engine.padding)
        clean = engine.forward(enc_base, expected=hb, expected_prefix=prefix)[0]
        source_result = engine.forward(enc_source, expected=hs)[0]
        clone = engine.forward(enc_base, expected=hb, expected_prefix=prefix, clone=True)[0]
        conditions = {"clean": clean, "source": source_result}
        null_deltas = [same(clean, clone)]
        for name, basis in bases.items():
            self_result = engine.forward(enc_base, fixed=engine.fixed(hb, hb, basis), expected=hb, expected_prefix=prefix)[0]
            null_deltas.append(same(clean, self_result))
        for name, patch in patches.items():
            conditions[name] = engine.forward(enc_base, fixed=patch, expected=hb, expected_prefix=prefix)[0]
            require(engine.tensor_hash(patch) == hashes[name], "Materialized patch changed across interfaces")
        record["formats"][interface] = {"conditions": conditions, "maximum_identity_delta": max(null_deltas)}
    return record


def consequence_group(engine, group, views, bases):
    captures = {}
    for label, location in (("clean", group["base"]), ("source", group["source"]), ("natural_d", task.PERM[group["source"]])):
        _, hidden, prefix = engine.forward(engine.encode(task.record(group, "direct", location), "original"))
        captures[label] = (hidden, prefix)
    hb, prefix = captures["clean"]
    hs = captures["source"][0]
    patches = {name: engine.fixed(hb, hs, basis) for name, basis in bases.items()}
    patches["full_source"] = engine.fixed(hb, hs)
    patches["full_d"] = engine.fixed(hb, captures["natural_d"][0])
    hashes = {name: engine.tensor_hash(value) for name, value in patches.items()}
    result = {"uid": group["uid"], "views": list(views), "group": group,
              "patch_sha256": hashes, "formats": {}, "forecasts": {
                  objective: {account: {view: task.predict(group, view, objective, account) for view in views}
                              for account in task.ACCOUNTS} for objective in ("f_star", "m3")}}
    for interface in ("original", "answer_prefill"):
        result["formats"][interface] = {}
        for view in views:
            conditions = {}
            for label, location in (("clean", group["base"]), ("source", group["source"]), ("natural_d", task.PERM[group["source"]])):
                event, reference_prefix = captures[label]
                conditions[label] = engine.forward(engine.encode(task.record(group, view, location), interface),
                                                   expected=event, expected_prefix=reference_prefix)[0]
            encoded = engine.encode(task.record(group, view), interface)
            clone = engine.forward(encoded, expected=hb, expected_prefix=prefix, clone=True)[0]
            null_deltas = [same(conditions["clean"], clone)]
            if view == "direct":
                for basis in bases.values():
                    self_result = engine.forward(encoded, fixed=engine.fixed(hb, hb, basis), expected=hb, expected_prefix=prefix)[0]
                    null_deltas.append(same(conditions["clean"], self_result))
            for name, patch in patches.items():
                conditions[name] = engine.forward(encoded, fixed=patch, expected=hb, expected_prefix=prefix)[0]
                require(engine.tensor_hash(patch) == hashes[name], "Patch changed across consequence views")
            result["formats"][interface][view] = {"conditions": conditions, "maximum_identity_delta": max(null_deltas)}
    return result


def run(args):
    from reproduce.integrity import verify
    from gpu.runtime.load import load_engine
    verify()
    output = Path(args.output)
    require(not output.exists(), "Use a new output directory; saved outcomes are never overwritten")
    output.mkdir(parents=True)
    population = datasets()
    split = args.split or ("lockbox" if args.study == "five_shifts" else "checking")
    require(split in (("calibration", "lockbox") if args.study == "five_shifts" else ("discovery", "checking")), "Wrong panel for this study")
    rows = [p for family in FAMILIES for p in population[split + "/" + family]] if args.study == "five_shifts" else population[split]
    padding = 1024 if args.model == "mistral" else (None if args.study == "five_shifts" else 256)
    views = task.DISCOVERY_VIEWS if split == "discovery" else task.VIEWS
    write(output / "RUN.json", {"model": args.model, "study": args.study, "split": split, "records": len(rows),
                               "padding": padding, "interfaces": ["original", "answer_prefill"],
                               "scientific_success_gate": None, "implementation": "derived_reviewer_implementation"})
    started = time.monotonic()
    try:
        native = load_engine(args.model, model_path=args.model_path, native_call_limit=100000,
                             journal_path=output / "native_loader.jsonl", deadline_seconds=43200)
        engine = BehaviorEngine(native, args.model, padding)
        engine.journal = lambda row: append(output / "calls.jsonl", row)
        bases, descriptions = bases_for(engine, args.model, args.basis_dir)
        if args.study == "consequences":
            seed = 20260908 if args.model == "mistral" else 2026090711
            matrix = engine.np.random.default_rng(seed).normal(size=(engine.width, 16))
            random_basis = engine.np.linalg.qr(matrix, mode="reduced")[0].T.astype(engine.np.float32)
            bases["random"] = engine.torch.tensor(random_basis, device=engine.device)
            descriptions["random"] = {"seed": seed, "role": "unfitted_random_control"}
        write(output / "BASES.json", descriptions)
        write(output / "ENVIRONMENT.json", engine.environment)
        for index, record in enumerate(rows):
            value = five_shift_pair(engine, record, bases, index) if args.study == "five_shifts" else consequence_group(engine, record, views, bases)
            append(output / "rows.jsonl", value)
            print(json.dumps({"study": args.study, "records": index + 1, "total": len(rows)}), flush=True)
        file = output / "rows.jsonl"
        with file.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        write(output / "COMPLETE.json", {"status": "COMPLETE", "records": len(rows), "rows_sha256": sha,
                                         "artifacts": inventory(output),
                                         "native_calls": engine.call_count, "seconds": time.monotonic() - started})
    except BaseException as error:
        write(output / "FAILED.json", {"status": "FAILED", "exception": type(error).__name__, "message": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mistral", "qwen"), required=True)
    parser.add_argument("--study", choices=("five_shifts", "consequences"), required=True)
    parser.add_argument("--split", choices=("lockbox", "calibration", "discovery", "checking"))
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--basis-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
