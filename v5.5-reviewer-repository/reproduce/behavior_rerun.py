import argparse
import hashlib
import json
from pathlib import Path

from gpu.behavior_engine import datasets
from reproduce.behavior import CONSEQUENCE_METHODS, FAMILIES, PRIMARY_METHODS, VIEWS, consequence_results, primary_results, require, valid_score


def compact(score):
    tokens = score["choice_token_ids"]
    decoded = next((c for c, token in tokens.items() if token == score["global_token_id"]), None)
    require(decoded == score["global_prediction"], "Global token and semantic answer disagree")
    require(set(score["choice_order"]) == set(score["scores"]), "Candidate order/support differs")
    require(score["prediction"] == max(score["choice_order"], key=score["scores"].get), "Candidate score winner disagrees")
    value = [score["prediction"], decoded, score["candidate_mass"]]
    valid_score(value)
    return value


def run(directory, output):
    from reproduce.integrity import verify
    verify()
    directory, output = Path(directory), Path(output)
    require(not output.exists(), "Output directory already exists")
    require(not (directory / "FAILED.json").exists(), "The GPU run has a failure marker")
    closure = json.loads((directory / "COMPLETE.json").read_text())
    require(closure["artifacts"], "GPU artifact inventory is missing")
    for name, item in closure["artifacts"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe GPU artifact path")
        file = directory / relative
        require(file.is_file() and not file.is_symlink() and file.stat().st_size == item["size_bytes"], "GPU artifact missing or changed")
        with file.open("rb") as stream:
            require(hashlib.file_digest(stream, "sha256").hexdigest() == item["sha256"], "GPU artifact digest differs")
    require({"RUN.json", "rows.jsonl", "BASES.json", "ENVIRONMENT.json"} <= set(closure["artifacts"]), "GPU inventory lacks required artifacts")
    specification = json.loads((directory / "RUN.json").read_text())
    file = directory / "rows.jsonl"
    with file.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    require(closure["status"] == "COMPLETE" and sha == closure["rows_sha256"], "Incomplete or changed GPU run")
    rows = [json.loads(line) for line in file.read_text().splitlines()]
    require(len(rows) == closure["records"] == specification["records"], "GPU case coverage differs")
    model, split = specification["model"], specification["split"]
    require(model in ("mistral", "qwen"), "Unknown model")
    original = datasets()
    if specification["study"] == "five_shifts":
        require(split == "lockbox" and len(rows) == 1500, "Main-table rerun analysis requires the complete lockbox")
        pairs = [p for family in FAMILIES for p in original["lockbox/" + family]]
        records = []
        for row, pair in zip(rows, pairs):
            require(row["pair_uid"] == pair["pair_uid"] and row["family"] == pair["family"] and
                    row["base_answer"] == pair["base"]["answer"] and row["source_answer"] == pair["source"]["answer"], "Rerun pair identity differs")
            records.append({k: row[k] for k in ("pair_uid", "family", "base_answer", "source_answer")})
        primary = {interface: [[compact(row["formats"][interface]["conditions"][name]) for name in PRIMARY_METHODS]
                               for row in rows] for interface in ("original", "answer_prefill")}
        result = primary_results({"primary_records": records, "models": {model: {"primary": primary}}})
    else:
        require(specification["study"] == "consequences" and split in ("discovery", "checking") and len(rows) == 120, "Unknown consequence population")
        cores = original[split]
        views = VIEWS[:-1] if split == "discovery" else VIEWS
        for row, core in zip(rows, cores):
            require(row["uid"] == core["uid"] and row["group"] == core and row["views"] == list(views), "Consequence core or view changed")
        result = {}
        for interface in ("original", "answer_prefill"):
            scores = [[[compact(row["formats"][interface][view]["conditions"][name]) for name in CONSEQUENCE_METHODS]
                       for view in views] for row in rows]
            result[interface] = consequence_results({"cores": {split: cores}, "models": {model: {
                "consequences": {split: {"views": list(views), "scores": scores}}}}})
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    (output / "COMPLETE.json").write_text(json.dumps({"status": "COMPLETE", "scope": "New GPU predictions to saved-fit statistics",
                                                     "rows_sha256": sha, "model": model, "study": specification["study"],
                                                     "split": split, "published_value_agreement_required": False}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run, args.output)


if __name__ == "__main__":
    main()
