import argparse
import csv
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def build(mechanism_dir, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mechanism_dir = Path(mechanism_dir)
    native = {model: read(mechanism_dir / f"native_{model}.json") for model in ("mistral", "qwen")}
    fig, ax = plt.subplots(figsize=(7, 3.6), layout="constrained")
    labels, locations = [], []
    for row, (model, frame) in enumerate((m, f) for m in ("mistral", "qwen") for f in ("P", "M")):
        value = native[model]["mappings"]["m3"]["primary"][f"full/{frame}/other_minus_null"]
        y = 3 - row
        mean = 100 * value["mean"]
        low, high = [100 * x for x in value["ci95"]]
        ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], color="black", marker="s", capsize=4, zorder=5)
        for index, fit in enumerate(value["per_fit"]):
            ax.plot(100 * fit["mean"], y - (index+1)*0.09, "o", color=("#0072B2", "#D55E00", "#009E73")[index],
                    markersize=4, label=f"Fit {fit['seed']}" if row == 0 else None)
        labels.append(("Mistral24B" if model == "mistral" else "Qwen72B") + (" — addition" if frame == "P" else " — removal"))
        locations.append(y)
    ax.set_yticks(locations, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Gain over matched null in actual-endpoint fidelity (percentage points)")
    ax.grid(axis="x", color="0.9")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Native key-only confirmation: 120 cores, three fixed fits")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"native_key_effects.{extension}", dpi=180)
    plt.close(fig)
    field_names = ("study", "model", "mapping", "exchange", "gain_over_null_pp", "ci_low_pp", "ci_high_pp",
                   "endpoint_fidelity_percent", "symbolic_correct_percent", "nonobserver_preserved_percent", "all_five_correct_percent")
    rows = []
    for family in ("native", "mapping"):
        for model in ("mistral", "qwen"):
            report = read(mechanism_dir / f"{family}_{model}.json")
            for mapping, results in report["mappings"].items():
                population = results["primary_population"]
                for frame in ("P", "M"):
                    effect = results["primary"][f"full/{frame}/other_minus_null"]
                    condition = results["conditions"][f"full/{frame}/other"][population]
                    metrics = ("affected3/destination_id", "affected3/aligned_oracle", "nonobserver_pair/base_oracle_joint",
                               "full_pattern/aligned_oracle_and_preservation")
                    values = [100 * condition[name]["mean"] for name in metrics]
                    rows.append([family, model, mapping, "addition" if frame == "P" else "removal",
                                 100*effect["mean"], *[100*x for x in effect["ci95"]], *values])
    with (output / "main_mechanism_tables.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(field_names)
        writer.writerows(rows)
    with (output / "main_mechanism_tables.md").open("w") as stream:
        stream.write("| Study | Model | Mapping | Exchange | Gain over null [95% CI], pp | Fidelity % | Correct % | Preserved % | All five correct % |\n")
        stream.write("|---|---|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            stream.write("| " + " | ".join(str(x) for x in row[:4]) + f" | {row[4]:.1f} [{row[5]:.1f}, {row[6]:.1f}] | " +
                         " | ".join(f"{x:.1f}" for x in row[7:]) + " |\n")
    return {"status": "COMPLETE", "source": "Independently recomputed mechanism statistics", "table_rows": len(rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.mechanism_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
