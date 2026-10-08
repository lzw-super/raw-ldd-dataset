"""Summarize downloaded Qualcomm profiles without network access or credentials."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


def percentile(values, percent):
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lower, upper = math.floor(index), math.ceil(index)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def summarize(root):
    manifest = json.loads((root / "jobs.json").read_text())
    results = {}
    for label, record in manifest["models"].items():
        rounds, all_times = [], []
        for index, job in enumerate(record.get("profile_jobs", []), 1):
            path = root / label / f"profile_{index}.json"
            if job.get("code") != "SUCCESS" or not path.exists():
                raise RuntimeError(f"{label} round {index} not successfully downloaded")
            profile = json.loads(path.read_text())
            summary = profile["execution_summary"]
            times = [t / 1000 for t in summary["all_inference_times"]]
            if not times:
                raise RuntimeError(f"{path}: no per-iteration timings")
            all_times.extend(times)
            rounds.append({
                "job": job["id"],
                "reported_ms": summary["estimated_inference_time"] / 1000,
                "sample_count": len(times),
                "mean_ms": statistics.mean(times),
                "min_ms": min(times),
                "max_ms": max(times),
                "p50_ms": percentile(times, 50),
                "p95_ms": percentile(times, 95),
                "compute_unit_counts": dict(Counter(
                    entry["compute_unit"] for entry in profile["execution_detail"]
                )),
            })
        if not rounds:
            raise RuntimeError(f"{label}: no completed rounds")
        mean = statistics.mean(all_times)
        model = root / label / "model.dlc"
        results[label] = {
            "rounds": rounds,
            "reported_mean_ms": statistics.mean(r["reported_ms"] for r in rounds),
            "sample_count": len(all_times),
            "sample_mean_ms": mean,
            "sample_p50_ms": percentile(all_times, 50),
            "sample_p95_ms": percentile(all_times, 95),
            "reciprocal_sample_mean_hz": 1000 / mean,
            "dlc_bytes": model.stat().st_size,
            "dlc_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        }
    output = dict(device=manifest["device"], shape=manifest["shape"],
                  time_unit="milliseconds", models=results)
    (root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    for label, row in sorted(results.items(), key=lambda item: item[1]["sample_mean_ms"]):
        print(f'{label:20} mean={row["sample_mean_ms"]:.4f} ms '
              f'P50={row["sample_p50_ms"]:.4f} ms P95={row["sample_p95_ms"]:.4f} ms')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("ref-doc/qaihub_w8a8_20261007"))
    summarize(parser.parse_args().output_dir)
