"""Run a small serial collection/training benchmark using the production CLIs."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def run(directory, name, command):
    started = time.perf_counter()
    print(f"Starting {name}", flush=True)
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    (directory / f"{name}.stdout.jsonl").write_text(result.stdout)
    (directory / f"{name}.stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"{name} failed:\n{result.stderr}\n{result.stdout}")
    records = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    print(json.dumps({"stage": name, "wallSeconds": elapsed}), flush=True)
    return {"command": command, "wallSeconds": elapsed, "records": records}


def collection_summary(result, path, simulations):
    rows = result["records"]
    samples = sum(row["samples"] for row in rows)
    seconds = sum(row["collectSeconds"] for row in rows)
    return {"games": len(rows), "samples": samples, "simulationsPerAction": simulations,
            "collectSeconds": seconds, "wallSeconds": result["wallSeconds"],
            "secondsPerGame": seconds / len(rows), "samplesPerSecond": samples / seconds,
            "msPerAction": seconds * 1000 / samples,
            "simulationsPerSecond": samples * simulations / seconds,
            "gameSecondsMin": min(row["collectSeconds"] for row in rows),
            "gameSecondsMax": max(row["collectSeconds"] for row in rows),
            "fileBytes": path.stat().st_size}


def main(args):
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)  # Never append to a previous benchmark.
    report = {"startedAt": datetime.now(timezone.utc).isoformat(), "settings": vars(args), "stages": {}}
    report["gitCommit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report["gitStatus"] = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    paths = [ROOT / "src/game/engine.ts", ROOT / "scripts/rl.ts", ROOT / "rl/train.py",
             *sorted((ROOT / "src/game/ai/rl").glob("*.ts"))]
    report["sourceSha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hardware = "import os from 'node:os'; console.log(JSON.stringify({node:process.version,platform:os.platform(),arch:os.arch(),cpu:os.cpus()[0]?.model,logicalCpus:os.cpus().length,memoryGiB:os.totalmem()/2**30}));"
    report["hardware"] = json.loads(subprocess.check_output(["node", "--input-type=module", "-e", hardware], text=True))
    report["python"] = sys.version
    def record(name, command):
        result = run(directory, name, command)
        report["stages"][name] = result
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return result
    initial = str(directory / "initial")
    trained = str(directory / "trained")
    record("init", [sys.executable, "rl/train.py", "init", "--seed", str(args.seed), "--out", initial])
    data, summaries = [], []
    for simulations in args.simulations:
        path = directory / f"collect-{simulations}.jsonl"
        result = record(f"collect-{simulations}", ["node", "--import", "tsx", "scripts/rl.ts", "collect",
                        "--model", initial + ".json", "--games", str(args.games), "--simulations", str(simulations),
                        "--seed", str(args.seed), "--output", str(path)])
        summaries.append(collection_summary(result, path, simulations))
        data.append(str(path))
        print(json.dumps(summaries[-1]), flush=True)
    training = record("train", [sys.executable, "rl/train.py", "train", "--resume", initial + ".pt",
                      "--data", *data, "--epochs", str(args.epochs), "--batch-size", "64",
                      "--seed", str(args.seed), "--threads", str(args.threads), "--device", args.device,
                      "--out", trained])
    epochs = [r for r in training["records"] if "epoch" in r]
    report["collection"] = summaries
    report["training"] = {**training["records"][-1], "wallSeconds": training["wallSeconds"],
                          "epochs": args.epochs, "trainSamples": epochs[0]["trainSamples"],
                          "validationSamples": epochs[0]["validationSamples"],
                          "medianEpochTrainSeconds": statistics.median(r["trainSeconds"] for r in epochs),
                          "firstEpochLoss": epochs[0]["train"], "lastEpochLoss": epochs[-1]["train"],
                          "firstValidationLoss": epochs[0].get("validation"),
                          "lastValidationLoss": epochs[-1].get("validation")}
    repetitions = [training]
    for repeat in range(2, args.train_repeats + 1):
        command = list(training["command"])
        command[-1] = str(directory / f"trained-repeat-{repeat}")
        repetitions.append(record(f"train-repeat-{repeat}", command))
    report["training"]["repetitions"] = len(repetitions)
    report["training"]["trainSecondsMedian"] = statistics.median(r["records"][-1]["trainSeconds"] for r in repetitions)
    report["training"]["wallSecondsMedian"] = statistics.median(r["wallSeconds"] for r in repetitions)
    print(json.dumps(report["training"]), flush=True)
    # Resume optimizer and execute the exported trained model on new full games.
    record("resume", [sys.executable, "rl/train.py", "train", "--resume", trained + ".pt", "--data", *data,
                      "--epochs", "1", "--seed", str(args.seed), "--threads", str(args.threads),
                      "--device", args.device, "--out", str(directory / "resumed")])
    path = directory / "trained-collect.jsonl"
    result = record("trained-collect", ["node", "--import", "tsx", "scripts/rl.ts", "collect",
                    "--model", trained + ".json", "--games", "2", "--simulations", str(max(args.simulations)),
                    "--seed", str(args.seed + 1000), "--output", str(path)])
    report["trainedCollection"] = collection_summary(result, path, max(args.simulations))
    report["finishedAt"] = datetime.now(timezone.utc).isoformat()
    (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Report: {directory / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory under rl/runs")
    parser.add_argument("--games", type=int, default=8, help="Games per simulation budget")
    parser.add_argument("--simulations", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.games, args.epochs, args.threads, args.train_repeats, *args.simulations) < 1 or len(set(args.simulations)) != len(args.simulations):
        parser.error("Use positive sizes and distinct simulation budgets")
    main(args)
