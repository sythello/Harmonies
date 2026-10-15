"""Sequential self-play -> recent replay window -> exactly N training rows."""
import argparse
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from loop_metrics import atomic_json, collection_metrics, dashboard, game_metrics, repair_tail, error_records
from gatekeeper import gatekeeper, file_hash
from replay import ReplayBuffer, sampling_passes, window_size
from train import Model, VALUE_HEADS, load, loss_components, save

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(games_per_iteration=1000, simulations=32, train_rows=400_000,
                min_rows=100_000, taper_scale=50_000, expand_window_per_row=0.4,
                taper_window_exponent=0.65, batch_size=64, lr=0.001, seed=1729,
                threads=1, device="cpu", hidden=[128, 128], embedding_dim=8,
                initial=None, wait_for_min_rows=False, gate_games=100, gate_simulations=128)


class TrainingMaskError(ValueError):
    def __init__(self, count):
        super().__init__(f"Training mask conflicts: {count}")
        self.count = count


def checkpoint(model, optimizer, prefix, steps):
    """Publish weights before the iteration's JSON commit marker."""
    temporary = str(prefix) + ".tmp"
    save(model, optimizer, temporary, steps)
    for suffix in (".pt", ".json"):
        Path(temporary + suffix).replace(str(prefix) + suffix)


def optimizer_for(model, config, old=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    if old is not None:
        optimizer.load_state_dict(old["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = config["lr"]
    return optimizer


def batches(start, stop, rows, size, seed):
    # Keep pass order; carry the final short slice into the next minibatch.
    # This gives exactly ceil(rows / size) steps, even across reshuffle boundaries.
    pending = []
    for selection in sampling_passes(start, stop, rows, seed):
        for row in selection:
            pending.append(row)
            if len(pending) == size:
                yield pending
                pending = []
    if pending:
        yield pending


def train_iteration(buffer, source, destination, config, start, stop, iteration):
    torch.manual_seed(config["seed"] + iteration)
    model, old = load(str(source) + ".pt", config["device"])
    model.train()
    optimizer = optimizer_for(model, config, old)
    rows = steps = 0
    totals = {h: 0.0 for h in ("policy", *VALUE_HEADS)}
    labels = {h: np.zeros(2) for h in VALUE_HEADS}
    predictions = {h: np.zeros(2) for h in VALUE_HEADS}
    norms = {name: {"sum": 0.0, "max": 0.0, "last": 0.0} for name, _ in model.named_parameters()}
    global_norms = {key: {"sum": 0.0, "max": 0.0, "last": 0.0} for key in ("beforeClip", "afterClip")}
    started = time.perf_counter()
    path = destination.parent
    sample_hash = hashlib.sha256()
    with (path / "training.jsonl").open("w") as log, (path / "sampled-rows.i64").open("wb") as selections:
        for ids in batches(start, stop, config["train_rows"], config["batch_size"], config["seed"] + iteration):
            raw_ids = np.asarray(ids, dtype="<i8").tobytes()
            selections.write(raw_ids)
            sample_hash.update(raw_ids)
            numeric, cards, target, mask, values = buffer.batch(ids, config["device"])
            conflicts = (~mask.any(1) | (target.masked_fill(mask, 0).abs().sum(1) > 1e-6) |
                         ((target.sum(1) - 1).abs() > 1e-5) | ~torch.isfinite(target).all(1) | (target < 0).any(1))
            if conflicts.any():
                raise TrainingMaskError(int(conflicts.sum().item()))
            optimizer.zero_grad()
            out = model(numeric, cards)
            components = loss_components(out, target, mask, values)
            loss = sum(components.values())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            gradnorm = {name: float(p.grad.norm().item()) if p.grad is not None else 0.0
                        for name, p in model.named_parameters()}
            before = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True).item())
            after = math.sqrt(sum(float(p.grad.square().sum().item()) for p in model.parameters() if p.grad is not None))
            optimizer.step()
            count = len(ids)
            rows += count
            steps += 1
            losses = {h: float(v.item()) for h, v in components.items()}
            label = {h: values[h].mean(0).tolist() for h in VALUE_HEADS}
            prediction = {h: out[h].detach().mean(0).tolist() for h in VALUE_HEADS}
            for h in totals:
                totals[h] += losses[h] * count
            for h in VALUE_HEADS:
                labels[h] += np.asarray(label[h]) * count
                predictions[h] += np.asarray(prediction[h]) * count
            for table, values_by_key in ((norms, gradnorm), (global_norms, {"beforeClip": before, "afterClip": after})):
                for name, v in values_by_key.items():
                    table[name]["sum"] += v
                    table[name]["max"] = max(table[name]["max"], v)
                    table[name]["last"] = v
            log.write(json.dumps({"step": steps, "rows": rows, "batchRows": count, "loss": losses,
                                  "labelAvg": label, "predictAvg": prediction, "gradnorm": gradnorm,
                                  "globalGradnorm": {"beforeClip": before, "afterClip": after}, "maskConflicts": 0}, allow_nan=False) + "\n")
            if steps % 100 == 0:
                log.flush()
                print(json.dumps({"event": "training_progress", "iteration": iteration, "steps": steps, "rows": rows,
                                  "loss": losses}), flush=True)
    if rows != config["train_rows"] or steps != math.ceil(rows / config["batch_size"]):
        raise AssertionError("Training row budget mismatch")
    checkpoint(model, optimizer, destination, old["steps"] + steps)
    elapsed = time.perf_counter() - started
    finish_norms = lambda table: {k: {"mean": v["sum"] / steps, "max": v["max"], "last": v["last"]} for k, v in table.items()}
    return {"rows": rows, "steps": steps, "totalSteps": old["steps"] + steps, "seconds": elapsed,
            "rowsPerSecond": rows / elapsed, "loss": {h: n / rows for h, n in totals.items()},
            "labelAvg": {h: (v / rows).tolist() for h, v in labels.items()},
            "predictAvg": {h: (v / rows).tolist() for h, v in predictions.items()},
            "gradnorm": finish_norms(norms), "globalGradnorm": finish_norms(global_norms), "maskConflicts": 0,
            "sampling": {"seed": config["seed"] + iteration, "passes": math.ceil(rows / (stop - start)),
                         "fullPasses": rows // (stop - start), "lastPartialRows": rows % (stop - start),
                         "uniqueRows": min(rows, stop-start), "sha256": sample_hash.hexdigest()}}


def train_or_resume(buffer, source, destination, config, start, stop, iteration):
    marker = destination.parent / "training-result.json"
    identity = {"source": str(source), "sourceHash": file_hash(str(source) + ".pt"),
                "window": [start, stop], "config": config}
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["identity"] != identity:
            raise ValueError("Completed training does not match the source/window/config")
        if all(Path(str(destination) + suffix).exists() and
               file_hash(str(destination) + suffix) == saved["hashes"][suffix] for suffix in (".pt", ".json")):
            return saved["metrics"]
    result = train_iteration(buffer, source, destination, config, start, stop, iteration)
    atomic_json(marker, {"identity": identity, "metrics": result,
                         "hashes": {suffix: file_hash(str(destination) + suffix) for suffix in (".pt", ".json")}})
    return result


def collect_iteration(buffer, directory, source, config, iteration, status):
    trajectory = directory / "games.jsonl"
    repair_tail(trajectory)
    model_id = hashlib.sha256(Path(str(source) + ".json").read_bytes()).hexdigest()
    count = 0
    base = (iteration - 1) * config["games_per_iteration"]

    def ingest(line):
        nonlocal count
        game = json.loads(line)
        index = base + count
        if (game["seed"] != config["seed"] + index or game["first"] != index % 2 or
                game["model"] != model_id or game["search"]["simulations"] != config["simulations"]):
            raise ValueError("Collected game does not match iteration/model/seed/budget")
        buffer.add_game(game, iteration, count, game_metrics(game))
        count += 1

    # Existing records are validated and ingestion is idempotent after a crash.
    with trajectory.open() as games:
        for line in games:
            ingest(line)
    if count > config["games_per_iteration"]:
        raise ValueError("Too many games in iteration")
    if count == config["games_per_iteration"]:
        return
    command = ["node", "--import", "tsx", "scripts/rl.ts", "collect", "--model", str(source) + ".json",
               "--games", str(config["games_per_iteration"] - count), "--simulations", str(config["simulations"]),
               "--seed", str(config["seed"]), "--game-offset", str(base + count), "--output", str(trajectory),
               "--error-output", str(directory / "errors.jsonl")]
    with trajectory.open() as games, (directory / "collection.log").open("a") as log:
        games.seek(0, 2)
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "game" in event:
                    ingest(games.readline())
                    status.update(collectedGames=count, totals=buffer.totals())
                    atomic_json(directory.parent / "state.json", status)
                    print(json.dumps({"event": "selfplay_progress", "iteration": iteration, "games": count,
                                      "rows": buffer.totals()["rows"]}), flush=True)
            if process.wait() != 0:
                raise RuntimeError(f"Self-play failed; see {directory / 'collection.log'}")
            # A finished append may survive interruption before its progress line.
            for line in games:
                ingest(line)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            process.stdout.close()
    if count != config["games_per_iteration"]:
        raise ValueError("Self-play game budget mismatch")


def run(directory, config, iterations):
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    initial = directory / "initial"
    if not Path(str(initial) + ".pt").exists() or not Path(str(initial) + ".json").exists():
        model, old = load(config["initial"], config["device"]) if config["initial"] else (
            Model(config["hidden"], config["embedding_dim"]).to(config["device"]), None)
        checkpoint(model, optimizer_for(model, config, old), initial, old["steps"] if old else 0)
    history = []
    while (directory / f"iteration-{len(history)+1:04d}" / "metrics.json").exists():
        history.append(json.loads((directory / f"iteration-{len(history)+1:04d}" / "metrics.json").read_text()))
    # Older runs have no gate history; their latest model is the initial incumbent.
    latest = directory / history[-1].get("latestModel", history[-1]["model"]) if history else initial
    best = directory / history[-1].get("bestModel", history[-1]["model"]) if history else initial
    status = {"phase": "complete", "completedIterations": len(history), "model": str(latest.relative_to(directory)),
              "latestModel": str(latest.relative_to(directory)), "bestModel": str(best.relative_to(directory))}
    buffer = ReplayBuffer(directory / "replay.sqlite3")
    status["totals"] = buffer.totals()
    current = None
    report = None
    try:
        dashboard(directory, history, status)
        for iteration in range(len(history) + 1, iterations + 1):
            started = time.perf_counter()
            current = directory / f"iteration-{iteration:04d}"
            current.mkdir(exist_ok=True)
            report = None
            status.pop("gatekeeper", None)
            status.update(phase="collecting", iteration=iteration, collectedGames=0, totals=buffer.totals())
            atomic_json(directory / "state.json", status)
            collect_iteration(buffer, current, best, config, iteration, status)
            totals = buffer.totals()
            n = window_size(totals["rows"], **{k: config[k] for k in ("min_rows", "taper_scale", "expand_window_per_row", "taper_window_exponent")})
            start, stop = totals["rows"] - n + 1, totals["rows"] + 1
            report = {"iteration": iteration, "totals": totals, "sourceModel": str(latest.relative_to(directory)),
                      "selfplayModel": str(best.relative_to(directory)), "gatekeeper": None,
                      "selfplay": collection_metrics(buffer.game_metrics(iteration), error_records(current / "errors.jsonl")),
                      "window": {"rows": n, "firstRowId": start, "lastRowId": stop-1}, "training": None}
            if not config["wait_for_min_rows"] or totals["rows"] >= config["min_rows"]:
                status["phase"] = "training"
                atomic_json(directory / "state.json", status)
                dashboard(directory, [*history, report], status)
                destination = current / "model"
                report["training"] = train_or_resume(buffer, latest, destination, config, start, stop, iteration)
                latest = destination
                status.update(phase="gatekeeper", latestModel=str(latest.relative_to(directory)))

                def gate_progress(result):
                    report["gatekeeper"] = result
                    status["gatekeeper"] = {k: result[k] for k in ("games", "requiredGames", "winRate", "complete", "promoted")}
                    atomic_json(current / "gatekeeper.json", result)
                    atomic_json(directory / "state.json", status)
                    dashboard(directory, [*history, report], status)
                    print(json.dumps({"event": "gatekeeper_progress", "iteration": iteration, **status["gatekeeper"]}), flush=True)

                report["gatekeeper"] = gatekeeper(current, latest, best, config, iteration, gate_progress)
                if report["gatekeeper"]["promoted"]:
                    best = latest
            else:
                report["warmup"] = True
            report.update(model=str(latest.relative_to(directory)), latestModel=str(latest.relative_to(directory)),
                          bestModel=str(best.relative_to(directory)), seconds=time.perf_counter()-started)
            # The commit marker comes last; interrupted training is replayed from source.
            atomic_json(current / "metrics.json", report)
            history.append(report)
            status.update(phase="complete", completedIterations=iteration, model=report["model"],
                          latestModel=report["latestModel"], bestModel=report["bestModel"], totals=totals)
            atomic_json(directory / "state.json", status)
            dashboard(directory, history, status)
            print(json.dumps({"event": "iteration_complete", **report}, allow_nan=False), flush=True)
        atomic_json(directory / "state.json", status)
    except BaseException as error:
        status.update(phase="failed", error=str(error), totals=buffer.totals())
        if current:
            if isinstance(error, TrainingMaskError):
                with (current / "errors.jsonl").open("a") as log:
                    log.write(json.dumps({"kind": "maskConflict", "count": error.count, "source": "training", "message": str(error)}) + "\n")
            status["selfplay"] = collection_metrics(buffer.game_metrics(status["iteration"]), error_records(current / "errors.jsonl"))
        atomic_json(directory / "state.json", status)
        dashboard(directory, [*history, report] if report is not None and report not in history else history, status)
        raise
    finally:
        buffer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Run directory; reused directories resume automatically")
    parser.add_argument("--iterations", required=True, type=int, help="Target TOTAL self-play iterations, not additional iterations")
    for key, value in DEFAULTS.items():
        option = "--" + key.replace("_", "-")
        if key == "hidden":
            parser.add_argument(option, nargs="+", type=int)
        elif key == "wait_for_min_rows":
            parser.add_argument(option, action=argparse.BooleanOptionalAction, default=None)
        else:
            parser.add_argument(option, type=type(value) if value is not None else str)
    args = parser.parse_args()
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # POSIX lock is released on process exit, including crashes; no stale lockfile problem.
    with (directory / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another loop process owns this run directory")
        config_path = directory / "config.json"
        saved = json.loads(config_path.read_text()) if config_path.exists() else None
        baseline = {**DEFAULTS, **(saved or {})}
        config = {k: getattr(args, k) if getattr(args, k) is not None else baseline[k] for k in DEFAULTS}
        if config["initial"]:
            config["initial"] = str(Path(config["initial"]).resolve())
        if saved and any(config[k] != saved[k] for k in saved):
            parser.error("Resume requires the original configuration; use a new output directory to change it")
        ints = ("games_per_iteration", "simulations", "train_rows", "min_rows", "taper_scale", "batch_size", "threads", "embedding_dim", "gate_games", "gate_simulations")
        if (args.iterations < 1 or any(config[k] < 1 for k in ints) or min(config["hidden"]) < 1 or config["seed"] < 0 or
                config["simulations"] > 100000 or config["gate_simulations"] > 100000 or config["gate_games"] % 2 != 0 or not 0 < config["expand_window_per_row"] <= 1 or
                not 0 < config["taper_window_exponent"] <= 1 or not math.isfinite(config["lr"]) or config["lr"] <= 0):
            parser.error("Invalid loop configuration")
        if config != saved:
            atomic_json(config_path, config)
        run(directory, config, args.iterations)


if __name__ == "__main__":
    main()
