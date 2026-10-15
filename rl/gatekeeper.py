"""Paired serving matches against the incumbent. No evaluation rows enter replay."""
import hashlib
import json
from pathlib import Path
import subprocess

from loop_metrics import collection_metrics, error_records, game_metrics, repair_tail

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_gate(games, required_games, errors=(), metrics=None):
    count = len(games)
    if count > required_games:
        raise ValueError("Too many gatekeeper games")
    results = [g["result"] for g in games]
    if any(result not in (0, 0.5, 1) for result in results):
        raise ValueError("Invalid gatekeeper result")
    wins, draws, losses = (results.count(result) for result in (1, 0.5, 0))
    rate = (wins + 0.5 * draws) / count if count else None
    if metrics is None:
        metrics = [[game_metrics(g), game_metrics(g, 0), game_metrics(g, 1)] for g in games]
    return {"games": count, "requiredGames": required_games, "wins": wins, "draws": draws, "losses": losses,
            "winRate": rate, "threshold": 0.5, "complete": count == required_games,
            "promoted": rate >= 0.5 if count == required_games else None,
            "meanCandidateScore": sum(g["candidateScore"] for g in games) / count if count else None,
            "meanIncumbentScore": sum(g["incumbentScore"] for g in games) / count if count else None,
            "meanMargin": sum(g["candidateScore"] - g["incumbentScore"] for g in games) / count if count else None,
            "telemetry": collection_metrics([m[0] for m in metrics], errors),
            "byModel": {name: collection_metrics([m[player+1] for m in metrics])
                        for player, name in enumerate(("candidate", "incumbent"))}}


def gatekeeper(directory, candidate, incumbent, config, iteration, on_progress):
    path = directory / "gate-games.jsonl"
    repair_tail(path)
    models = [file_hash(str(prefix) + ".json") for prefix in (candidate, incumbent)]
    # A separate evaluation seed stream, paired by environment and alternating seats.
    seed = config["seed"] + 1_000_000_000 + (iteration - 1) * (config["gate_games"] // 2)
    games = []
    metrics = []

    def ingest(line):
        game = json.loads(line)
        index = len(games)
        c = game["search"]
        if (game.get("purpose") != "gatekeeper" or game["phase"] != "ended" or not game["decisions"] or
                game["models"] != models or game["seed"] != seed + index // 2 or game["first"] != index % 2 or
                c["noiseWeight"] != 0 or c["temperature"] != 0 or c["simulations"] != config["gate_simulations"]):
            raise ValueError("Gatekeeper game does not match candidate/incumbent/serving settings")
        seat = game["first"]  # candidate ID 0 is seat 0 when first=0, otherwise seat 1
        a, b = game["ranks"][seat], game["ranks"][1-seat]
        expected = 0.5 if a == b else int(a < b)
        if (game["result"] != expected or game["candidateScore"] != game["scores"][seat] or
                game["incumbentScore"] != game["scores"][1-seat] or len(game["actions"]) != len(game["decisions"]) or
                any(d["rootVisits"] - d["reusedVisits"] != config["gate_simulations"] for d in game["decisions"])):
            raise ValueError("Gatekeeper result/diagnostics mismatch")
        games.append(game)
        metrics.append([game_metrics(game), game_metrics(game, 0), game_metrics(game, 1)])

    def report():
        return {**summarize_gate(games, config["gate_games"], error_records(directory / "gate-errors.jsonl"), metrics),
                "candidateModel": str(candidate.relative_to(directory.parent)),
                "incumbentModel": str(incumbent.relative_to(directory.parent)),
                "modelHashes": models, "seed": seed, "simulations": config["gate_simulations"],
                "noiseWeight": 0, "temperature": 0, "treeReuse": "within-turn"}

    with path.open() as stream:
        for line in stream:
            ingest(line)
    on_progress(report())
    remaining = config["gate_games"] - len(games)
    if remaining:
        command = ["node", "--import", "tsx", "scripts/rl.ts", "gate", "--model", str(candidate) + ".json",
                   "--opponent", str(incumbent) + ".json", "--games", str(remaining),
                   "--game-offset", str(len(games)), "--simulations", str(config["gate_simulations"]),
                   "--seed", str(seed), "--output", str(path), "--error-output", str(directory / "gate-errors.jsonl")]
        with path.open() as stream, (directory / "gate.log").open("a") as log:
            stream.seek(0, 2)
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") == "gate_game":
                        ingest(stream.readline())
                        on_progress(report())
                if process.wait() != 0:
                    raise RuntimeError(f"Gatekeeper failed; see {directory / 'gate.log'}")
                for line in stream:
                    ingest(line)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                process.stdout.close()
                # Persist available metrics, including fatal legality errors.
                on_progress(report())
    result = report()
    if not result["complete"]:
        raise ValueError("Gatekeeper game budget mismatch")
    return result
