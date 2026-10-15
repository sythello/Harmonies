"""Mergeable self-play metrics and a dependency-free local dashboard."""
import json
import math
from pathlib import Path

import numpy as np


def moments(values):
    xs = list(values)
    if not xs:
        return {"count": 0, "mean": None, "std": None, "meanAbs": None, "rms": None, "min": None, "max": None}
    return {"count": len(xs), "mean": float(np.mean(xs)), "std": float(np.std(xs)),
            "meanAbs": float(np.mean(np.abs(xs))), "rms": float(np.sqrt(np.mean(np.square(xs)))),
            "min": min(xs), "max": max(xs)}


def merge_moments(items):
    items = [s for s in items if s["count"]]
    n = sum(s["count"] for s in items)
    if not n:
        return moments([])
    mean = sum(s["mean"] * s["count"] for s in items) / n
    square = sum((s["std"]**2 + s["mean"]**2) * s["count"] for s in items) / n
    return {"count": n, "mean": mean, "std": math.sqrt(max(0, square - mean**2)),
            "meanAbs": sum(s["meanAbs"] * s["count"] for s in items) / n,
            "rms": math.sqrt(max(0, square)), "min": min(s["min"] for s in items), "max": max(s["max"] for s in items)}


def game_metrics(game, player_id=None):
    decisions = game.get("samples", game.get("decisions", []))
    if player_id is not None:
        decisions = [s for s in decisions if s["playerId"] == player_id]
    ds = [s["diagnostics"] for s in decisions]
    metrics = {"search." + k: moments(d[k] for d in ds) for k in ("rootWidth", "visitedWidth", "meanDepth", "maxDepth")}
    metrics.update({"value." + k: moments(d[k] for d in ds) for k in ("searchValue", "networkValue")})
    for policy in ("networkPolicy", "searchPolicy"):
        for name in ("top1", "top5", "entropy"):
            metrics[f"policy.{policy}.{name}"] = moments(d[policy][name] for d in ds)
    metrics["consistency.searchMinusNetwork"] = moments(d["searchValue"] - d["networkValue"] for d in ds)
    metrics["consistency.rootMinusInherited"] = moments(d["searchValue"] - d["inheritedSearchValue"]
                                                       for d in ds if d["inheritedSearchValue"] is not None)
    result = "draw" if game["ranks"][0] == game["ranks"][1] else ("P1" if game["ranks"][0] < game["ranks"][1] else "P2")
    return {"rows": len(ds), "rounds": game["rounds"], "turns": game["turns"], "result": result,
            "firstPlayerId": game["first"], "collectionSeconds": game.get("collectionSeconds", 0), "metrics": metrics,
            "legality": {"checkedActions": len(ds), "checkedNodes": sum(d["checkedNodes"] for d in ds),
                         "checkedEdges": sum(d["checkedEdges"] for d in ds)},
            "simulations": sum(s["rootVisits"] - s["reusedVisits"] for s in decisions)}


def collection_metrics(games, errors=()):
    rounds = [g["rounds"] for g in games]
    result = {"games": len(games), "rows": sum(g["rows"] for g in games),
              "rounds": {"mean": float(np.mean(rounds)) if rounds else None,
                         **{f"p{p}": float(np.percentile(rounds, p)) if rounds else None for p in (10, 50, 90)}},
              "results": {key: sum(g["result"] == key for g in games) / len(games) if games else None for key in ("P1", "P2", "draw")},
              "metrics": {k: merge_moments(g["metrics"][k] for g in games) for k in (games[0]["metrics"] if games else [])},
              "collectionSeconds": sum(g["collectionSeconds"] for g in games),
              "simulations": sum(g["simulations"] for g in games),
              "legality": {**{k: sum(g["legality"][k] for g in games) for k in ("checkedActions", "checkedNodes", "checkedEdges")},
                           "illegalMoves": sum(e.get("count", 1) for e in errors if e.get("kind") == "illegalMove"),
                           "maskConflicts": sum(e.get("count", 1) for e in errors if e.get("kind") == "maskConflict")}}
    return result


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def repair_tail(path):
    """Only an unterminated final record can be discarded after interrupted append."""
    if not path.exists():
        path.touch()
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                stream.truncate(start)
                return
            # Complete malformed records must fail loudly rather than lose data.
            json.loads(line)


def error_records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def dashboard(directory, history, status):
    atomic_json(directory / "metrics.json", {"status": status, "iterations": history})
    payload = json.dumps({"status": status, "iterations": history}, allow_nan=False).replace("<", "\\u003c")
    page = Path(__file__).with_name("loop_dashboard.html").read_text().replace("__LOOP_DATA__", payload)
    tmp = directory / "dashboard.html.tmp"
    tmp.write_text(page)
    tmp.replace(directory / "dashboard.html")
