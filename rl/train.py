"""Small FFN trainer. The TypeScript engine owns features, legality, and trajectories."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

SCHEMA = json.loads((Path(__file__).parent / "schema.json").read_text())
VALUE_HEADS = ("rank", "relative", "absolute")


class Model(nn.Module):
    def __init__(self, hidden=(128, 128), embedding_dim=8):
        super().__init__()
        self.config = {"hidden": list(hidden), "embedding_dim": embedding_dim}
        self.embedding = nn.Embedding(SCHEMA["cardVocabulary"], embedding_dim, padding_idx=0)
        sizes = [SCHEMA["featureSize"] + SCHEMA["cardInputs"] * embedding_dim, *hidden]
        self.trunk = nn.ModuleList(nn.Linear(a, b) for a, b in zip(sizes, sizes[1:]))
        self.heads = nn.ModuleDict({name: nn.Linear(sizes[-1], size) for name, size in
                                   [("policy", SCHEMA["actionSize"]), *[(h, 2) for h in VALUE_HEADS]]})

    def forward(self, numeric, cards):
        x = torch.cat((numeric, self.embedding(cards).flatten(1)), dim=1)
        for layer in self.trunk:
            x = F.relu(layer(x))
        out = {name: head(x) for name, head in self.heads.items()}
        out["rank"] = out["rank"].tanh()
        return out


def export(model, path):
    def layer(module):
        return {"weight": module.weight.detach().cpu().tolist(), "bias": module.bias.detach().cpu().tolist()}
    data = {"version": SCHEMA["version"], "embedding": model.embedding.weight.detach().cpu().tolist(),
            "trunk": [layer(m) for m in model.trunk], "heads": {k: layer(m) for k, m in model.heads.items()}}
    Path(path).write_text(json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n")


def save(model, optimizer, prefix, steps):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "config": model.config, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "steps": steps}, str(prefix) + ".pt")
    export(model, str(prefix) + ".json")


def load(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["schema"] != SCHEMA:
        raise ValueError("Incompatible checkpoint schema")
    model = Model(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, checkpoint


def read_games(paths):
    games = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            game = json.loads(line)
            if game.get("version") != SCHEMA["version"]:
                raise ValueError(f"Incompatible trajectory schema; expected {SCHEMA['version']}")
            if game.get("phase") != "ended" or not game.get("samples"):
                raise ValueError("Only complete, nonempty trajectories may be trained")
            if len(game["samples"]) != len(game["actions"]):
                raise ValueError("Trajectory sample/action mismatch")
            for sample in game["samples"]:
                legal, policy = sample["legal"], sample["policy"]
                if (len(sample["numeric"]) != SCHEMA["featureSize"] or
                        not all(math.isfinite(v) for v in sample["numeric"]) or
                        len(sample["cards"]) != SCHEMA["cardInputs"] or
                        any(type(c) is not int or not 0 <= c < SCHEMA["cardVocabulary"] for c in sample["cards"])):
                    raise ValueError("Invalid feature schema")
                if (not legal or len(legal) != len(policy) or len(set(legal)) != len(legal) or
                        any(type(i) is not int or not 0 <= i < SCHEMA["actionSize"] for i in legal) or
                        sample["actionId"] not in legal or
                        any(not math.isfinite(p) or p < 0 for p in policy) or abs(sum(policy) - 1) > 1e-5):
                    raise ValueError("Invalid legal policy target")
                seat = sample["seat"]
                if seat not in (0, 1):
                    raise ValueError("Invalid player seat")
                ranks, scores = game["ranks"], game["scores"]
                rank = (ranks[1 - seat] > ranks[seat]) - (ranks[1 - seat] < ranks[seat])
                relative = (scores[seat] - scores[1 - seat]) / SCHEMA["scoreScale"]
                expected = {"rank": [rank, -rank], "relative": [relative, -relative],
                            "absolute": [scores[seat] / SCHEMA["scoreScale"], scores[1 - seat] / SCHEMA["scoreScale"]]}
                for head in VALUE_HEADS:
                    if (len(sample["value"][head]) != 2 or
                            any(not math.isfinite(a) or abs(a - b) > 1e-6
                                for a, b in zip(sample["value"][head], expected[head]))):
                        raise ValueError("Value target does not match the terminal outcome")
            games.append(game)
    if not games:
        raise ValueError("No training games")
    return games


def batch(samples, device):
    numeric = torch.tensor([s["numeric"] for s in samples], device=device, dtype=torch.float32)
    cards = torch.tensor([s["cards"] for s in samples], device=device, dtype=torch.long)
    target = torch.zeros(len(samples), SCHEMA["actionSize"], device=device)
    mask = torch.zeros_like(target, dtype=torch.bool)
    for i, s in enumerate(samples):
        mask[i, s["legal"]] = True
        target[i, s["legal"]] = torch.tensor(s["policy"], device=device, dtype=torch.float32)
    values = {name: torch.tensor([s["value"][name] for s in samples], device=device, dtype=torch.float32) for name in VALUE_HEADS}
    return numeric, cards, target, mask, values


def losses(model, samples, device):
    numeric, cards, target, mask, values = batch(samples, device)
    out = model(numeric, cards)
    # Finite sentinel avoids 0 * -inf in soft-target cross entropy.
    log_policy = F.log_softmax(out["policy"].masked_fill(~mask, -1e9), dim=1)
    return {"policy": -(target * log_policy).sum(1).mean(),
            **{name: F.mse_loss(out[name], values[name]) for name in VALUE_HEADS}}


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model, checkpoint = load(args.resume, args.device) if args.resume else (
        Model(args.hidden, args.embedding_dim).to(args.device), None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        steps = checkpoint["steps"]
    if args.command == "init":
        save(model, optimizer, args.out, steps)
        print(json.dumps({"event": "initialized", "parameters": sum(p.numel() for p in model.parameters()),
                          "config": model.config, "seconds": time.perf_counter() - started}), flush=True)
        return
    if not args.data:
        raise ValueError("train requires --data trajectories.jsonl [more.jsonl ...]")
    load_started = time.perf_counter()
    games = read_games(args.data)
    data_seconds = time.perf_counter() - load_started
    # Split by environment seed, never by positions from the same game/paired seed.
    keys = sorted({g["seed"] for g in games})
    random.shuffle(keys)
    count = min(len(keys) - 1, max(1, round(len(keys) * args.validation))) if len(keys) > 1 and args.validation else 0
    held_out = set(keys[:count])
    training = [s for g in games if g["seed"] not in held_out for s in g["samples"]]
    validation = [s for g in games if g["seed"] in held_out for s in g["samples"]]
    # Synchronize accelerators so measured time includes completed work, not just dispatch.
    def sync():
        device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    train_seconds = validation_seconds = 0.0
    for epoch in range(args.epochs):
        sync()
        epoch_started = time.perf_counter()
        model.train()
        random.shuffle(training)
        totals = dict.fromkeys(("policy", *VALUE_HEADS), 0.0)
        for offset in range(0, len(training), args.batch_size):
            samples = training[offset:offset + args.batch_size]
            optimizer.zero_grad()
            components = losses(model, samples, args.device)
            total = sum(components.values())
            if not torch.isfinite(total):
                raise ValueError("Nonfinite training loss")
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
            for name, value in components.items():
                totals[name] += value.item() * len(samples)
        sync()
        train_elapsed = time.perf_counter() - epoch_started
        train_seconds += train_elapsed
        report = {"epoch": epoch + 1, "steps": steps, "trainSamples": len(training), "validationSamples": len(validation),
                  "train": {k: v / len(training) for k, v in totals.items()},
                  "trainSeconds": train_elapsed, "trainSamplesPerSecond": len(training) / train_elapsed}
        validation_started = time.perf_counter()
        if validation:
            model.eval()
            totals = dict.fromkeys(totals, 0.0)
            with torch.no_grad():
                for offset in range(0, len(validation), args.batch_size):
                    samples = validation[offset:offset + args.batch_size]
                    for k, v in losses(model, samples, args.device).items():
                        totals[k] += v.item() * len(samples)
            report["validation"] = {k: v / len(validation) for k, v in totals.items()}
        sync()
        validation_elapsed = time.perf_counter() - validation_started
        validation_seconds += validation_elapsed
        report["validationSeconds"] = validation_elapsed
        print(json.dumps(report), flush=True)
    save_started = time.perf_counter()
    save(model, optimizer, args.out, steps)
    print(json.dumps({"event": "training_complete", "steps": steps, "device": args.device,
                      "threads": args.threads, "parameters": sum(p.numel() for p in model.parameters()),
                      "dataSeconds": data_seconds, "trainSeconds": train_seconds,
                      "validationSeconds": validation_seconds, "saveSeconds": time.perf_counter() - save_started,
                      "runSeconds": time.perf_counter() - started,
                      "trainSamplesPerSecond": args.epochs * len(training) / train_seconds}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "train"])
    parser.add_argument("--out", required=True, help="Checkpoint prefix; writes .pt and .json")
    parser.add_argument("--resume", help="Trusted .pt checkpoint, including Adam state")
    parser.add_argument("--data", nargs="+")
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--validation", type=float, default=0.1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    args = parser.parse_args()
    if (min(*args.hidden, args.embedding_dim, args.epochs, args.batch_size, args.threads) < 1
            or not 0 <= args.validation < 1 or not math.isfinite(args.lr) or args.lr <= 0):
        parser.error("Invalid architecture or training parameters")
    run(args)
