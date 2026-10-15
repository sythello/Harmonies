"""Fixed endgame fitting experiment, using the production model/loss and TS serving."""
import argparse
import csv
import json
import math
import random
import subprocess
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from train import Model, SCHEMA, VALUE_HEADS, batch, losses, read_games, save

ROOT = Path(__file__).resolve().parents[1]


def value_metrics(prediction, target, head):
    if head == "rank":
        choices = torch.tensor([-1., 0., 1.], device=prediction.device)
        rounded = choices[(prediction[..., None] - choices).abs().argmin(-1)]
        expected = target
        scale = 1
    else:
        scale = SCHEMA["scoreScale"]
        points = prediction * scale
        rounded = points.sign() * (points.abs() + 0.5).floor()
        expected = (target * scale).round()
    exact = rounded == expected
    mse = ((prediction - target) ** 2).mean(0)
    return {"exact": exact.float().mean().item(), "exactOwn": exact[:, 0].float().mean().item(),
            "exactOpponent": exact[:, 1].float().mean().item(),
            "exactBoth": exact.all(1).float().mean().item(), "mse": mse.mean().item(),
            "mseOwn": mse[0].item(), "mseOpponent": mse[1].item(),
            "msePoints": (mse.mean() * scale ** 2).item()}


def metrics(model, samples):
    model.eval()
    with torch.no_grad():
        numeric, cards, target, mask, values = batch(samples, "cpu")
        out = model(numeric, cards)
        logits = out["policy"].masked_fill(~mask, -1e9)
        argmax = logits.argmax(1).tolist()
        logp = F.log_softmax(logits, 1)
        ce = -(target * logp).sum(1).mean().item()
        entropy = -(target * target.clamp_min(1e-30).log()).sum(1).mean().item()
        result = {"samples": len(samples), "policyHit": sum(a in s["exact"]["optimalIds"]
                  for a, s in zip(argmax, samples)) / len(samples), "policyCE": ce, "policyKL": ce - entropy,
                  **{head: value_metrics(out[head], values[head], head) for head in VALUE_HEADS}}
    return result, out, argmax


def attach_serving(result, samples, prediction, argmax, rows):
    indexed = {row["id"]: row for row in rows}
    rows = [indexed[s["id"]] for s in samples]
    # Compare both runtimes on all 1000 root predictions every epoch.
    error = max(abs(row["value"][h][j] - prediction[h][i, j].item())
                for i, row in enumerate(rows) for h in VALUE_HEADS for j in range(2))
    if error > 2e-5:
        raise ValueError(f"Python/TS prediction mismatch: {error}")
    disagreements = 0
    for i, (a, row) in enumerate(zip(argmax, rows)):
        if a != row["argmax"]:
            disagreements += 1
            if abs(prediction["policy"][i, a].item() - prediction["policy"][i, row["argmax"]].item()) > 2e-5:
                raise ValueError("Python/TS policy argmax mismatch away from a numerical tie")
    n = len(rows)
    result.update({"servingHit": sum(r["serving"]["hit"] for r in rows)/n,
                   "servingFirstHit": sum(r["serving"]["firstHit"] for r in rows)/n,
                   "servingRegret": sum(r["serving"]["regret"] for r in rows)/n,
                   "greedyTurnHit": sum(r["greedy"]["hit"] for r in rows)/n,
                   "greedyRegret": sum(r["greedy"]["regret"] for r in rows)/n,
                   "servingSeconds": sum(r["seconds"] for r in rows),
                   "inheritedVisits": sum(r["serving"]["inherited"] for r in rows),
                   "parityMaxValueError": error, "parityArgmaxTies": disagreements})


def report(directory, history):
    (directory / "epochs.json").write_text(json.dumps(history, indent=2) + "\n")
    rows = []
    for epoch in history:
        for split in ("train", "test"):
            row = {"epoch": epoch["epoch"], "split": split}
            for k, v in epoch[split].items():
                if isinstance(v, dict):
                    row.update({k+"_"+a: b for a, b in v.items()})
                else:
                    row[k] = v
            rows.append(row)
    with (directory / "epochs.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Self-contained SVG curves and an exact per-epoch table; no CDN dependency.
    def chart(title, key, percent=True):
        def get(entry):
            for part in key.split("."):
                entry = entry[part]
            return entry
        series = [[get(e[s]) for e in history] for s in ("train", "test")]
        high = 1 if percent else max(0.0001, max(max(x) for x in series))
        svg = f'<svg viewBox="0 0 700 220" role="img" aria-label="{title}">'
        for i in range(5):
            y = 185 - i*40
            svg += f'<path d="M60 {y} H680" stroke="#dce3ec"/><text x="3" y="{y+4}">{high*i/4:.2f}</text>'
        for values, color in zip(series, ("#2167d5", "#d67812")):
            points = " ".join(f'{60+i*620/max(1,len(history)-1):.1f},{185-v/high*160:.1f}' for i,v in enumerate(values))
            svg += f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.4"/>'
        svg += f'<text x="60" y="211">epoch 1</text><text x="615" y="211">epoch {history[-1]["epoch"]}</text></svg>'
        return f'<section><h2>{title}</h2>{svg}</section>'
    table = "".join(f'<tr><td>{e["epoch"]}</td><td>{"训练" if s=="train" else "测试"}</td>'
                    + "".join(f'<td>{e[s][k]:.1%}</td>' for k in ("policyHit", "servingFirstHit", "servingHit", "greedyTurnHit"))
                    + f'<td>{e[s]["servingRegret"]:.3f}</td>'
                    + "".join(f'<td>{e[s][h]["exact"]:.1%}</td><td>{e[s][h]["mse"]:.6f}</td>' for h in VALUE_HEADS)
                    + '</tr>' for e in history for s in ("train","test"))
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Harmonies NN 拟合实验</title><style>body{font:15px system-ui;margin:28px;background:#f3f6fb;color:#17283e}
main{max-width:1450px;margin:auto}h1{font-size:28px}p{line-height:1.7}section{background:white;padding:16px;border-radius:12px}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,500px),1fr));gap:16px}h2{font-size:18px}svg{width:100%;font-size:12px}
.table{overflow:auto;background:white;border-radius:12px;margin-top:24px}table{border-collapse:collapse;white-space:nowrap;width:100%}
th,td{padding:10px;text-align:right;border-bottom:1px solid #dce3ec}th{background:#e9eff8}tr:nth-child(even){background:#f7f9fc}</style>
<main><h1>Harmonies NN 末回合拟合实验</h1>'''
    html += f'<p>已完成 {len(history)} epochs · 训练 {history[-1]["train"]["samples"]} / 测试 {history[-1]["test"]["samples"]} 个局面 · 按整局划分。<br>'
    html += '随机对弈到最后回合，按真实时序穷举双方最优得分。每个局面仅监督回合起点；policy target = 前 512 次 PUCT 路径访问 + 最优路径 512 次访问。</p>'
    html += '<p>Policy 命中：首动作可通向任一最优整回合。Serving 首动作 / 整回合分别统计；每个原子动作新增 128 次模拟，复用回合内搜索树。'
    html += 'Serving 使用生产默认综合 value 权重（排名 1、相对分 .25、绝对分 .05）；穷举目标是自己得分，所以目标并不完全相同。'
    html += '贪心整回合额外检查直接逐步取 policy argmax 的表现。回合内中间状态没有训练标签。</p>'
    html += '<p>Value exact：排名取最近 {-1,0,1}；分数还原 ×100 后四舍五入。表内 exact 为己方/对手两输出平均；MSE 为网络归一化单位。'
    html += 'CSV / JSON 另含双方分别、双方同时 exact 和分数单位 MSE。蓝色＝训练，橙色＝测试。</p>'
    if (directory / "supervision-coverage.json").exists():
        coverage = json.loads((directory / "supervision-coverage.json").read_text())
        html += f'<p><strong>监督范围：</strong>训练集中，138 个放置动作的 policy 输出仅有 {coverage["coverage"]["place"]["supervisedHeads"]} 个进入过合法 mask。'
        html += '因此首动作拟合成功不代表放置动作已经学会；中间节点的待放片和已拿片等特征也没有在这批回合起点数据中得到训练。</p>'
    if (directory / "tiny-overfit.json").exists():
        tiny = json.loads((directory / "tiny-overfit.json").read_text())
        html += f'<p><strong>独立容量对照：</strong>{tiny["samples"]} 个训练样本，全 batch 1000 步，750 步后 lr 降至 1/10。'
        html += f'Policy 首动作命中 {tiny["after"]["policyHit"]:.1%}；'
        html += '、'.join(f'{h} exact {tiny["after"][h]["exact"]:.1%}' for h in VALUE_HEADS)
        html += '。此对照不参与下方 50 epoch 的权重或评测。</p>'
    html += '<p><a href="epochs.csv">完整 CSV</a> · <a href="epochs.json">完整 JSON</a> · <a href="manifest.json">实验协议</a> · <a href="verification.json">穷举核查</a></p><div class="charts">'
    html += chart("Policy 首动作最优命中率", "policyHit") + chart("Serving 整回合最优命中率", "servingHit")
    html += chart("Policy KL（去除 target 熵）", "policyKL", False) + chart("Serving 平均失分", "servingRegret", False)
    html += chart("排名 exact", "rank.exact") + chart("相对得分 exact", "relative.exact")
    html += chart("绝对得分 exact", "absolute.exact") + chart("绝对得分 MSE（分数单位）", "absolute.msePoints", False)
    html += '</div><div class="table"><table><thead><tr><th>Epoch</th><th>集合</th><th>Policy 首动作</th><th>Serving 首动作</th><th>Serving 整回合</th><th>贪心整回合</th><th>Serving 失分</th>'
    html += ''.join(f'<th>{h} exact</th><th>{h} MSE</th>' for h in VALUE_HEADS)
    html += '</tr></thead><tbody>'+table+'</tbody></table></div></main></html>'
    (directory / "report.html").write_text(html)


def run(args):
    directory = Path(args.out).resolve()
    games = read_games([directory / "dataset.jsonl"])
    assert len({g["seed"] for g in games}) == len(games)
    assert all(len(g["samples"]) == 2 for g in games)
    splits = {k: [s for g in games if g["split"] == k for s in g["samples"]] for k in ("train", "test")}
    assert all(splits.values())
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.tiny_overfit:
        samples = splits["train"][:16]
        before = metrics(model, samples)[0]
        started = time.perf_counter()
        for step in range(1000):
            if step == 750:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr / 10
            model.train()
            optimizer.zero_grad()
            loss = sum(losses(model, samples, "cpu").values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
        result = {"samples": len(samples), "steps": 1000, "seed": args.seed,
                  "lr": args.lr, "lrAfterStep750": args.lr/10,
                  "seconds": time.perf_counter()-started, "sampleIds": [s["id"] for s in samples],
                  "before": before, "after": metrics(model, samples)[0]}
        (directory / "tiny-overfit.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result), flush=True)
        return
    (directory / "models").mkdir(exist_ok=True)
    (directory / "evaluations").mkdir(exist_ok=True)
    config = {**vars(args), "parameters": sum(p.numel() for p in model.parameters()),
              "architecture": model.config, "batchSize": 64, "threads": 1, "device": "cpu"}
    (directory / "training-config.json").write_text(json.dumps(config, indent=2)+"\n")
    history, steps = [], 0
    started = time.perf_counter()
    baseline = {s: metrics(model, samples)[0] for s, samples in splits.items()}
    (directory / "untrained.json").write_text(json.dumps(baseline, indent=2)+"\n")
    for epoch in range(1, args.epochs+1):
        t = time.perf_counter()
        model.train()
        samples = list(splits["train"])
        random.shuffle(samples)
        for offset in range(0, len(samples), 64):
            optimizer.zero_grad()
            loss = sum(losses(model, samples[offset:offset+64], "cpu").values())
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
            optimizer.step()
            steps += 1
        training_seconds = time.perf_counter()-t
        prefix = directory / "models" / f"epoch-{epoch:03}"
        save(model, optimizer, prefix, steps)
        evaluated = directory / "evaluations" / f"epoch-{epoch:03}.json"
        print(json.dumps({"event": "epoch_trained", "epoch": epoch, "trainSeconds": training_seconds}), flush=True)
        subprocess.run(["node", "--import", "tsx", "scripts/nn-fit.ts", "evaluate", "--out", str(directory),
                        "--model", str(prefix)+".json", "--result", str(evaluated), "--workers", str(args.workers)],
                       cwd=ROOT, check=True)
        serving = json.loads(evaluated.read_text())
        entry = {"epoch": epoch, "steps": steps, "trainSeconds": training_seconds, "evalSeconds": serving["seconds"]}
        for split, samples in splits.items():
            entry[split], prediction, argmax = metrics(model, samples)
            attach_serving(entry[split], samples, prediction, argmax, serving["rows"])
        history.append(entry)
        report(directory, history)
        print(json.dumps(entry), flush=True)
    timing = {"epochs": args.epochs, "wallSeconds": time.perf_counter()-started,
              "trainSeconds": sum(e["trainSeconds"] for e in history),
              "evalSeconds": sum(e["evalSeconds"] for e in history)}
    (directory / "timing.json").write_text(json.dumps(timing, indent=2)+"\n")
    print(json.dumps({"event": "complete", **timing}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tiny-overfit", action="store_true", help="Separate 16-training-sample capacity control")
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Invalid training parameters")
    run(args)
