import {
  applyAction,
  CARDS,
  COLORS,
  patternCoordinates,
  scorePlayer,
  type Color,
  type Player,
  type Score,
} from "../../src/game/engine";
import { sampleGame } from "../../src/game/ai/observation";
import type { AuditPlan, CaseResult } from "./mcts-audit";

const esc = (value: unknown) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ]!,
  );
const json = (value: unknown) => JSON.stringify(value).replace(/</g, "\\u003c");
const seconds = (ms: number) => `${(ms / 1000).toFixed(3)} 秒`;

function board(player: Player, previous?: Player): string {
  const hex = Array.from(
    { length: 6 },
    (_, i) =>
      `${Math.cos((i * Math.PI) / 3) * 43},${Math.sin((i * Math.PI) / 3) * 43}`,
  ).join(" ");
  return `<svg class="board" viewBox="0 0 396 443" role="img" aria-label="${esc(player.name)} 的版图">${player.board
    .map((cell, i) => {
      const changed =
        previous && JSON.stringify(cell) !== JSON.stringify(previous.board[i]);
      const x = 48 + cell.col * 75,
        y = 60 + (cell.row + (cell.col % 2) * 0.5) * 82;
      const name = `${cell.col + 1}-${cell.row + 1}：${cell.tokens.map((c) => COLORS[c].name).join(" / ") || "空"}${cell.animal ? `；${CARDS[cell.animal].name}入住` : ""}`;
      return `<g transform="translate(${x} ${y})"><title>${esc(name)}</title>
      <polygon points="${hex}" fill="${changed ? "#e0ece1" : "#eee9dd"}" stroke="${changed ? "#21705c" : "#d9d0bd"}" stroke-width="${changed ? 3 : 1}"/>
      ${cell.tokens.map((c, j) => `<ellipse cy="${8 - j * 12}" rx="29" ry="20" fill="${COLORS[c].hex}" stroke="#fff6d8" stroke-width="1.5"/>`).join("")}
      ${cell.tokens.length ? `<text y="${14 - (cell.tokens.length - 1) * 12}" text-anchor="middle" fill="white" font-size="20">${COLORS[cell.tokens.at(-1)!].symbol}</text><text x="30" y="13" text-anchor="middle" font-size="10">${cell.tokens.length}层</text>` : ""}
      ${cell.animal ? `<path d="M-8 -23 L0 -31 L8 -23 L0 -15 Z" fill="#ffc750" stroke="#8e632d"/><text text-anchor="middle" y="-34" font-size="10">${esc(CARDS[cell.animal].name)}</text>` : ""}
      <text y="35" text-anchor="middle" fill="#4f5c50" font-size="10">${cell.col + 1}-${cell.row + 1}</text>
    </g>`;
    })
    .join("")}</svg>`;
}
function pattern(id: number): string {
  const card = CARDS[id],
    coords = patternCoordinates(card, { col: 3, row: 2 }, 0).map((c) => ({
      x: c.col * 38,
      y: (c.row + (c.col % 2) * 0.5) * 44,
    }));
  const minX = Math.min(...coords.map((c) => c.x)) - 25,
    minY = Math.min(...coords.map((c) => c.y)) - 52;
  const width = Math.max(...coords.map((c) => c.x)) - minX + 25,
    height = Math.max(...coords.map((c) => c.y)) - minY + 22;
  return `<svg class="pattern" viewBox="${minX} ${minY} ${width} ${height}" aria-label="栖息地图案">${coords
    .map(
      (p, i) =>
        `<g>${[...card.pattern[i].colors]
          .reverse()
          .map(
            (c, j) =>
              `<ellipse cx="${p.x}" cy="${p.y - j * 10}" rx="19" ry="14" fill="${c === 7 ? "#b7aa8f" : COLORS[c as Color].hex}" stroke="#fff6df"/><text x="${p.x}" y="${p.y - j * 10 + 4}" text-anchor="middle" fill="white" font-size="12">${c === 7 ? "?" : COLORS[c as Color].symbol}</text>`,
          )
          .join(
            "",
          )}${card.pattern[i].allowCube ? `<path d="M${p.x} ${p.y - (card.pattern[i].colors.length - 1) * 10 - 32}l6 6-6 6-6-6Z" fill="#ffc750" stroke="#8e632d"/>` : ""}</g>`,
    )
    .join("")}</svg>`;
}
function card(id: number, placed?: number): string {
  const c = CARDS[id];
  return `<div class="card"><b>#${id} ${esc(c.name)}</b>${pattern(id)}<small>积分 ${c.points.join(" / ")}</small>${placed !== undefined ? `<strong>${placed} / ${c.points.length} 已入住${placed === c.points.length ? " · 完成" : ""}</strong>` : ""}</div>`;
}
function scores(score: Score): string {
  return `<div class="scores">${(
    [
      ["树", score.trees],
      ["山", score.mountains],
      ["田野", score.fields],
      ["建筑", score.buildings],
      ["水域", score.water],
      ["动物", score.animals],
    ] as const
  )
    .map(([name, value]) => `<span>${name} <b>${value}</b></span>`)
    .join("")}</div>`;
}
const css = `
:root{font-family:ui-sans-serif,system-ui,-apple-system,"PingFang SC",sans-serif;color:#203a30;background:#f4f2eb;font-size:15px;line-height:1.6}*{box-sizing:border-box}body{margin:0}main{max-width:1460px;margin:auto;padding:40px 32px 80px}h1{font-size:34px;line-height:1.3;margin:8px 0 12px}h2{font-size:23px;margin:30px 0 16px}h3{margin:0 0 6px}p{margin:8px 0 14px}a{color:#176858}small,.muted{color:#64766b}small{display:block}nav{display:flex;justify-content:space-between;gap:16px}.eyebrow{letter-spacing:2px;text-transform:uppercase;font-size:12px;color:#58715f;font-weight:700}.badge{display:inline-block;background:#dce9de;color:#245740;padding:3px 10px;border-radius:4px;font-size:12px}.warning{background:#fff0d2;color:#714c15;border-left:3px solid #ca9433;padding:12px 16px}.intro{max-width:980px}.panel,.method,.plan{background:#fffef9;border:1px solid #dedfd2;border-radius:12px;padding:20px}.position{display:grid;grid-template-columns:1fr 1fr;gap:20px}.player-layout{display:grid;grid-template-columns:minmax(200px,1.2fr) minmax(150px,1fr);align-items:start;gap:10px}.board{width:100%;height:auto;max-width:410px;display:block;margin:auto}.cards{display:flex;flex-wrap:wrap;gap:8px;align-content:start}.card{background:#f4f1e6;border:1px solid #e2ddce;border-radius:8px;padding:9px;min-width:96px;flex:1;max-width:150px;font-size:12px;line-height:1.5}.card strong{display:block;font-size:11px;color:#476337}.pattern{display:block;width:100%;height:78px;margin:5px 0}.scores{display:flex;flex-wrap:wrap;gap:6px;font-size:11px}.scores span{padding:3px 7px;background:#f0f2e9;border-radius:4px}.market{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:20px}.market .card{max-width:none}.supply{display:flex;gap:12px;flex-wrap:wrap}.supply-group{padding:10px;background:#f4f1e6;border-radius:8px;text-align:center}.token{display:inline-flex;width:32px;height:32px;border-radius:50%;align-items:center;justify-content:center;color:#fff;font-size:20px;margin:3px}.legend{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0;font-size:12px}.methods{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;align-items:start}.method{padding:16px}.method>p{font-size:13px}.plan{padding:14px;margin-top:15px;background:#fff}.plan h3{font-size:17px}.plan .board{max-width:330px}.plan ol{padding-left:23px;font-size:12px;line-height:1.85}.plan li.current{background:#e1efe1;font-weight:700}.score-big{font-size:28px;font-weight:700}.gain{font-size:13px;color:#54715d;margin-left:8px}.step{display:flex;gap:8px;align-items:center;font-size:12px}input[type=range]{width:100%}details{margin-top:12px}summary{cursor:pointer;color:#4a6857;font-size:13px}.metadata{font-size:12px}table{border-collapse:collapse;width:100%;font-size:13px;background:#fffef9}th,td{text-align:left;padding:10px 13px;border-bottom:1px solid #e6e5da}th{background:#e8ece0;position:sticky;top:0}tr:hover{background:#f1f5eb}.summary{display:flex;flex-wrap:wrap;gap:14px;margin:22px 0}.stat{background:#fffef9;border:1px solid #dedfd2;padding:16px 22px;border-radius:10px}.stat b{display:block;font-size:27px}.filters{display:flex;gap:18px;flex-wrap:wrap;margin:15px 0}button{padding:6px 12px;background:#e5edde;border:1px solid #c6d6c1;border-radius:5px;color:#224b39;cursor:pointer}.foot{margin-top:30px;color:#69766d;font-size:12px}@media(max-width:1100px){.player-layout{grid-template-columns:1fr}.cards .card{max-width:none}.methods{gap:10px}.method{padding:10px}}@media(max-width:760px){main{padding:22px 15px}.position,.market,.methods{grid-template-columns:1fr}h1{font-size:27px}.card{flex:0 1 120px}table{font-size:11px}th,td{padding:6px}.table-wrap{overflow:auto}}`;
function page(title: string, content: string, script = "") {
  return `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>${esc(title)}</title><style>${css}</style></head><body><main>${content}</main>${script ? `<script>${script}</script>` : ""}</body></html>`;
}

export function caseHtml(result: CaseResult, ids: string[]): string {
  const { position: c, mcts, exact } = result,
    view = c.input,
    active = view.active;
  const index = ids.indexOf(c.id),
    initial = view.players[active];
  const frames: Record<string, { player: Player; score: Score }[]> = {};
  function planHtml(plan: AuditPlan, key: string, rank: number) {
    let game = sampleGame(view, 0);
    frames[key] = [
      {
        player: game.players[active],
        score: scorePlayer(game.players[active], game.side),
      },
    ];
    for (const action of plan.actions) {
      game = applyAction(game, action);
      game.log = [];
      frames[key].push({
        player: game.players[active],
        score: scorePlayer(game.players[active], game.side),
      });
    }
    return `<article class="plan" id="${key}"><h3>#${rank + 1} <span class="score-big">${plan.score.total}</span> 分<span class="gain">${plan.gain >= 0 ? "+" : ""}${plan.gain}</span></h3>
      ${scores(plan.score)}<div class="frame-board">${board(plan.player, initial)}</div>
      <div class="step"><span>起点</span><input aria-label="方案 ${key} 动作步骤" type="range" min="0" max="${plan.actions.length}" value="${plan.actions.length}" data-plan="${key}"><span>终点</span></div>
      <small class="frame-label">第 ${plan.actions.length} 步 · ${esc(plan.labels.at(-1))} · ${plan.score.total} 分</small>
      <ol>${plan.labels.map((label, i) => `<li data-step="${i + 1}">${esc(label)}</li>`).join("")}</ol>
      <details><summary>回合末持牌</summary><div class="cards">${plan.player.cards.map((c) => card(c.id, c.placed)).join("")}</div></details></article>`;
  }
  const methods = mcts
    .map(
      (
        m,
        index,
      ) => `<section class="method"><span class="eyebrow">PUCT · ${m.simulations}</span><h3>${m.simulations} 次模拟</h3>
    <p>${seconds(m.elapsedMs)} · ${m.nodes.toLocaleString()} 节点<br>完成回合 ${m.completions} 次 · ${m.uniquePlans} 个不同终点</p>
    <p class="muted">根节点实际推荐：${esc(m.selectedLabel)}</p>
    ${m.plans.length < 3 ? `<p class="warning">预算内仅找到 ${m.plans.length} 个完整方案。没有额外补搜或补齐动作。</p>` : ""}
    ${m.plans.map((p, i) => planHtml(p, `m${index}-${i}`, i)).join("")}
    <details><summary>根节点访问次数与 Q</summary><table><tr><th>动作</th><th>N</th><th>Q（分）</th><th>修正权重</th></tr>${[
      ...m.rootPolicy,
    ]
      .sort((a, b) => b.visits - a.visits)
      .map(
        (e) =>
          `<tr><td>${esc(JSON.stringify(e.action))}</td><td>${e.visits}</td><td>${(e.q * 100).toFixed(2)}</td><td>${e.selectionWeight}</td></tr>`,
      )
      .join("")}</table></details>
    </section>`,
    )
    .join("");
  const exactMethod = `<section class="method"><span class="eyebrow">EXHAUSTIVE · 60s LIMIT</span><h3>${exact.complete ? "穷举完成 · 已证最优" : "穷举超时 · 当前最好"}</h3><p>${seconds(exact.elapsedMs)} · ${exact.states.toLocaleString()} 唯一状态<br>${exact.transitions.toLocaleString()} 条边 · ${exact.uniquePlans.toLocaleString()} 个不同终点<br>合并 ${exact.duplicates.toLocaleString()} 个等价状态</p>${!exact.complete ? '<p class="warning">达到时限，以下结果没有全局最优保证。</p>' : ""}${exact.plans.map((p, i) => planHtml(p, `e-${i}`, i)).join("")}</section>`;
  const content = `<nav><a href="index.html">← 200 局面索引</a><div>${index > 0 ? `<a href="${ids[index - 1]}.html">上一局面</a>　` : ""}${index + 1 < ids.length ? `<a href="${ids[index + 1]}.html">下一局面 →</a>` : ""}</div></nav>
    <header><p class="eyebrow">HARMONIES / MCTS VALIDATION</p><h1>第 ${c.gameIndex} 局 · ${esc(initial.name)} 的最后一回合</h1>
    <p>第 ${view.round} 轮 · 顺位 ${active + 1} · seed ${c.seed} · <span class="badge">A 面 / 2 人 / 无自然之灵</span> <span class="badge">最后一回合 = true</span></p></header>
    <p class="intro muted">均匀合法动作先验；value = 当前绝对得分 / 100，使用现有 PUCT 默认参数。每个预算只运行一棵树，列出实际到达的完整回合终点中得分最高的 3 个。相同己方版图和持牌仅保留一条动作路径。这里的 Top 3 与根节点按访问权重推荐的动作是不同指标。</p>
    <h2>回合开始时</h2><div class="position">${[active, 1 - active]
      .map((seat) => {
        const p = view.players[seat],
          s = scorePlayer(p, view.side);
        return `<section class="panel"><h3>${esc(p.name)} ${seat === active ? "· 当前行动" : "· 对手"} <span class="score-big">${s.total}</span> 分</h3>${scores(s)}<div class="player-layout">${board(p)}<div class="cards">${p.cards.map((c) => card(c.id, c.placed)).join("") || '<p class="muted">无持牌</p>'}</div></div></section>`;
      })
      .join("")}</div>
    <div class="panel market"><section><h3>市场动物牌</h3><div class="cards">${view.market.map((id) => (id ? card(id) : '<div class="card">空槽位</div>')).join("")}</div></section><section><h3>地形供应</h3><div class="supply">${view.supply.map((group, i) => `<div class="supply-group"><small>第 ${i + 1} 组</small>${group.map((c) => `<span class="token" title="${COLORS[c].name}" style="background:${COLORS[c].hex}">${COLORS[c].symbol}</span>`).join("") || "空"}</div>`).join("")}</div><p class="metadata">牌堆剩余 ${view.deckCount} · 地形袋剩余 ${view.bagCount}<br>原始最后一轮标记：${c.original.finalRound ?? "未触发"}；测试输入设为第 ${view.round} 轮。<br>金色菱形为动物落点；版图坐标为“列-行”。绿色边框表示方案改变的格子。堆叠从下往上绘制，右侧标注层数。</p><div class="legend">${Object.values(
      COLORS,
    )
      .map((c) => `<span style="color:${c.hex}">${c.symbol} ${c.name}</span>`)
      .join("")}</div></section></div>
    <h2>整回合方案对照</h2><p class="muted">拖动方案下方滑块可逐步查看版图；分数栏给出方案最终分数。得分相同的终点按固定键排序，未用动物数量破同分。</p><div class="methods">${methods}${exactMethod}</div>
    <p class="foot"><a href="${c.id}.json">下载局面及搜索结果 JSON</a> · 所有展示方案均已通过引擎合法动作重放及得分复核。穷举只合并等价状态，不剪除低分状态，不交换操作规则。对手后续行动不影响己方绝对得分。</p>`;
  return page(
    `${c.id} · MCTS 验证`,
    content,
    `const __name=(x)=>x;const COLORS=${json(COLORS)},CARDS=${json(CARDS)};const esc=${esc.toString()};const board=${board.toString()};const frames=${json(frames)},initial=${json(initial)};document.querySelectorAll('input[data-plan]').forEach(input=>input.addEventListener('input',()=>{const id=input.dataset.plan,n=Number(input.value),article=document.getElementById(id),frame=frames[id][n];article.querySelector('.frame-board').innerHTML=board(frame.player,initial);const label=n?article.querySelector('[data-step="'+n+'"]').textContent:'回合开始';article.querySelector('.frame-label').textContent='第 '+n+' 步 · '+label+' · '+frame.score.total+' 分';article.querySelectorAll('li').forEach(li=>li.classList.toggle('current',Number(li.dataset.step)===n));}));`,
  );
}

export function indexHtml(
  results: CaseResult[],
  metadata: Record<string, unknown>,
): string {
  const complete = results.filter((r) => r.exact.complete);
  const hit = (budget: number) =>
    complete.filter(
      (r) =>
        r.mcts[budget].plans[0]?.score.total === r.exact.plans[0]?.score.total,
    ).length;
  const insufficient = (budget: number) =>
    results.filter((r) => r.mcts[budget].plans.length < 3).length;
  const rows = results
    .map((r) => {
      const c = r.position,
        optimum = r.exact.complete ? r.exact.plans[0]?.score.total : undefined;
      const cells = r.mcts
        .map((m) => {
          const score = m.plans[0]?.score.total;
          return `<td>${score ?? "—"}${score !== undefined && optimum !== undefined ? `<small>差距 ${optimum - score}</small>` : ""}</td><td>${m.plans.length}/3</td>`;
        })
        .join("");
      return `<tr data-incomplete="${r.mcts.some((m) => m.plans.length < 3)}" data-miss="${r.exact.complete && r.mcts[1].plans[0]?.score.total !== optimum}"><td><a href="${c.id}.html">第 ${c.gameIndex} 局 · 顺位 ${c.input.active + 1}</a><small>${c.input.players[c.input.active].name} · seed ${c.seed}</small></td><td>${c.input.round}</td><td>${scorePlayer(c.input.players[c.input.active], "A").total}</td>${cells}<td>${r.exact.plans[0]?.score.total ?? "—"}<small>${r.exact.complete ? "已证最优" : "超时 / 下界"}</small></td><td>${seconds(r.exact.elapsedMs)}</td></tr>`;
    })
    .join("");
  return page(
    "Harmonies · MCTS 最后一回合验证",
    `<p class="eyebrow">HARMONIES / SEARCH AUDIT</p><h1>MCTS 最后一回合验证</h1><p class="intro">100 局均匀随机自对弈，双方最后一回合开始时各取一个局面，共 200 个。比较整棵树 128 / 1024 次模拟实际找到的 Top 3 完整回合方案，以及每局面最多 60 秒的穷举结果。</p>
    <div class="summary"><div class="stat"><b>${results.length} / 200</b>已生成局面报告</div><div class="stat"><b>${complete.length} / ${results.length}</b>穷举完成</div><div class="stat"><b>${hit(0)} / ${complete.length}</b>128 次找到最优分</div><div class="stat"><b>${hit(1)} / ${complete.length}</b>1024 次找到最优分</div></div>
    <p class="warning">128 次有 ${insufficient(0)} 个局面不足 3 个完整方案；1024 次有 ${insufficient(1)} 个。空结果表示树尚未走到 endTurn，不代表没有合法完整回合。MCTS 的结果是预算内已发现的最好方案，只有穷举完成才证明全局最优。</p>
    <details><summary>实验口径与复现信息</summary><p>uniform policy；value = 当前绝对得分 / 100；rank/relative 权重为 0，absolute 权重为 1；无根噪声、temperature 0。沿用生产 MCTS 的 PUCT、FPU、展开、回传和根动作选择，新增只读完整回合观察回调。首位玩家之后仍按现有算法搜索对手的最后一回合；绝对得分不会因此改变。Top 3 按不同己方回合末版图和持牌去重，仅展示实际访问到的终点，未做额外补搜。</p><pre>${esc(JSON.stringify(metadata, null, 2))}</pre><a href="inputs.json">原始 100 局动作及 200 个输入局面</a> · <a href="summary.json">汇总 JSON</a></details>
    <div class="filters"><label><input type="checkbox" id="incomplete"> 只看不足 3 方案</label><label><input type="checkbox" id="miss"> 只看 1024 未找到已证最优分</label></div>
    <div class="table-wrap"><table><thead><tr><th>局面 / 详细报告</th><th>轮次</th><th>当前分</th><th>128 最好</th><th>方案数</th><th>1024 最好</th><th>方案数</th><th>穷举最好</th><th>穷举耗时</th></tr></thead><tbody>${rows}</tbody></table></div><p class="foot">静态 HTML，无网络依赖。每个局面报告包含双方版图、持牌栖息地图案、市场和供应、逐步方案回放及根节点搜索统计。</p>`,
    `for(const input of document.querySelectorAll('.filters input'))input.addEventListener('change',()=>document.querySelectorAll('tbody tr').forEach(row=>row.hidden=(document.getElementById('incomplete').checked&&row.dataset.incomplete!=='true')||(document.getElementById('miss').checked&&row.dataset.miss!=='true')));`,
  );
}
