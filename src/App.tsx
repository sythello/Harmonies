import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import {
  ArrowRight,
  Check,
  ChevronRight,
  CircleHelp,
  Leaf,
  Mountain,
  Waves,
  Flower2,
  House,
  Trees,
  Undo2,
  RotateCcw,
  Users,
  Sun,
  Sparkles,
  X,
  Trophy,
  Eye,
  CheckCircle2,
  BookOpen,
  Pause,
  Play,
  PawPrint,
  Flag,
  Maximize2,
  ScrollText,
} from "lucide-react";
import {
  CARDS,
  COLORS,
  activeCards,
  canEndTurn,
  canPlace,
  createGame,
  dispatch,
  emptyCount,
  endingReason,
  isComplete,
  key,
  matches,
  patternCoordinates,
  ranking,
  scorePlayer,
  soloSuns,
  undo,
  type Action,
  type Card,
  type Cell,
  type Color,
  type Game,
  type OwnedCard,
  type Player,
  type Session,
  type Side,
} from "./game/engine";

const SAVE_KEY = "harmonies.session.v1";
const PLAYER_COLORS = ["#628979", "#ca9662", "#7e86ad", "#c28083"];
function readSave(): Session | null {
  try {
    const s = JSON.parse(localStorage.getItem(SAVE_KEY) || "null");
    if (
      s?.game?.version === 1 &&
      Array.isArray(s.game.players) &&
      Array.isArray(s.history)
    )
      return s;
  } catch {}
  return null;
}
const cx = (...values: (string | false | undefined)[]) =>
  values.filter(Boolean).join(" ");
function Token({ color, small = false }: { color: Color; small?: boolean }) {
  return (
    <span
      className={cx("token", small && "small")}
      style={{ "--token": COLORS[color].hex } as CSSProperties}
      aria-label={COLORS[color].name}
    >
      {COLORS[color].symbol}
    </span>
  );
}
function Brand() {
  return (
    <div className="brand">
      <span className="brand-symbol">
        <Leaf size={22} />
      </span>
      <div>
        HARMONIES<small>万 物 和 鸣</small>
      </div>
    </div>
  );
}
function Modal({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose?: () => void;
  wide?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    ref.current?.showModal();
    return () => ref.current?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={cx("modal", wide && "wide")}
      onCancel={(e) => {
        e.preventDefault();
        onClose?.();
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <div className="modal-heading">
        <h2>{title}</h2>
        {onClose && (
          <button className="icon-button" onClick={onClose} aria-label="关闭">
            <X size={22} />
          </button>
        )}
      </div>
      {children}
    </dialog>
  );
}
function CardFace({
  id,
  owned,
  className = "",
  onClick,
  onEnter,
  onLeave,
}: {
  id: number;
  owned?: OwnedCard;
  className?: string;
  onClick?: () => void;
  onEnter?: () => void;
  onLeave?: () => void;
}) {
  const card = CARDS[id],
    style = {
      backgroundPosition: `${(((id - 1) % 10) * 100) / 9}% ${Math.floor((id - 1) / 10) * 25}%`,
    } as CSSProperties;
  const content = (
    <>
      <span className="card-paint" style={style} />
      {owned &&
        card.points.map(
          (_, i) =>
            i >= owned.placed && (
              <span
                key={i}
                className={cx("card-cube", card.spirit && "spirit-cube")}
                style={{ top: `${5.5 + (card.points.length - 1 - i) * 16.7}%` }}
              />
            ),
        )}
      <span className="card-name">{card.name}</span>
      {owned && isComplete(owned) && (
        <span className="card-completed">
          <CheckCircle2 size={14} /> 已完成
        </span>
      )}
    </>
  );
  return onClick ? (
    <button
      className={cx("animal-card", className)}
      onClick={onClick}
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      onFocus={onEnter}
      onBlur={onLeave}
      aria-label={`${card.name}${owned ? `，已入住 ${owned.placed} / ${card.points.length}` : "，查看卡牌"}`}
    >
      {content}
    </button>
  ) : (
    <div
      className={cx("animal-card", className)}
      role="img"
      aria-label={card.name}
    >
      {content}
    </div>
  );
}
function Pattern({ card }: { card: Card }) {
  const cells = patternCoordinates(card, { col: 3, row: 2 }, 0);
  const coords = cells.map((c) => ({
    x: c.col * 42,
    y: (c.row + (c.col % 2) * 0.5) * 48,
  }));
  const minX = Math.min(...coords.map((c) => c.x)) - 30,
    minY = Math.min(...coords.map((c) => c.y)) - 45;
  const w = Math.max(...coords.map((c) => c.x)) - minX + 30,
    h = Math.max(...coords.map((c) => c.y)) - minY + 40;
  return (
    <svg
      className="pattern"
      viewBox={`${minX} ${minY} ${w} ${h}`}
      aria-label="栖息地图案，白色菱形为动物落点"
    >
      {coords.map((p, i) => (
        <g key={i}>
          {[...card.pattern[i].colors].reverse().map((v, j) => (
            <g key={j}>
              <ellipse
                cx={p.x}
                cy={p.y - j * 11}
                rx="22"
                ry="17"
                fill={v === 7 ? "#b7aa8f" : COLORS[v as Color].hex}
                stroke="#fff6df"
                strokeWidth="2"
              />
              <text
                x={p.x}
                y={p.y - j * 11 + 5}
                textAnchor="middle"
                fill="#fff7dc"
                fontSize="16"
              >
                {v === 7 ? "?" : COLORS[v as Color].symbol}
              </text>
            </g>
          ))}
          {card.pattern[i].allowCube && (
            <path
              d={`M${p.x} ${p.y - (card.pattern[i].colors.length - 1) * 11 - 26}l7 7-7 7-7-7Z`}
              fill="#fff9d4"
              stroke="#845b2a"
              strokeWidth="2"
            />
          )}
        </g>
      ))}
    </svg>
  );
}
function Board({
  player,
  side,
  selectedColor,
  selectedAnimal,
  onCell,
  mini = false,
}: {
  player: Player;
  side: Side;
  selectedColor?: Color;
  selectedAnimal?: number | null;
  onCell?: (i: number) => void;
  mini?: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const matchList = selectedAnimal ? matches(player.board, selectedAnimal) : [];
  const hoverMatch = matchList.find((m) => m.target === hover);
  const width = side === "A" ? 395 : 545,
    height = side === "A" ? 458 : 419;
  const points = Array.from(
    { length: 6 },
    (_, i) =>
      `${Math.cos((i * Math.PI) / 3) * 47},${Math.sin((i * Math.PI) / 3) * 47}`,
  ).join(" ");
  return (
    <svg
      className={cx(
        "board",
        side === "B" && "island-board",
        mini && "mini-board",
      )}
      viewBox={`0 0 ${width} ${height}`}
      role="group"
      aria-label={`${player.name} 的 ${side} 面版图`}
    >
      <defs>
        <filter
          id={`shadow-${player.id}-${mini}`}
          x="-50%"
          y="-50%"
          width="200%"
          height="220%"
        >
          <feDropShadow
            dx="0"
            dy="3"
            stdDeviation="2"
            floodColor="#584c31"
            floodOpacity=".2"
          />
        </filter>
      </defs>
      {player.board.map((cell, i) => {
        const x = 47 + cell.col * 75,
          y = 49 + (cell.row + (cell.col % 2) * 0.5) * 82;
        const valid = selectedColor
            ? canPlace(cell, selectedColor)
            : !!matchList.find((m) => m.target === i),
          pattern = hoverMatch?.cells.includes(i);
        const label = `格 ${cell.col + 1}-${cell.row + 1}：${cell.tokens.length ? cell.tokens.map((c) => COLORS[c].name).join("、") : "空地"}${cell.animal ? `，${CARDS[cell.animal].name}已入住` : ""}`;
        return (
          <g
            key={key(cell)}
            transform={`translate(${x} ${y})`}
            className={cx(
              "hex-cell",
              valid && "legal",
              pattern && "pattern-match",
              !!cell.tokens.length && "occupied",
            )}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
            onClick={() => onCell?.(i)}
            role={onCell ? "button" : undefined}
            tabIndex={onCell ? 0 : undefined}
            aria-label={label}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onCell?.(i);
              }
            }}
          >
            <title>{label}</title>
            <polygon className="hex-ground" points={points} />
            {!cell.tokens.length && (
              <text className="cell-coordinate" textAnchor="middle" y="5">
                {cell.col + 1}·{cell.row + 1}
              </text>
            )}
            {cell.tokens.map((color, j) => (
              <g
                key={j}
                transform={`translate(0 ${8 - j * 10})`}
                filter={`url(#shadow-${player.id}-${mini})`}
              >
                <ellipse
                  cy="5"
                  rx="31"
                  ry="27"
                  fill={COLORS[color].hex}
                  stroke="#00000022"
                  strokeWidth="3"
                />
                <ellipse
                  rx="31"
                  ry="27"
                  fill={COLORS[color].hex}
                  stroke="#ffffff65"
                  strokeWidth="2"
                />
                {j === cell.tokens.length - 1 && (
                  <text
                    textAnchor="middle"
                    y="8"
                    fontSize="30"
                    fill="#fff5d6"
                    opacity=".9"
                  >
                    {COLORS[color].symbol}
                  </text>
                )}
              </g>
            ))}
            {cell.tokens.length > 1 && (
              <g transform="translate(30 23)">
                <circle r="10" fill="#fcf7e9" />
                <text
                  textAnchor="middle"
                  y="4"
                  fontSize="12"
                  fontWeight="700"
                  fill="#655c43"
                >
                  {cell.tokens.length}
                </text>
              </g>
            )}
            {cell.animal !== null && (
              <g
                transform={`translate(0 ${-8 - (cell.tokens.length - 1) * 10})`}
              >
                <path
                  d="M0-14 13-6 13 8 0 16-13 8-13-6Z"
                  fill={CARDS[cell.animal].spirit ? "#fffae3" : "#e4a64b"}
                  stroke="#956323"
                  strokeWidth="1.5"
                />
                <path
                  d="m-13-6 13 8L13-6M0 2v14"
                  fill="none"
                  stroke="#fff7cf"
                  strokeWidth="1.5"
                />
                {!mini && (
                  <text
                    x="0"
                    y="35"
                    textAnchor="middle"
                    fontSize="10"
                    fill="#4b3823"
                    fontWeight="700"
                  >
                    {CARDS[cell.animal].name.replace("之灵", "")}
                  </text>
                )}
              </g>
            )}
            {valid && <circle className="legal-dot" cx="32" cy="-27" r="6" />}
          </g>
        );
      })}
    </svg>
  );
}
function Help({ side, onClose }: { side: Side; onClose: () => void }) {
  return (
    <Modal title="自然旅行手册" onClose={onClose} wide>
      <div className="help-grid">
        <section>
          <h3>每回合，创造一点自然</h3>
          <p>
            <b>必须：</b>从供应板取一整组 3 枚地形片，并全部放到自己的版图。
          </p>
          <p>
            <b>可选：</b>拿取最多 1
            张动物牌；让动物入住任意次。三种操作可穿插进行。
          </p>
          <p>
            最多保留 4 张未完成的牌，自然之灵也占 1
            个位置。牌上动物全部入住后，该牌移入完成区并保留得分。
          </p>
          <h3>堆叠规则</h3>
          <p>
            山：1–3 枚灰色。树：绿色下方可有 0–2 枚棕色。建筑：红色放在 1
            枚红、灰或棕色上方。水和田野只能放在空格。
          </p>
          <p>
            有动物的格子不能再叠片。栖息地可旋转，不能镜像；图案与高度必须准确匹配。动物入住后，即使周围图案改变，也不会离开。
          </p>
        </section>
        <section>
          <h3>何时结束？</h3>
          <p>
            任一玩家回合结束时只剩 2
            个或更少空格，或需要补充时地形袋已空，触发最后一轮。完成当前轮，让所有玩家的回合数相同。
          </p>
          <h3>独自旅行 · 官方 Solo</h3>
          <p>
            供应为 3 组地形、3
            张动物牌。每回合结束将未取的地形移出游戏，再补充供应。若本回合没有拿动物牌，可在结束时替换
            1 张公共动物牌。
          </p>
          <p>
            40 / 70 / 90 / 110 / 130 / 140 / 150 / 160 分分别获得 1–8 个太阳；A
            面加 1，无自然之灵加 2，按群组计分的自然之灵加 1。
          </p>
          <h3>撤销与保存</h3>
          <p>
            结束回合前可撤销最后一步或整个回合；结束后会揭示新供应，因此不再允许回退。游戏自动保存在当前浏览器。
          </p>
        </section>
      </div>
      <ScoreGuide side={side} />
      <div className="source-links">
        <a
          href="https://cdn.svc.asmodee.net/production-libellud/uploads/2024/12/HARMONIES_Rules_EN.pdf"
          target="_blank"
          rel="noreferrer"
        >
          官方规则 PDF ↗
        </a>
        <a
          href="https://en.doc.boardgamearena.com/Tips_harmonies"
          target="_blank"
          rel="noreferrer"
        >
          BGA 社区策略 ↗
        </a>
        <span>游戏 Johan Benvenuto · 插画 Maëva Da Silva · © Libellud</span>
      </div>
    </Modal>
  );
}
function ScoreGuide({ side }: { side: Side }) {
  const rows = [
    {
      icon: <Trees />,
      name: "树木",
      text: "高度 1 / 2 / 3 → 1 / 3 / 7 分",
      detail: "单独树干不计分。",
      color: "#879850",
    },
    {
      icon: <Mountain />,
      name: "山脉",
      text: "高度 1 / 2 / 3 → 1 / 3 / 7 分",
      detail: "必须相邻另一座山；孤山 0 分。",
      color: "#7e8c89",
    },
    {
      icon: <Flower2 />,
      name: "田野",
      text: "每组至少 2 格 → 5 分",
      detail: "再大的一组也只得 5 分。",
      color: "#c5a046",
    },
    {
      icon: <House />,
      name: "建筑",
      text: "两层且邻接至少 3 色 → 5 分",
      detail: "只看邻格最上方的颜色，红色也算。",
      color: "#ba7260",
    },
    {
      icon: <Waves />,
      name: side === "A" ? "河流" : "岛屿",
      text:
        side === "A" ? "长度 1–6 → 0 / 2 / 5 / 8 / 11 / 15" : "每个岛屿 → 5 分",
      detail:
        side === "A"
          ? "只计最佳河流；两端最短路径，超过 6 格每格 +4。"
          : "非水格（含空格）的连通区域；至少算 1 岛。",
      color: "#458e95",
    },
  ];
  return (
    <div className="score-guide">
      {rows.map((r) => (
        <div key={r.name} title={r.detail}>
          <span style={{ color: r.color }}>{r.icon}</span>
          <div>
            <b>{r.name}</b>
            <p>{r.text}</p>
            <small>{r.detail}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

function SetupScreen({
  saved,
  onStart,
  onResume,
  onHelp,
}: {
  saved: Session | null;
  onStart: (
    names: string[],
    side: Side,
    spirits: boolean,
    first: number,
  ) => void;
  onResume: () => void;
  onHelp: () => void;
}) {
  const [count, setCount] = useState(2),
    [names, setNames] = useState(["玩家 1", "玩家 2", "玩家 3", "玩家 4"]),
    [side, setSide] = useState<Side>("A"),
    [spirits, setSpirits] = useState(false),
    [first, setFirst] = useState(0);
  return (
    <div className="setup-page">
      <header className="setup-header">
        <Brand />
        <button className="text-button" onClick={onHelp}>
          <BookOpen size={17} /> 游戏规则
        </button>
      </header>
      <main className="setup-main">
        <section className="setup-story">
          <div className="eyebrow">
            <span /> A LITTLE WORLD, IN HARMONY
          </div>
          <h1>
            方寸之间，
            <br />
            万物<span>和鸣。</span>
          </h1>
          <p className="intro">
            让河流蜿蜒，让森林生长。
            <br />
            用一片片地形，构筑属于你的自然栖所。
          </p>
          <div className="hero-cards">
            <CardFace id={19} />
            <CardFace id={33} />
            <CardFace id={16} />
          </div>
          <div className="story-foot">
            <span>
              <Users size={16} /> 1–4 位旅人
            </span>
            <span>
              <Leaf size={16} /> 约 30–45 分钟
            </span>
            <span>同屏轮流 · 本地保存</span>
          </div>
        </section>
        <section className="setup-panel">
          <div className="eyebrow">NEW JOURNEY</div>
          <h2>开启自然之旅</h2>
          <p className="muted">邀上朋友，或独自享受一场安静的创造。</p>
          <div className="setup-field">
            <div className="label-row">
              <label>同行玩家</label>
              <small>
                {count === 1 ? "官方单人模式" : "同一设备，轮流操作"}
              </small>
            </div>
            <div className="count-choices">
              {[1, 2, 3, 4].map((n) => (
                <button
                  key={n}
                  className={cx(count === n && "selected")}
                  onClick={() => {
                    setCount(n);
                    if (first >= n) setFirst(0);
                  }}
                >
                  <Users size={18} />
                  {n} 人
                </button>
              ))}
            </div>
            <div className="player-inputs">
              {names.slice(0, count).map((name, i) => (
                <div className="player-input" key={i}>
                  <span
                    className="player-dot"
                    style={{ background: PLAYER_COLORS[i] }}
                  >
                    {i + 1}
                  </span>
                  <input
                    aria-label={`玩家 ${i + 1} 名称`}
                    maxLength={20}
                    value={name}
                    onChange={(e) =>
                      setNames(
                        names.map((v, j) => (j === i ? e.target.value : v)),
                      )
                    }
                  />
                  <select
                    aria-label={`玩家 ${i + 1} 类型`}
                    value="human"
                    onChange={() => {}}
                  >
                    <option value="human">人类</option>
                    <option value="ai" disabled>
                      AI · 敬请期待
                    </option>
                  </select>
                </div>
              ))}
            </div>
          </div>
          <div className="setup-field">
            <div className="label-row">
              <label>选择风景</label>
              <small>所有玩家使用同一面</small>
            </div>
            <div className="landscape-choices">
              {(["A", "B"] as Side[]).map((s) => (
                <button
                  key={s}
                  className={cx(side === s && "selected")}
                  onClick={() => setSide(s)}
                >
                  {s === "A" ? <Waves size={27} /> : <Mountain size={27} />}
                  <strong>{s === "A" ? "蜿蜒河流" : "静谧群岛"}</strong>
                  <span>
                    {s} 面 · {s === "A" ? "推荐初次游玩" : "探索岛屿计分"}
                  </span>
                  {side === s && (
                    <CheckCircle2 className="choice-check" size={17} />
                  )}
                </button>
              ))}
            </div>
          </div>
          <button
            className={cx("spirit-toggle", spirits && "enabled")}
            role="switch"
            aria-checked={spirits}
            onClick={() => setSpirits(!spirits)}
          >
            <span className="spirit-icon">
              <Sparkles size={22} />
            </span>
            <span>
              <strong>自然之灵</strong>
              <small>开局二选一，解锁额外计分目标</small>
            </span>
            <span className="switch">
              <i />
            </span>
          </button>
          <div className="first-player">
            <label htmlFor="first-player">先手玩家</label>
            <select
              id="first-player"
              value={first}
              onChange={(e) => setFirst(+e.target.value)}
            >
              {names.slice(0, count).map((n, i) => (
                <option key={i} value={i}>
                  {n || `玩家 ${i + 1}`}
                </option>
              ))}
            </select>
          </div>
          {count === 1 && (
            <div className="solo-note">
              <Sun size={18} />
              <span>3 组地形 · 3 张动物牌 · 回合末刷新供应 · 太阳评级</span>
            </div>
          )}
          <button
            className="primary start-button"
            onClick={() => onStart(names.slice(0, count), side, spirits, first)}
          >
            开始游戏 <ArrowRight size={20} />
          </button>
          {saved && (
            <button className="resume-button" onClick={onResume}>
              <Play size={15} /> 继续上次旅程{" "}
              <span>
                第 {saved.game.round} 轮 · {saved.game.players.length} 人
              </span>
            </button>
          )}
          <p className="setup-bottom">每一片风景，都有生命在等待。</p>
        </section>
      </main>
      <footer className="setup-footer">
        <span>HARMONIES · 本地桌游习作</span>
        <span>
          游戏设计 Johan Benvenuto · 原版插画 Maëva Da Silva / Libellud
        </span>
      </footer>
    </div>
  );
}

function Scores({
  game,
  onClose,
  onNew,
}: {
  game: Game;
  onClose: () => void;
  onNew: () => void;
}) {
  const ranked = ranking(game),
    scores = game.players.map((p) => scorePlayer(p, game.side)),
    maxRows = Math.max(5, ...scores.map((s) => s.cardScores.length));
  return (
    <Modal title="万物和鸣 · 结算记录" onClose={onClose} wide>
      <div className="result-banner">
        <Trophy size={36} />
        <div>
          <span className="eyebrow">A WORLD BEAUTIFULLY MADE</span>
          <h3>
            {game.players.length === 1
              ? `${game.players[0].name} 的自然之旅`
              : ranked
                  .filter((r) => r.rank === 1)
                  .map((r) => r.player.name)
                  .join(" 与 ") + " 获得胜利"}
          </h3>
          <p>
            {game.players.length === 1
              ? `获得 ${soloSuns(game.players[0], game.side).total} 个太阳 · 地形与生命，共同谱成风景。`
              : `共 ${game.round} 轮 · ${game.side} 面${game.side === "A" ? "河流" : "岛屿"} · 同分时比较已入住动物数量`}
          </p>
        </div>
      </div>
      <p className="muted">
        按官方计分纸分列：左列地形，右列逐张动物与自然之灵；分别小计后相加。
      </p>
      <div className="scoresheets">
        {game.players.map((p, i) => {
          const s = scores[i],
            r = ranked.find((r) => r.player.id === p.id)!;
          const terrain = [
            ["✤ 树木", s.trees],
            ["▲ 山脉", s.mountains],
            ["✿ 田野", s.fields],
            ["▥ 建筑", s.buildings],
            ["≈ " + (game.side === "A" ? "河流" : "岛屿"), s.water],
          ];
          return (
            <section className="scoresheet" key={p.id}>
              <header>
                <span>#{r.rank}</span>
                <h3>{p.name}</h3>
                <small>{s.cubes} 只动物</small>
              </header>
              <table>
                <thead>
                  <tr>
                    <th colSpan={2}>地形景观</th>
                    <th colSpan={2}>动物 / 自然之灵</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: maxRows }, (_, j) => (
                    <tr key={j}>
                      <td>{terrain[j]?.[0] ?? ""}</td>
                      <td>{terrain[j]?.[1] ?? ""}</td>
                      <td>
                        {s.cardScores[j] ? CARDS[s.cardScores[j].id].name : "—"}
                      </td>
                      <td>{s.cardScores[j]?.score ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
                <tfoot>
                  <tr>
                    <th>小计</th>
                    <td>{s.landscape}</td>
                    <th>小计</th>
                    <td>{s.animalTotal}</td>
                  </tr>
                </tfoot>
              </table>
              <div className="score-total">
                <span>
                  {s.landscape} + {s.animalTotal} =
                </span>
                <strong>{s.total}</strong>
                <small>分</small>
              </div>
              {game.players.length === 1 && (
                <div className="sun-breakdown">
                  <Sun size={22} />
                  <b>{soloSuns(p, game.side).total}</b>
                  <span>
                    分数 {soloSuns(p, game.side).base} + 版图{" "}
                    {soloSuns(p, game.side).sideBonus} + 自然之灵配置{" "}
                    {soloSuns(p, game.side).spiritBonus}
                  </span>
                </div>
              )}
            </section>
          );
        })}
      </div>
      <div className="modal-actions">
        <button className="secondary" onClick={onClose}>
          回看版图
        </button>
        <button className="primary" onClick={onNew}>
          再启一程 <ArrowRight size={17} />
        </button>
      </div>
    </Modal>
  );
}

export default function App() {
  const [session, setSession] = useState<Session | null>(readSave),
    [screen, setScreen] = useState<"setup" | "game">("setup"),
    [help, setHelp] = useState(false),
    [scores, setScores] = useState(false),
    [selectedToken, setSelectedToken] = useState(0),
    [selectedAnimal, setSelectedAnimal] = useState<number | null>(null),
    [error, setError] = useState(""),
    [cardDetail, setCardDetail] = useState<{
      id: number;
      slot?: number;
    } | null>(null),
    [previewPlayer, setPreviewPlayer] = useState<Player | null>(null),
    [hoverPlayer, setHoverPlayer] = useState<Player | null>(null),
    [endDialog, setEndDialog] = useState(false),
    [discardSlot, setDiscardSlot] = useState<number | undefined>(),
    [handoff, setHandoff] = useState(false),
    [showLog, setShowLog] = useState(false),
    [saveError, setSaveError] = useState(false);
  const game = session?.game,
    player = game?.players[game.active];
  useEffect(() => {
    if (session)
      try {
        localStorage.setItem(SAVE_KEY, JSON.stringify(session));
        setSaveError(false);
      } catch {
        setSaveError(true);
      }
  }, [session]);
  useEffect(() => {
    if (error) {
      const t = setTimeout(() => setError(""), 4500);
      return () => clearTimeout(t);
    }
  }, [error]);
  useEffect(() => {
    if (
      selectedAnimal &&
      player?.cards.find((c) => c.id === selectedAnimal && isComplete(c))
    )
      setSelectedAnimal(null);
  }, [player, selectedAnimal]);
  function act(action: Action) {
    if (!session) return;
    try {
      const next = dispatch(session, action);
      setSession(next);
      setError("");
      if (action.type === "place")
        setSelectedToken(
          Math.min(
            selectedToken,
            Math.max(0, next.game.turn.pending.length - 1),
          ),
        );
      if (action.type === "endTurn") {
        setSelectedAnimal(null);
        setSelectedToken(0);
        setEndDialog(false);
        setDiscardSlot(undefined);
        if (next.game.phase === "playing" && next.game.players.length > 1)
          setHandoff(true);
      }
    } catch (e) {
      setError((e as Error).message);
    }
  }
  function start(names: string[], side: Side, spirits: boolean, first: number) {
    setSession({
      game: createGame({ names, side, spirits, first }),
      history: [],
    });
    setScreen("game");
    setSelectedAnimal(null);
    setSelectedToken(0);
    setScores(false);
    setHandoff(false);
    setError("");
  }
  function undoAction(all = false) {
    if (session) {
      setSession(undo(session, all));
      setSelectedAnimal(null);
      setSelectedToken(0);
      setError("");
    }
  }
  function clickCell(index: number) {
    if (!game || game.phase === "ended") return;
    if (selectedAnimal)
      act({ type: "settle", card: selectedAnimal, cell: index });
    else if (game.turn.pending.length)
      act({ type: "place", pending: selectedToken, cell: index });
    else setError("先从供应板选择一组地形片，或选择动物牌以安排入住。");
  }
  const detailCard = cardDetail ? CARDS[cardDetail.id] : null;
  if (screen === "setup" || !game || !player || !session)
    return (
      <>
        <SetupScreen
          saved={session}
          onStart={start}
          onResume={() => {
            setScreen("game");
            setHandoff(false);
          }}
          onHelp={() => setHelp(true)}
        />
        {help && (
          <Help side={game?.side ?? "A"} onClose={() => setHelp(false)} />
        )}
      </>
    );
  const currentScore = scorePlayer(player, game.side),
    owned = activeCards(player),
    completed = player.cards.filter(isComplete),
    pendingColor = game.turn.pending[selectedToken],
    reason = endingReason(game),
    ended = game.phase === "ended";
  const steps = [
    game.turn.tookTokens,
    game.turn.tookTokens && !game.turn.pending.length,
    game.turn.tookCard,
  ];
  return (
    <div className="game-page">
      <header className="game-header">
        <Brand />
        <div className="game-meta">
          <span className="round-pill">
            第 <b>{game.round.toString().padStart(2, "0")}</b> 轮
          </span>
          <span>
            {game.players.length === 1
              ? "单人旅行"
              : `${game.players.length} 人同行`}
          </span>
          <span>
            {game.side} 面 · {game.side === "A" ? "蜿蜒河流" : "静谧群岛"}
          </span>
          {game.spirits && (
            <span>
              <Sparkles size={14} /> 自然之灵
            </span>
          )}
        </div>
        <div className="header-actions">
          <button
            className="icon-button"
            title="操作记录"
            aria-label="操作记录"
            onClick={() => setShowLog(true)}
          >
            <ScrollText size={19} />
          </button>
          <button
            className="icon-button"
            title="规则手册"
            aria-label="规则手册"
            onClick={() => setHelp(true)}
          >
            <CircleHelp size={19} />
          </button>
          <button className="text-button" onClick={() => setScreen("setup")}>
            <Pause size={16} /> 保存并离开
          </button>
        </div>
      </header>
      {saveError && (
        <div className="final-banner">
          浏览器存储不可用，请保持页面打开以继续本局。
        </div>
      )}
      {(reason || ended) && (
        <div
          className={cx("final-banner", ended && "ended-banner")}
          role="status"
        >
          <Flag size={19} />
          <strong>{ended ? "游戏结束" : "最后一轮"}</strong>
          <span>
            {ended
              ? "风景已经完成，来看看这段旅程的收获。"
              : `${reason}。本轮结束后结算，所有玩家回合数相同。`}
          </span>
          {ended && (
            <button className="primary" onClick={() => setScores(true)}>
              查看结算 <Trophy size={17} />
            </button>
          )}
        </div>
      )}
      <main className="game-content">
        <section className="market-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">WILDLIFE RESERVE</span>
              <h2>等待相遇的生命</h2>
            </div>
            <p>
              <PawPrint size={16} /> 每回合可邀请 1 种动物 <span>·</span>{" "}
              牌库剩余 {game.deck.length} 张
            </p>
            <span className="status-tag">
              {game.turn.tookCard ? "本回合已拿牌" : "点击卡牌查看栖息地"}
            </span>
          </div>
          <div className="market-row">
            {game.market.map((id, i) =>
              id ? (
                <CardFace
                  key={i}
                  id={id}
                  className={cx(game.turn.tookCard && "unavailable")}
                  onClick={() => setCardDetail({ id, slot: i })}
                />
              ) : (
                <div key={i} className="empty-market">
                  <Leaf />
                  <span>回合结束后补充</span>
                </div>
              ),
            )}
            <div className="market-note">
              <Leaf size={24} />
              <p>
                让栖息地彼此交织，
                <br />
                让每片风景都有生命。
              </p>
              <small>
                提示：选择落点颜色不同的动物，
                <br />
                更容易充分利用你的版图。
              </small>
            </div>
          </div>
        </section>
        <div className="table-layout">
          <aside className="left-column">
            <section className="panel supply-panel">
              <div className="section-heading">
                <h2>地形供应</h2>
                <span className="tiny-label">袋中 {game.bag.length} 枚</span>
              </div>
              <p className="muted small-text">
                选择一整组，将 3 枚全部放入版图。
              </p>
              <div className="supply-groups">
                {game.supply.map((group, i) => (
                  <button
                    key={i}
                    className={cx("supply-group", !group.length && "taken")}
                    disabled={
                      game.turn.tookTokens ||
                      ended ||
                      group.length !== 3 ||
                      !!player.spiritChoices.length
                    }
                    onClick={() => {
                      act({ type: "takeTokens", group: i });
                      setSelectedAnimal(null);
                      setSelectedToken(0);
                    }}
                    aria-label={`拿取第 ${i + 1} 组地形${group.length ? "：" + group.map((c) => COLORS[c].name).join("、") : ""}`}
                  >
                    <span className="group-number">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    {group.length ? (
                      group.map((c, j) => <Token color={c} key={j} />)
                    ) : (
                      <span className="group-empty">
                        <Check size={17} /> 已拿取
                      </span>
                    )}
                    <ChevronRight className="supply-arrow" size={15} />
                  </button>
                ))}
              </div>
              {game.players.length === 1 && (
                <div className="supply-solo">
                  <Sun size={15} />
                  <span>回合结束，剩余地形移出游戏。</span>
                </div>
              )}
            </section>
            <section className="panel turn-panel">
              <span className="eyebrow">YOUR TURN</span>
              <h3>慢慢来，风景在生长</h3>
              {["拿取一组地形", "放置全部 3 枚", "邀请 1 种动物（可选）"].map(
                (s, i) => (
                  <div className={cx("turn-step", steps[i] && "done")} key={s}>
                    <span>{steps[i] ? <Check size={12} /> : i + 1}</span>
                    {s}
                  </div>
                ),
              )}
              <div className="turn-step">
                <span>
                  <PawPrint size={12} />
                </span>
                动物入住（可选，不限次数）
              </div>
              <p>拿牌、放地形、动物入住，可以穿插进行。</p>
            </section>
          </aside>
          <section className="personal-area">
            <div className="personal-heading">
              <div>
                <span
                  className="player-dot"
                  style={{ background: PLAYER_COLORS[player.id] }}
                >
                  {player.id + 1}
                </span>
                <h2>{player.name} 的自然栖所</h2>
                {!ended && <span className="status-tag green">当前回合</span>}
              </div>
              <span className="tiny-label">
                {emptyCount(player)} / {player.board.length} 空格
              </span>
            </div>
            <div className="personal-grid">
              <div className="landscape-workspace">
                <div className="board-wrap">
                  <div className="board-corner">
                    <span>{game.side}</span>
                    <small>{game.side === "A" ? "RIVER" : "ISLANDS"}</small>
                  </div>
                  <Board
                    player={player}
                    side={game.side}
                    selectedColor={
                      !ended && !selectedAnimal ? pendingColor : undefined
                    }
                    selectedAnimal={!ended ? selectedAnimal : null}
                    onCell={clickCell}
                  />
                  <div className="board-caption">
                    {selectedAnimal
                      ? `正在安排 ${CARDS[selectedAnimal].name} 入住 · 点击高亮落点`
                      : pendingColor
                        ? `正在放置${COLORS[pendingColor].name} · 点击带绿点的格子`
                        : "一片地形，一处栖所，一段自然的故事。"}
                  </div>
                </div>
                <div className="placement-bar">
                  <div className="tray-label">
                    <span className="eyebrow">IN YOUR HAND</span>
                    <b>
                      {selectedAnimal
                        ? "动物入住"
                        : game.turn.pending.length
                          ? "待放置地形"
                          : "本回合地形"}
                    </b>
                  </div>
                  {selectedAnimal ? (
                    <div className="settlement-status">
                      <PawPrint size={20} />
                      <span>
                        {CARDS[selectedAnimal].name} ·{" "}
                        {matches(player.board, selectedAnimal).length}{" "}
                        个可用落点
                      </span>
                      <button
                        className="text-button"
                        onClick={() => setSelectedAnimal(null)}
                      >
                        返回放地形
                      </button>
                    </div>
                  ) : game.turn.pending.length ? (
                    <div className="pending-tokens">
                      {game.turn.pending.map((c, i) => (
                        <button
                          key={i}
                          className={cx(selectedToken === i && "selected")}
                          onClick={() => setSelectedToken(i)}
                          aria-label={`选择待放置的${COLORS[c].name} ${i + 1}`}
                        >
                          <Token color={c} />
                          <span>{COLORS[c].name}</span>
                        </button>
                      ))}
                    </div>
                  ) : (
                    <span className="tray-empty">
                      {game.turn.tookTokens ? (
                        <>
                          <CheckCircle2 size={18} /> 地形已全部放置
                        </>
                      ) : (
                        "从左侧供应板，选一组心仪的地形。"
                      )}
                    </span>
                  )}
                </div>
              </div>
              <div className="habitat-workspace">
                <div className="my-cards-heading">
                  <h3>
                    <PawPrint size={17} /> 我的动物{" "}
                    <span>{owned.length} / 4 未完成</span>
                  </h3>
                  <small>选择卡牌，查看入住落点</small>
                </div>
                <div className="owned-cards">
                  {owned.map((c) => (
                    <div className="owned-card-wrap" key={c.id}>
                      <CardFace
                        id={c.id}
                        owned={c}
                        className={cx(selectedAnimal === c.id && "active-card")}
                        onClick={() => {
                          setSelectedAnimal(
                            selectedAnimal === c.id ? null : c.id,
                          );
                        }}
                      />
                      <button
                        className="card-info-button"
                        aria-label={`查看${CARDS[c.id].name}详情`}
                        onClick={() => setCardDetail({ id: c.id })}
                      >
                        <Maximize2 size={13} />
                      </button>
                      <div
                        className={cx(
                          "card-progress",
                          matches(player.board, c.id).length > 0 &&
                            "can-settle",
                        )}
                      >
                        {matches(player.board, c.id).length > 0 ? (
                          <>
                            <span /> 可入住
                          </>
                        ) : (
                          <>
                            {c.placed} / {CARDS[c.id].points.length} 已入住
                          </>
                        )}
                      </div>
                    </div>
                  ))}
                  {Array.from(
                    { length: Math.max(0, 4 - owned.length) },
                    (_, i) => (
                      <div className="empty-card-slot" key={i}>
                        <PawPrint size={24} />
                        <span>{i === 0 ? "等待新朋友" : "空闲位置"}</span>
                      </div>
                    ),
                  )}
                </div>
                {selectedAnimal && (
                  <div className="selected-pattern">
                    <Pattern card={CARDS[selectedAnimal]} />
                    <div>
                      <b>{CARDS[selectedAnimal].name}</b>
                      <p>
                        {CARDS[selectedAnimal].description ||
                          `每次入住依次解锁 ${CARDS[selectedAnimal].points.join(" / ")} 分，最终只计最高已解锁值。`}
                      </p>
                      <small>
                        图案可以旋转；白色菱形为动物落点；建筑底层可为红 / 灰 /
                        棕。
                      </small>
                    </div>
                  </div>
                )}
                {completed.length > 0 && (
                  <details className="completed-cards">
                    <summary>
                      <CheckCircle2 size={15} /> 已完成 {completed.length} 张 ·
                      得分保留，不占牌位
                    </summary>
                    <div>
                      {completed.map((c) => (
                        <CardFace
                          key={c.id}
                          id={c.id}
                          owned={c}
                          onClick={() => setCardDetail({ id: c.id })}
                        />
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
          </section>
          <aside className="right-column">
            <section className="panel score-panel">
              <div className="section-heading">
                <h3>风景的收获</h3>
                <Leaf size={18} />
              </div>
              <div className="live-score">
                <strong>{currentScore.total}</strong>
                <span>当前预估分数</span>
              </div>
              <div className="score-split">
                <span>
                  地形 <b>{currentScore.landscape}</b>
                </span>
                <span>
                  动物与灵 <b>{currentScore.animalTotal}</b>
                </span>
              </div>
              <p>会随景观改变 · 最终以结算为准</p>
            </section>
            {game.players.length > 1 && (
              <section className="opponents">
                <div className="section-heading">
                  <h3>同行的风景</h3>
                  <Eye size={16} />
                </div>
                <p className="muted small-text">悬停放大 · 点击固定查看</p>
                {game.players
                  .filter((p) => p.id !== player.id)
                  .map((p) => (
                    <button
                      key={p.id}
                      className="opponent"
                      onMouseEnter={() => setHoverPlayer(p)}
                      onMouseLeave={() => setHoverPlayer(null)}
                      onFocus={() => setHoverPlayer(p)}
                      onBlur={() => setHoverPlayer(null)}
                      onClick={() => {
                        setHoverPlayer(null);
                        setPreviewPlayer(p);
                      }}
                    >
                      <span className="opponent-heading">
                        <i style={{ background: PLAYER_COLORS[p.id] }} />
                        <b>{p.name}</b>
                        <span>{scorePlayer(p, game.side).total} 分</span>
                      </span>
                      <Board player={p} side={game.side} mini />
                      <span className="opponent-foot">
                        {emptyCount(p)} 个空格 <span>{p.turns} 回合</span>
                      </span>
                    </button>
                  ))}
              </section>
            )}
            {game.players.length === 1 && (
              <section className="panel solo-target">
                <Sun size={25} />
                <h3>收集你的太阳</h3>
                <p>每一次旅行，都可以比上次走得更远。</p>
                <div>
                  <b>{soloSuns(player, game.side).total}</b>
                  <span>当前太阳评级</span>
                </div>
                <small>
                  分数门槛：40 · 70 · 90 · 110
                  <br />
                  130 · 140 · 150 · 160
                </small>
              </section>
            )}
            <button className="guide-link" onClick={() => setHelp(true)}>
              <BookOpen size={17} />
              <span>翻阅自然旅行手册</span>
              <ChevronRight size={16} />
            </button>
          </aside>
        </div>
        <section className="reference-panel">
          <div className="section-heading">
            <h3>地形计分提示</h3>
            <span>把美好的风景，变成分数。</span>
          </div>
          <ScoreGuide side={game.side} />
        </section>
      </main>
      <footer className="action-bar">
        <div className="action-player">
          <span
            className="player-dot"
            style={{ background: PLAYER_COLORS[player.id] }}
          >
            {player.id + 1}
          </span>
          <div>
            <b>{ended ? "旅程已完成" : `轮到 ${player.name}`}</b>
            <small>
              {ended
                ? "感谢你创造的这片风景"
                : game.turn.pending.length
                  ? `还需放置 ${game.turn.pending.length} 枚地形`
                  : game.turn.tookTokens
                    ? "可以继续拿牌、入住，或结束回合"
                    : "请选择一组地形片"}
            </small>
          </div>
        </div>
        <div className="undo-actions">
          <button
            className="text-button"
            disabled={!session.history.length || ended}
            onClick={() => undoAction()}
          >
            <Undo2 size={17} /> 撤销上一步
          </button>
          <button
            className="text-button"
            disabled={!session.history.length || ended}
            onClick={() => undoAction(true)}
          >
            <RotateCcw size={16} /> 重置本回合
          </button>
        </div>
        <button
          className="primary end-turn"
          disabled={!ended && !canEndTurn(game)}
          onClick={() => {
            if (ended) setScores(true);
            else {
              setEndDialog(true);
              setDiscardSlot(undefined);
            }
          }}
        >
          {ended ? "查看结算" : "结束回合"} <ArrowRight size={18} />
        </button>
      </footer>
      {error && (
        <div className="toast" role="alert">
          {error}
          <button
            className="icon-button"
            onClick={() => setError("")}
            aria-label="关闭提示"
          >
            <X size={16} />
          </button>
        </div>
      )}
      {hoverPlayer && (
        <div className="hover-board">
          <div className="section-heading">
            <h3>{hoverPlayer.name} 的版图</h3>
            <span>{scorePlayer(hoverPlayer, game.side).total} 分</span>
          </div>
          <Board player={hoverPlayer} side={game.side} />
        </div>
      )}
      {previewPlayer && (
        <Modal
          title={`${previewPlayer.name} 的自然栖所`}
          onClose={() => setPreviewPlayer(null)}
        >
          <Board player={previewPlayer} side={game.side} />
          <div className="preview-card-list">
            {previewPlayer.cards.map((c) => (
              <CardFace key={c.id} id={c.id} owned={c} />
            ))}
          </div>
          <p className="muted">
            地形 {scorePlayer(previewPlayer, game.side).landscape} +
            动物与自然之灵 {scorePlayer(previewPlayer, game.side).animalTotal} ={" "}
            {scorePlayer(previewPlayer, game.side).total} 分
          </p>
        </Modal>
      )}
      {detailCard && cardDetail && (
        <Modal title={detailCard.name} onClose={() => setCardDetail(null)}>
          <div className="card-detail">
            <CardFace
              id={detailCard.id}
              owned={player.cards.find((c) => c.id === detailCard.id)}
            />
            <div>
              <span className="eyebrow">
                {detailCard.spirit ? "NATURE’S SPIRIT" : "HABITAT"}
              </span>
              <h3>{detailCard.spirit ? "自然之灵的祝福" : "为它创造栖息地"}</h3>
              <Pattern card={detailCard} />
              <p>
                {detailCard.description ||
                  `入住 ${detailCard.points.length} 次可完成此卡。依次获得 ${detailCard.points.join(" / ")} 分，只计最高已解锁值。`}
              </p>
              <small>
                白色菱形：入住落点
                <br />
                图案可旋转，不可镜像
                <br />
                树木与山脉的高度须完全相同
                <br />
                建筑须两层，底层可为红、灰或棕
              </small>
            </div>
          </div>
          <div className="modal-actions">
            <button className="secondary" onClick={() => setCardDetail(null)}>
              返回
            </button>
            {cardDetail.slot !== undefined && (
              <button
                className="primary"
                disabled={
                  ended ||
                  game.turn.tookCard ||
                  owned.length >= 4 ||
                  !!player.spiritChoices.length
                }
                onClick={() => {
                  act({ type: "takeCard", slot: cardDetail.slot! });
                  setCardDetail(null);
                }}
              >
                邀请这位朋友 <ArrowRight size={17} />
              </button>
            )}
            {cardDetail.slot === undefined &&
              !ended &&
              player.cards.some(
                (c) => c.id === detailCard.id && !isComplete(c),
              ) && (
                <button
                  className="primary"
                  onClick={() => {
                    setSelectedAnimal(detailCard.id);
                    setCardDetail(null);
                  }}
                >
                  查看入住落点
                </button>
              )}
          </div>
          {cardDetail.slot !== undefined &&
            (owned.length >= 4 || game.turn.tookCard) && (
              <p className="muted">
                {game.turn.tookCard
                  ? "本回合已经拿过动物牌。"
                  : "未完成的卡牌已满 4 张，先完成一张再邀请新朋友。"}
              </p>
            )}
        </Modal>
      )}
      {!!player.spiritChoices.length && !handoff && (
        <Modal title={`${player.name}，选择你的自然之灵`}>
          <p className="muted">
            选择 1
            张，另一张移出游戏。先完成栖息地并放置自然之灵，才能获得它的额外分数。
          </p>
          <div className="spirit-choices">
            {player.spiritChoices.map((id) => (
              <div key={id}>
                <CardFace id={id} />
                <h3>{CARDS[id].name}</h3>
                <p>{CARDS[id].description}</p>
                <button
                  className="primary"
                  onClick={() => act({ type: "chooseSpirit", card: id })}
                >
                  选择这位自然之灵
                </button>
              </div>
            ))}
          </div>
        </Modal>
      )}
      {handoff && !ended && (
        <Modal title="把风景交给下一位旅人">
          <div className="handoff-content">
            <span
              className="player-dot"
              style={{ background: PLAYER_COLORS[player.id] }}
            >
              {player.id + 1}
            </span>
            <h3>轮到 {player.name}</h3>
            <p>
              第 {game.round} 轮{reason ? " · 最后一轮" : ""}
            </p>
            <button className="primary" onClick={() => setHandoff(false)}>
              我准备好了 <ArrowRight size={18} />
            </button>
          </div>
        </Modal>
      )}
      {endDialog && (
        <Modal
          title={reason ? "完成你的最后回合" : "结束当前回合"}
          onClose={() => setEndDialog(false)}
        >
          <p>供应将在回合结束后补充，之后不能撤销本回合。</p>
          {owned.some((c) => matches(player.board, c.id).length) && (
            <div className="inline-notice">
              <PawPrint size={18} />{" "}
              你还有动物可以入住。可返回继续操作，也可以直接结束。
            </div>
          )}
          {game.players.length === 1 &&
            !game.turn.tookCard &&
            game.deck.length > 0 && (
              <>
                <h3>单人可选操作：替换 1 张公共动物牌</h3>
                <p className="muted">
                  本回合未拿牌，可以选择弃掉一张并补入新牌，也可以全部保留。
                </p>
                <div className="solo-discard-cards">
                  {game.market.map(
                    (id, i) =>
                      id && (
                        <CardFace
                          key={i}
                          id={id}
                          className={cx(discardSlot === i && "active-card")}
                          onClick={() =>
                            setDiscardSlot(discardSlot === i ? undefined : i)
                          }
                        />
                      ),
                  )}
                </div>
                <p className="tiny-label">
                  {discardSlot === undefined
                    ? "当前选择：保留全部动物牌"
                    : `将替换：${CARDS[game.market[discardSlot]!].name}`}
                </p>
              </>
            )}
          {reason && (
            <div className="inline-notice">
              <Flag size={18} />
              {reason}。
              {game.active === game.players.length - 1
                ? "本回合结束即结束游戏。"
                : "其余玩家完成本轮后结束游戏。"}
            </div>
          )}
          <div className="modal-actions">
            <button className="secondary" onClick={() => setEndDialog(false)}>
              继续本回合
            </button>
            <button
              className="primary"
              onClick={() => act({ type: "endTurn", discardSlot })}
            >
              确认结束 <Check size={17} />
            </button>
          </div>
        </Modal>
      )}
      {help && <Help side={game.side} onClose={() => setHelp(false)} />}{" "}
      {scores && (
        <Scores
          game={game}
          onClose={() => setScores(false)}
          onNew={() => {
            setScores(false);
            setScreen("setup");
          }}
        />
      )}
      {showLog && (
        <Modal title="这段旅程的足迹" onClose={() => setShowLog(false)}>
          <ol className="log-list">
            {game.log.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ol>
        </Modal>
      )}
    </div>
  );
}
