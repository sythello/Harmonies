import rawCards from "./cards.json" with { type: "json" };

export type Color = 1 | 2 | 3 | 4 | 5 | 6;
export type Side = "A" | "B";
export interface Coord {
  col: number;
  row: number;
}
export interface Cell extends Coord {
  tokens: Color[];
  animal: number | null;
}
export interface Card {
  id: number;
  name: string;
  points: number[];
  spirit: boolean;
  description?: string;
  pattern: { colors: number[]; position: number; allowCube: boolean }[];
}
export interface OwnedCard {
  id: number;
  placed: number;
}
export interface Player {
  id: number;
  name: string;
  type: "human";
  board: Cell[];
  cards: OwnedCard[];
  spiritChoices: number[];
  turns: number;
}
export interface Setup {
  names: string[];
  side: Side;
  spirits: boolean;
  first: number;
  seed?: number;
}
export interface Game {
  version: 1;
  players: Player[];
  active: number;
  round: number;
  side: Side;
  spirits: boolean;
  bag: Color[];
  supply: Color[][];
  deck: number[];
  market: (number | null)[];
  discardedTokens: Color[];
  discardedCards: number[];
  turn: { tookTokens: boolean; tookCard: boolean; pending: Color[] };
  finalRound: number | null;
  finalReason: string;
  phase: "playing" | "ended";
  log: string[];
}
export type Action =
  | { type: "takeTokens"; group: number }
  | { type: "place"; pending: number; cell: number }
  | { type: "takeCard"; slot: number }
  | { type: "settle"; card: number; cell: number }
  | { type: "chooseSpirit"; card: number }
  | { type: "endTurn"; discardSlot?: number };
export interface Session {
  game: Game;
  history: Game[];
}
export const COLORS: Record<
  Color,
  { name: string; hex: string; symbol: string }
> = {
  1: { name: "水域", hex: "#398c9a", symbol: "≈" },
  2: { name: "岩石", hex: "#85918e", symbol: "▲" },
  3: { name: "树干", hex: "#987050", symbol: "◎" },
  4: { name: "树冠", hex: "#91aa53", symbol: "✤" },
  5: { name: "田野", hex: "#e5b24a", symbol: "✿" },
  6: { name: "建筑", hex: "#bd6658", symbol: "▥" },
};
const NAMES = [
  "鳄鱼",
  "鳐鱼",
  "蝾螈",
  "水獭",
  "青蛙",
  "野鸭",
  "火烈鸟",
  "壁虎",
  "鼩鼱",
  "孔雀",
  "松鼠",
  "刺猬",
  "熊蜂",
  "棕熊",
  "野兔",
  "鹦鹉",
  "野猪",
  "考拉",
  "灰狼",
  "翠鸟",
  "企鹅",
  "蝙蝠",
  "耳廓狐",
  "狒狒",
  "猛禽",
  "狐獴",
  "渡鸦",
  "羊驼",
  "北极狐",
  "浣熊",
  "瓢虫",
  "黑豹",
  "狮子之灵",
  "蝴蝶之灵",
  "鹿之灵",
  "猫头鹰之灵",
  "猫之灵",
  "鹳之灵",
  "盘羊之灵",
  "旱獭之灵",
  "蜻蜓之灵",
  "龟之灵",
];
const SPIRIT_TEXT = [
  "每组田野：1–2 格得 2 分，3 格及以上得 10 分。",
  "每组田野得 5 分，单独一格也算一组。",
  "每棵 2 层或 3 层树额外得 4 分。",
  "每棵 1 / 2 / 3 层树额外得 3 / 3 / 1 分。",
  "每组完整建筑得 4 分，单独一栋也算一组。",
  "每组至少 2 栋相连的完整建筑得 6 分。",
  "每座 2 层或 3 层山额外得 4 分，包括孤山。",
  "每座 1 / 2 / 3 层山额外得 3 / 3 / 1 分，包括孤山。",
  "每组至少 2 格相连水域得 7 分。",
  "每格水域额外得 2 分。",
];
export const CARDS: Record<number, Card> = Object.fromEntries(
  Object.values(rawCards).map((c) => [
    c.type_arg,
    {
      id: c.type_arg,
      name: NAMES[c.type_arg - 1],
      points: c.pointLocations,
      pattern: c.pattern,
      spirit: c.isSpirit,
      description: c.isSpirit ? SPIRIT_TEXT[c.type_arg - 33] : undefined,
    },
  ]),
);
export const clone = <T>(value: T): T => structuredClone(value);
export const key = (c: Coord) => `${c.col},${c.row}`;
export function neighbor(c: Coord, direction: number): Coord {
  const even = c.col % 2 === 0;
  const dirs = even
    ? [
        [1, 0],
        [1, -1],
        [0, -1],
        [-1, -1],
        [-1, 0],
        [0, 1],
      ]
    : [
        [1, 1],
        [1, 0],
        [0, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
      ];
  const [dc, dr] = dirs[direction % 6];
  return { col: c.col + dc, row: c.row + dr };
}
export const neighbors = (c: Coord) =>
  Array.from({ length: 6 }, (_, i) => neighbor(c, i));
export function emptyBoard(side: Side): Cell[] {
  return Array.from({ length: side === "A" ? 5 : 7 }, (_, col) =>
    Array.from({ length: side === "A" ? (col % 2 ? 4 : 5) : 4 }, (_, row) => ({
      col,
      row,
      tokens: [],
      animal: null,
    })),
  ).flat();
}
export function seededRandom(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function shuffle<T>(arr: T[], rng: () => number): T[] {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}
export function createGame(setup: Setup): Game {
  if (
    setup.names.length < 1 ||
    setup.names.length > 4 ||
    !["A", "B"].includes(setup.side) ||
    setup.first < 0 ||
    setup.first >= setup.names.length
  )
    throw new Error("请设置 1–4 名玩家并选择有效版图与先手。");
  const rng = setup.seed === undefined ? Math.random : seededRandom(setup.seed);
  const bag = shuffle(
    [23, 23, 21, 19, 19, 15].flatMap((n, i) =>
      Array<Color>(n).fill((i + 1) as Color),
    ),
    rng,
  );
  const deck = shuffle(
      Array.from({ length: 32 }, (_, i) => i + 1),
      rng,
    ),
    spirits = shuffle(
      Array.from({ length: 10 }, (_, i) => i + 33),
      rng,
    );
  const ordered = setup.names.map((name, id) => ({
    id,
    name: name.trim().slice(0, 20) || `玩家 ${id + 1}`,
  }));
  ordered.push(...ordered.splice(0, setup.first));
  const players: Player[] = ordered.map((p) => ({
    ...p,
    type: "human",
    board: emptyBoard(setup.side),
    cards: [],
    spiritChoices: setup.spirits ? spirits.splice(0, 2) : [],
    turns: 0,
  }));
  const size = players.length === 1 ? 3 : 5;
  return {
    version: 1,
    players,
    active: 0,
    round: 1,
    side: setup.side,
    spirits: setup.spirits,
    bag,
    supply: Array.from({ length: size }, () => bag.splice(0, 3)),
    deck,
    market: deck.splice(0, size),
    discardedTokens: [],
    discardedCards: [],
    turn: { tookTokens: false, tookCard: false, pending: [] },
    finalRound: null,
    finalReason: "",
    phase: "playing",
    log: ["新的自然旅程开始了。"],
  };
}
export const top = (c: Cell) => c.tokens.at(-1);
export const emptyCount = (p: Player) =>
  p.board.filter((c) => !c.tokens.length).length;
export const isComplete = (c: OwnedCard) =>
  c.placed === CARDS[c.id].points.length;
export const activeCards = (p: Player) => p.cards.filter((c) => !isComplete(c));
export function canPlace(cell: Cell, color: Color): boolean {
  if (cell.animal !== null) return false;
  const t = cell.tokens;
  if (!t.length) return true;
  if (color === 2) return t.every((c) => c === 2) && t.length < 3;
  if (color === 3) return t.length === 1 && t[0] === 3;
  if (color === 4) return t.every((c) => c === 3) && t.length <= 2;
  if (color === 6) return t.length === 1 && [2, 3, 6].includes(t[0]);
  return false;
}
export function patternCoordinates(
  card: Card,
  origin: Coord,
  rotation: number,
): Coord[] {
  let current = origin;
  return card.pattern.map((step, i) => {
    if (i) current = neighbor(current, (step.position + rotation) % 6);
    return current;
  });
}
export function stackMatches(cell: Cell, expected: number[]): boolean {
  if (expected[0] === 6 && expected[1] === 7)
    return top(cell) === 6 && cell.tokens.length === 2;
  return (
    cell.tokens.length === expected.length &&
    expected.every((v, i) => cell.tokens[cell.tokens.length - 1 - i] === v)
  );
}
export interface Match {
  target: number;
  cells: number[];
  rotation: number;
}
export function matches(board: Cell[], cardId: number): Match[] {
  const card = CARDS[cardId],
    lookup = new Map(board.map((c, i) => [key(c), i])),
    found: Match[] = [];
  for (const origin of board)
    for (let rotation = 0; rotation < 6; rotation++) {
      const indices = patternCoordinates(card, origin, rotation).map(
        (c) => lookup.get(key(c)) ?? -1,
      );
      if (
        indices.every(
          (index, i) =>
            index >= 0 && stackMatches(board[index], card.pattern[i].colors),
        )
      ) {
        const target = indices[card.pattern.findIndex((s) => s.allowCube)];
        if (
          board[target].animal === null &&
          !found.some((m) => m.target === target)
        )
          found.push({ target, cells: indices, rotation });
      }
    }
  return found;
}
export function groups(
  board: Cell[],
  predicate: (c: Cell) => boolean,
): Cell[][] {
  const left = new Map(board.filter(predicate).map((c) => [key(c), c])),
    result: Cell[][] = [];
  while (left.size) {
    const first = left.values().next().value!;
    const group: Cell[] = [];
    const queue = [first];
    left.delete(key(first));
    for (let i = 0; i < queue.length; i++) {
      const c = queue[i];
      group.push(c);
      for (const n of neighbors(c)) {
        const next = left.get(key(n));
        if (next) {
          left.delete(key(n));
          queue.push(next);
        }
      }
    }
    result.push(group);
  }
  return result;
}
export function riverLength(board: Cell[]): number {
  const water = new Map(
    board.filter((c) => top(c) === 1).map((c) => [key(c), c]),
  );
  let longest = 0;
  for (const start of water.values()) {
    const distances = new Map([[key(start), 1]]),
      queue = [start];
    for (let i = 0; i < queue.length; i++) {
      const cell = queue[i],
        distance = distances.get(key(cell))!;
      longest = Math.max(longest, distance);
      for (const n of neighbors(cell)) {
        const k = key(n);
        if (water.has(k) && !distances.has(k)) {
          distances.set(k, distance + 1);
          queue.push(water.get(k)!);
        }
      }
    }
  }
  return longest;
}
export function spiritScore(board: Cell[], id: number): number {
  const trees = board.filter((c) => top(c) === 4),
    mountains = board.filter((c) => top(c) === 2);
  const fieldGroups = () => groups(board, (c) => top(c) === 5),
    buildings = () =>
      groups(board, (c) => top(c) === 6 && c.tokens.length === 2);
  switch (id) {
    case 33:
      return fieldGroups().reduce((s, g) => s + (g.length >= 3 ? 10 : 2), 0);
    case 34:
      return fieldGroups().length * 5;
    case 35:
      return trees.filter((c) => c.tokens.length >= 2).length * 4;
    case 36:
      return trees.reduce((s, c) => s + (c.tokens.length === 3 ? 1 : 3), 0);
    case 37:
      return buildings().length * 4;
    case 38:
      return buildings().filter((g) => g.length >= 2).length * 6;
    case 39:
      return mountains.filter((c) => c.tokens.length >= 2).length * 4;
    case 40:
      return mountains.reduce((s, c) => s + (c.tokens.length === 3 ? 1 : 3), 0);
    case 41:
      return (
        groups(board, (c) => top(c) === 1).filter((g) => g.length >= 2).length *
        7
      );
    case 42:
      return board.filter((c) => top(c) === 1).length * 2;
    default:
      return 0;
  }
}
export interface Score {
  trees: number;
  mountains: number;
  fields: number;
  buildings: number;
  water: number;
  landscape: number;
  animals: number;
  spirit: number;
  animalTotal: number;
  total: number;
  cubes: number;
  river: number;
  islands: number;
  cardScores: { id: number; score: number; placed: number }[];
}
export function scorePlayer(player: Player, side: Side): Score {
  const board = player.board,
    lookup = new Map(board.map((c) => [key(c), c]));
  const height = [0, 1, 3, 7];
  const trees = board
    .filter((c) => top(c) === 4)
    .reduce((s, c) => s + height[c.tokens.length], 0);
  const mountains = board
    .filter(
      (c) =>
        top(c) === 2 &&
        neighbors(c).some(
          (n) =>
            top(lookup.get(key(n)) ?? { ...n, tokens: [], animal: null }) === 2,
        ),
    )
    .reduce((s, c) => s + height[c.tokens.length], 0);
  const fields =
    groups(board, (c) => top(c) === 5).filter((g) => g.length >= 2).length * 5;
  const buildings =
    board.filter(
      (c) =>
        top(c) === 6 &&
        c.tokens.length === 2 &&
        new Set(
          neighbors(c)
            .map((n) => lookup.get(key(n)))
            .filter((n): n is Cell => !!n && n.tokens.length > 0)
            .map(top),
        ).size >= 3,
    ).length * 5;
  const river = riverLength(board),
    islands = Math.max(1, groups(board, (c) => top(c) !== 1).length);
  const water =
    side === "B"
      ? islands * 5
      : river > 6
        ? 15 + (river - 6) * 4
        : [0, 0, 2, 5, 8, 11, 15][river];
  const cardScores = player.cards.map((c) => ({
    id: c.id,
    placed: c.placed,
    score:
      c.placed === 0
        ? 0
        : CARDS[c.id].spirit
          ? spiritScore(board, c.id)
          : CARDS[c.id].points[c.placed - 1],
  }));
  const animals = cardScores
      .filter((c) => !CARDS[c.id].spirit)
      .reduce((s, c) => s + c.score, 0),
    spirit = cardScores
      .filter((c) => CARDS[c.id].spirit)
      .reduce((s, c) => s + c.score, 0);
  const landscape = trees + mountains + fields + buildings + water,
    animalTotal = animals + spirit;
  return {
    trees,
    mountains,
    fields,
    buildings,
    water,
    landscape,
    animals,
    spirit,
    animalTotal,
    total: landscape + animalTotal,
    cubes: board.filter((c) => c.animal !== null && !CARDS[c.animal].spirit)
      .length,
    river,
    islands,
    cardScores,
  };
}
export function soloSuns(player: Player, side: Side) {
  const score = scorePlayer(player, side).total,
    base = [40, 70, 90, 110, 130, 140, 150, 160].filter(
      (n) => score >= n,
    ).length;
  const spirit = player.cards.find((c) => CARDS[c.id].spirit),
    spiritBonus = spirit
      ? [33, 34, 37, 38, 41].includes(spirit.id)
        ? 1
        : 0
      : 2;
  const sideBonus = side === "A" ? 1 : 0;
  return {
    base,
    sideBonus,
    spiritBonus,
    total: base + sideBonus + spiritBonus,
  };
}
export function ranking(game: Game) {
  const list = game.players
    .map((player) => ({
      player,
      score: scorePlayer(player, game.side),
      rank: 0,
    }))
    .sort(
      (a, b) => b.score.total - a.score.total || b.score.cubes - a.score.cubes,
    );
  list.forEach((entry, i) => {
    entry.rank =
      i &&
      entry.score.total === list[i - 1].score.total &&
      entry.score.cubes === list[i - 1].score.cubes
        ? list[i - 1].rank
        : i + 1;
  });
  return list;
}
export function endingReason(game: Game): string {
  if (game.finalRound) return game.finalReason;
  const player = game.players[game.active];
  if (emptyCount(player) <= 2)
    return `${player.name} 的版图只剩 ${emptyCount(player)} 个空格`;
  if (game.bag.length === 0) return "地形袋已空";
  return "";
}
export function canEndTurn(game: Game) {
  return (
    game.phase === "playing" &&
    game.turn.tookTokens &&
    game.turn.pending.length === 0 &&
    game.players[game.active].spiritChoices.length === 0
  );
}
function requireRule(ok: unknown, message: string): asserts ok {
  if (!ok) throw new Error(message);
}
export function applyAction(previous: Game, action: Action): Game {
  requireRule(previous.phase === "playing", "游戏已经结束。");
  const game = clone(previous),
    player = game.players[game.active];
  requireRule(
    !player.spiritChoices.length || action.type === "chooseSpirit",
    "请先选择一张自然之灵。",
  );
  let message = "";
  switch (action.type) {
    case "chooseSpirit":
      requireRule(
        player.spiritChoices.includes(action.card),
        "请选择发给你的自然之灵。",
      );
      player.cards.push({ id: action.card, placed: 0 });
      player.spiritChoices = [];
      message = `选择了${CARDS[action.card].name}`;
      break;
    case "takeTokens": {
      const group = game.supply[action.group];
      requireRule(
        !game.turn.tookTokens && group?.length === 3,
        "每回合必须且只能拿取一组地形片。",
      );
      game.turn.pending = [...group];
      game.supply[action.group] = [];
      game.turn.tookTokens = true;
      message = `拿取${group.map((c) => COLORS[c].name).join("、")}`;
      break;
    }
    case "place": {
      const color = game.turn.pending[action.pending],
        cell = player.board[action.cell];
      requireRule(
        color && cell && canPlace(cell, color),
        "这里不能放置这枚地形片：请检查堆叠规则或动物占用。",
      );
      cell.tokens.push(color);
      game.turn.pending.splice(action.pending, 1);
      message = `在 ${cell.col + 1}-${cell.row + 1} 放置${COLORS[color].name}`;
      break;
    }
    case "takeCard": {
      const id = game.market[action.slot];
      requireRule(
        !game.turn.tookCard && id && activeCards(player).length < 4,
        "每回合最多拿 1 张牌，同时最多持有 4 张未完成的牌。",
      );
      player.cards.push({ id, placed: 0 });
      game.market[action.slot] = null;
      game.turn.tookCard = true;
      message = `邀请了${CARDS[id].name}`;
      break;
    }
    case "settle": {
      const card = player.cards.find((c) => c.id === action.card);
      requireRule(card && !isComplete(card), "请选择一张未完成的牌。");
      requireRule(
        matches(player.board, card.id).some((m) => m.target === action.cell),
        "栖息地的图案、高度或动物落点不符合要求。",
      );
      player.board[action.cell].animal = card.id;
      card.placed++;
      message = `让${CARDS[card.id].name}入住${isComplete(card) ? "，完成卡牌" : ""}`;
      break;
    }
    case "endTurn": {
      requireRule(canEndTurn(game), "请先拿取一组地形片，并放完全部 3 枚。");
      if (action.discardSlot !== undefined) {
        const id = game.market[action.discardSlot];
        requireRule(
          game.players.length === 1 &&
            !game.turn.tookCard &&
            id &&
            game.deck.length,
          "仅单人模式未拿动物牌的回合，可以在回合结束时替换 1 张公共动物牌。",
        );
        game.discardedCards.push(id);
        game.market[action.discardSlot] = null;
      }
      const reason = endingReason(game);
      if (reason && !game.finalRound) {
        game.finalRound = game.round;
        game.finalReason = reason;
      }
      player.turns++;
      if (game.finalRound && game.active === game.players.length - 1) {
        game.phase = "ended";
        message = "结束回合，游戏结束";
        break;
      }
      if (game.players.length === 1) {
        game.discardedTokens.push(...game.supply.flat());
        game.supply = game.supply.map(() => []);
      }
      // Only complete groups are offered. If the bag contains the final three
      // tokens, solo gets one last choice; an empty bag triggers at the next refill.
      game.supply = game.supply.map((group) =>
        group.length ? group : game.bag.splice(0, 3),
      );
      game.market = game.market.map((id) => id ?? game.deck.shift() ?? null);
      game.active = (game.active + 1) % game.players.length;
      if (game.active === 0) game.round++;
      game.turn = { tookTokens: false, tookCard: false, pending: [] };
      message = "结束回合";
      break;
    }
  }
  game.log.push(`第 ${previous.round} 轮 · ${player.name} ${message}`);
  return game;
}
export function dispatch(session: Session, action: Action): Session {
  const game = applyAction(session.game, action);
  return {
    game,
    history:
      action.type === "endTurn" ? [] : [...session.history, session.game],
  };
}
export function undo(session: Session, wholeTurn = false): Session {
  if (!session.history.length || session.game.phase === "ended") return session;
  return {
    game: clone(wholeTurn ? session.history[0] : session.history.at(-1)!),
    history: wholeTurn ? [] : session.history.slice(0, -1),
  };
}
