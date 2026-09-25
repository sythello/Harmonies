import { test, expect, type Page } from "@playwright/test";
import { createGame, type Game } from "../../src/game/engine";

async function loadGame(page: Page, game: Game) {
  await page.addInitScript(
    (g) =>
      localStorage.setItem(
        "harmonies.session.v1",
        JSON.stringify({ game: g, history: [] }),
      ),
    game,
  );
  await page.goto("/");
  await page.getByRole("button", { name: /继续上次旅程/ }).click();
}
async function placeAll(page: Page) {
  for (let i = 0; i < 3; i++)
    await page
      .locator(".personal-area .hex-cell.legal:not(.occupied)")
      .first()
      .click();
}
async function finish(page: Page) {
  await page.getByRole("button", { name: "结束回合", exact: true }).click();
  await page.getByRole("button", { name: "确认结束" }).click();
}

test("four-player setup supports human names, first player, B side and spirit choices", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByRole("button", { name: "4 人", exact: true }).click();
  await page.getByLabel("玩家 3 名称").fill("山间旅人");
  await page.getByRole("button", { name: /静谧群岛/ }).click();
  await page.getByRole("switch", { name: /自然之灵/ }).click();
  await page.getByLabel("先手玩家").selectOption("2");
  await page.getByRole("button", { name: "开始游戏" }).click();
  await expect(
    page.getByRole("heading", { name: "山间旅人，选择你的自然之灵" }),
  ).toBeVisible();
  await expect(page.locator(".spirit-choices .animal-card")).toHaveCount(2);
  await page.getByRole("button", { name: "选择这位自然之灵" }).first().click();
  await expect(page.locator(".personal-area .hex-cell")).toHaveCount(28);
  await expect(page.locator(".owned-card-wrap")).toHaveCount(1);
  await expect(page.locator(".opponent")).toHaveCount(3);
  expect(errors).toEqual([]);
});
test("token placement, card drafting, undo, whole-turn reset, handoff and persisted resume", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "开始游戏" }).click();
  await expect(
    page.getByRole("button", { name: "结束回合", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: /拿取第 1 组地形/ }).click();
  await placeAll(page);
  await page.locator(".market-row .animal-card").first().click();
  await page.getByRole("button", { name: "邀请这位朋友" }).click();
  await expect(page.locator(".owned-card-wrap")).toHaveCount(1);
  await page.getByRole("button", { name: "撤销上一步" }).click();
  await expect(page.locator(".owned-card-wrap")).toHaveCount(0);
  await page.getByRole("button", { name: "重置本回合" }).click();
  await expect(page.locator(".personal-area .hex-cell.occupied")).toHaveCount(
    0,
  );
  await expect(page.locator(".supply-group:not(:disabled)")).toHaveCount(5);
  await page.getByRole("button", { name: /拿取第 2 组地形/ }).click();
  await placeAll(page);
  await finish(page);
  await expect(
    page.getByRole("heading", { name: "轮到 玩家 2" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "我准备好了" }).click();
  await expect(page.getByRole("button", { name: "撤销上一步" })).toBeDisabled();
  await page.locator(".opponent").hover();
  await expect(page.locator(".hover-board")).toBeVisible();
  await page.locator(".opponent").click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByRole("button", { name: "关闭", exact: true }).click();
  await page.reload();
  await page.getByRole("button", { name: /继续上次旅程/ }).click();
  await expect(page.locator(".personal-heading h2")).toHaveText(
    "玩家 2 的自然栖所",
  );
  await expect(page.locator(".opponent .occupied")).toHaveCount(3);
});
test("animal settlement highlights legal targets, preserves habitats and is undoable", async ({
  page,
}) => {
  const game = createGame({
    names: ["森林"],
    side: "A",
    spirits: false,
    first: 0,
    seed: 1,
  });
  game.players[0].cards = [{ id: 5, placed: 0 }];
  game.deck = game.deck.filter((id) => id !== 5);
  game.market = game.market.map((id) => (id === 5 ? game.deck.shift()! : id));
  game.players[0].board[11].tokens = [4];
  game.players[0].board[6].tokens = [1];
  await loadGame(page, game);
  await page.locator(".owned-cards .animal-card").click();
  await expect(page.locator(".personal-area .hex-cell.legal")).toHaveCount(1);
  await page.locator(".personal-area .hex-cell.legal").click();
  await expect(
    page.locator('.personal-area [aria-label*="青蛙已入住"]'),
  ).toHaveCount(1);
  await expect(page.locator(".live-score strong")).toHaveText("3");
  await page.getByRole("button", { name: "撤销上一步" }).click();
  await expect(
    page.locator('.personal-area [aria-label*="青蛙已入住"]'),
  ).toHaveCount(0);
  await expect(page.locator(".live-score strong")).toHaveText("1");
});
test("solo optional replacement appears only without drafting and removes unused supply", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "1 人", exact: true }).click();
  await page.getByRole("button", { name: "开始游戏" }).click();
  await expect(page.locator(".supply-group")).toHaveCount(3);
  await expect(page.locator(".market-row .animal-card")).toHaveCount(3);
  const before = await page
    .locator(".market-row .animal-card")
    .first()
    .getAttribute("aria-label");
  await page.getByRole("button", { name: /拿取第 1 组地形/ }).click();
  await placeAll(page);
  await page.getByRole("button", { name: "结束回合", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: /单人可选操作/ }),
  ).toBeVisible();
  await page.locator(".solo-discard-cards .animal-card").first().click();
  await page.getByRole("button", { name: "确认结束" }).click();
  expect(
    await page
      .locator(".market-row .animal-card")
      .first()
      .getAttribute("aria-label"),
  ).not.toEqual(before);
  const saved = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("harmonies.session.v1")!),
  );
  expect(saved.game.discardedTokens).toHaveLength(6);
  expect(saved.game.bag).toHaveLength(102);
  expect(saved.history).toHaveLength(0);
});
test("a complete solo journey reaches final-round notice and user-triggered official scoresheet", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: "1 人", exact: true }).click();
  await page.getByRole("button", { name: "开始游戏" }).click();
  for (let turn = 1; turn <= 7; turn++) {
    await page.locator(".supply-group:not(:disabled)").first().click();
    await placeAll(page);
    if (turn === 7)
      await expect(page.locator(".final-banner")).toContainText("最后一轮");
    await finish(page);
  }
  await expect(page.locator(".ended-banner")).toContainText("游戏结束");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page
    .locator(".ended-banner")
    .getByRole("button", { name: "查看结算" })
    .click();
  await expect(
    page.getByRole("heading", { name: "万物和鸣 · 结算记录" }),
  ).toBeVisible();
  await expect(page.locator(".scoresheet")).toHaveCount(1);
  await expect(page.locator(".scoresheet")).toContainText("地形景观");
  await expect(page.locator(".sun-breakdown")).toBeVisible();
  await page.screenshot({ path: "/tmp/harmonies-results.png", fullPage: true });
});
test("multiplayer game ends after equal turns, with scored ranking and no premature result dialog", async ({
  page,
}) => {
  const game = createGame({
    names: ["橡木", "溪流"],
    side: "A",
    spirits: false,
    first: 0,
    seed: 5,
  });
  game.players[0].board.slice(0, 21).forEach((c) => (c.tokens = [1]));
  game.turn = { tookTokens: true, tookCard: false, pending: [] };
  await loadGame(page, game);
  await expect(page.locator(".final-banner")).toContainText("最后一轮");
  await finish(page);
  await page.getByRole("button", { name: "我准备好了" }).click();
  await expect(page.locator(".final-banner")).toContainText("最后一轮");
  await page.locator(".supply-group:not(:disabled)").first().click();
  await placeAll(page);
  await finish(page);
  await expect(page.locator(".ended-banner")).toBeVisible();
  const g = await page.evaluate(
    () => JSON.parse(localStorage.getItem("harmonies.session.v1")!).game,
  );
  expect(g.players.map((p: { turns: number }) => p.turns)).toEqual([1, 1]);
  await page.locator(".end-turn").click();
  await expect(page.locator(".scoresheet")).toHaveCount(2);
  await expect(page.locator(".result-banner h3")).toContainText("获得胜利");
});
test("mobile setup and play have no page-level horizontal overflow, and dialogs remain usable", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.screenshot({
    path: "/tmp/harmonies-mobile-setup.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.getByRole("button", { name: "开始游戏" }).click();
  await page.screenshot({
    path: "/tmp/harmonies-mobile-game.png",
    fullPage: true,
  });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page.locator(".market-row .animal-card").first().click();
  await expect(
    page.getByRole("button", { name: "邀请这位朋友" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "邀请这位朋友" }).click();
  await page.getByRole("button", { name: /拿取第 1 组地形/ }).click();
  await placeAll(page);
  await finish(page);
  await expect(page.getByRole("button", { name: "我准备好了" })).toBeVisible();
});
