import { test, expect } from "@playwright/test";
import { readFile } from "node:fs/promises";
import { createGame } from "../../src/game/engine";

test("AI setup, cancellation, human handoff, top-k export and IndexedDB recovery", async ({
  page,
}) => {
  test.setTimeout(60000);
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.getByLabel("玩家 1 类型").selectOption("ai");
  await page.getByLabel("玩家 1 AI 搜索深度").selectOption("1");
  await page.getByLabel("玩家 1 AI 思考预算").selectOption("1000");
  await page.getByLabel("玩家 1 AI 日志 top-k").selectOption("1");
  await page.screenshot({
    path: "/tmp/harmonies-ai-setup.png",
    fullPage: true,
  });
  await page.getByRole("button", { name: "开始游戏" }).click();
  await expect(page.getByRole("button", { name: "暂停 AI" })).toBeVisible();
  await page.getByRole("button", { name: "暂停 AI" }).click();
  const paused = await page.evaluate(() =>
    localStorage.getItem("harmonies.session.v1"),
  );
  await page.waitForTimeout(600);
  expect(
    await page.evaluate(() => localStorage.getItem("harmonies.session.v1")),
  ).toBe(paused);
  await expect(
    page.getByRole("button", { name: "结束回合", exact: true }),
  ).toBeDisabled();
  await page.getByRole("button", { name: "继续 AI" }).click();
  await expect(page.getByRole("heading", { name: "轮到 玩家 2" })).toBeVisible({
    timeout: 25000,
  });
  await page.getByRole("button", { name: "我准备好了" }).click();
  await page.getByRole("button", { name: "操作记录", exact: true }).click();
  await page.locator(".ai-decision > summary").first().click();
  await page.locator(".candidate-log > summary").first().click();
  await expect(page.locator(".estimate-detail").first()).toContainText("B=");
  await page
    .getByRole("dialog")
    .screenshot({ path: "/tmp/harmonies-ai-log.png" });
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("button", { name: "导出游戏日志" }).click();
  const download = await downloadPromise;
  const output = JSON.parse(await readFile((await download.path())!, "utf8"));
  expect(output.aiDecisions.length).toBeGreaterThanOrEqual(5);
  expect(
    output.aiDecisions.every((d: any) =>
      Object.values(d.topByOperation).every((list: any) => list.length <= 1),
    ),
  ).toBe(true);
  expect(output.currentPosition.bag).toBeUndefined();
  expect(output.players[0].ai.depth).toBe(1);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page
    .getByRole("dialog")
    .screenshot({ path: "/tmp/harmonies-ai-log-mobile.png" });
  await page.reload();
  await page.getByRole("button", { name: /继续上次旅程/ }).click();
  await page.getByRole("button", { name: "操作记录", exact: true }).click();
  await expect(page.locator(".ai-decision")).toHaveCount(
    output.aiDecisions.length,
  );
  expect(errors).toEqual([]);
});

test("AI completes a solo final turn and opens scores only on request", async ({
  page,
}) => {
  const game = createGame({
    names: ["AI 森林"],
    side: "A",
    spirits: false,
    first: 0,
    seed: 1,
    controllers: [{ type: "ai", ai: { depth: 2, timeMs: 100, topK: 1 } }],
  });
  game.players[0].board.slice(3).forEach((cell) => {
    cell.tokens = [1];
  });
  game.players[0].cards = [];
  game.turn = { tookCard: true, tookTokens: true, pending: [1, 4, 5] };
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
  await expect(page.getByText("游戏结束", { exact: true })).toBeVisible({
    timeout: 20000,
  });
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await page.getByRole("button", { name: "查看结算" }).first().click();
  await expect(page.locator(".scoresheet")).toHaveCount(1);
});

test("mobile AI settings and decision log stay inside viewport", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await page.getByLabel("玩家 1 类型").selectOption("ai");
  await expect(page.getByLabel("玩家 1 AI 日志 top-k")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.screenshot({
    path: "/tmp/harmonies-ai-mobile.png",
    fullPage: true,
  });
  await page.getByLabel("玩家 1 类型").selectOption("human");
  await page.getByRole("button", { name: "开始游戏" }).click();
  await page.getByRole("button", { name: "操作记录", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "导出游戏日志" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
});
