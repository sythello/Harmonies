# Harmonies · 万物和鸣

基于 Harmonies 基础版规则的中文 Web 桌游，使用 React、TypeScript 和 Vite。1–4 人在同一设备轮流操作，游戏保存在当前浏览器。

## 启动

```bash
npm install
npm run dev
```

打开终端输出的本地地址（默认 http://127.0.0.1:5173）。生产构建使用 `npm run build`；`npm run preview` 可预览构建结果。建议 Node.js 22 或更新版本。

## 已实现

- 设置页：1–4 位人类玩家、姓名、先手、A/B 面、自然之灵开关。AI 选项仅作为未来扩展提示。
- 32 张动物牌、10 张自然之灵、原版卡牌图案与点数、120 枚地形的正确配比。
- 取整组地形、合法堆叠、拿动物牌、六方向栖息地匹配、动物/自然之灵入住。可穿插操作；完成卡不占 4 张持牌上限。
- 本回合的逐步撤销、整回合重置；回合提交后不可跨玩家撤销。
- 官方 Solo 的 3 组供应、3 张动物牌、未选地形移出游戏、回合末可选换牌、太阳评级。
- 版图、他人缩略版图、悬停放大/点击查看、合法落点、原版卡牌详情与可读图案、操作记录、常驻计分提示。
- 最后一轮提示、同轮等回合数结束、显式结算按钮、官方计分纸的地形/动物双列小计与总分、排名及同分判定。
- 自动保存、继续上次游戏、键盘操作、手机布局。

本版仅包含基础版和基础版自然之灵，不包含 Pulse、Crescendo 或宣传卡。

## 验证

```bash
npm test
npm run test:e2e
npm run build
```

浏览器测试默认使用本机 Chrome，若未安装，可安装 Chrome，或调整 `playwright.config.ts` 使用 Playwright Chromium。测试会自动启动本地开发服务，也会复用已启动的 5173 服务。

- 22 项规则测试：包含 42 张牌 × 6 个方向、全部自然之灵、河流/岛屿、撤销、终局、官方 116 分示例，以及 64 局不同人数/版图/自然之灵配置的完整对局与组件守恒检查。
- 7 项浏览器测试：设置、多人交接、动物入住、撤销、保存恢复、Solo 刷新与整局结算、移动端操作。

## 代码结构

- `src/game/engine.ts`：纯规则引擎。`applyAction` 处理动作；`dispatch`/`undo` 管理本回合历史。UI 不决定规则。
- `src/game/cards.json`：完整卡牌的栖息地与计分数据。
- `src/App.tsx`：设置页、游戏桌、卡牌/规则对话框及结算。
- `tests/engine.test.ts`：规则与完整对局测试。
- `tests/browser/game.spec.ts`：浏览器交互验收。

未来 AI 可生成同一 `Action` 联合类型并交给 `dispatch`；本版没有 AI 决策或联机服务。存档为 `Session`，位于 localStorage 的 `harmonies.session.v1`。

规则核对及界面参考见 [docs/rules-and-sources.md](docs/rules-and-sources.md)。图像、卡牌数据及许可说明见 [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md)。
