# Harmonies · 万物和鸣

基于 Harmonies 基础版规则的中文 Web 桌游，使用 React、TypeScript 和 Vite。1–4 人在同一设备轮流操作，游戏保存在当前浏览器。

## 启动

```bash
npm install
npm run dev
```

打开终端输出的本地地址（默认 http://127.0.0.1:5173）。生产构建使用 `npm run build`；`npm run preview` 可预览构建结果。建议 Node.js 22 或更新版本。

## 已实现

- 设置页：1–4 位人类或启发式 AI 玩家、姓名、先手、A/B 面、自然之灵开关。
- AI v3.1：只搜索当前回合，使用 v1 便宜评分与六项目 V 的拿牌收益下限，同分优先更多空牌位，评分前去重、缓存复用、预算内逐次扩大束宽；独立思考预算、暂停/继续及全 AI 观战，不使用 RL 或远程模型。
- 32 张动物牌、10 张自然之灵、原版卡牌图案与点数、120 枚地形的正确配比。
- 取整组地形、合法堆叠、拿动物牌、六方向栖息地匹配、动物/自然之灵入住。可穿插操作；完成卡不占 4 张持牌上限。
- 本回合的逐步撤销、整回合重置；回合提交后不可跨玩家撤销。
- 官方 Solo 的 3 组供应、3 张动物牌、未选地形移出游戏、回合末可选换牌、太阳评级。
- 版图、他人缩略版图、悬停放大/点击查看、合法落点、原版卡牌详情与可读图案、操作记录、常驻计分提示。
- 最后一轮提示、同轮等回合数结束、显式结算按钮、官方计分纸的地形/动物双列小计与总分、排名及同分判定。
- 自动保存、继续上次游戏、键盘操作、手机布局。
- 操作记录中展开 AI 决策，按操作类型显示已评分候选的 top-k、评分变化和 m/d/p/B/F 等变量；另有完整方案 V 对比。仅转储搜索已有值，可导出 JSON，兼容旧 v1/v2 日志。

本版仅包含基础版和基础版自然之灵，不包含 Pulse、Crescendo 或宣传卡。

## 验证

```bash
npm test
npm run test:e2e
npm run build
```

浏览器测试默认使用本机 Chrome，若未安装，可安装 Chrome，或调整 `playwright.config.ts` 使用 Playwright Chromium。测试会自动启动本地开发服务，也会复用已启动的 5173 服务。

- 22 项规则测试：包含 42 张牌 × 6 个方向、全部自然之灵、河流/岛屿、撤销、终局、官方 116 分示例，以及 64 局不同人数/版图/自然之灵配置的完整对局与组件守恒检查。
- 35 项 AI 测试：精确补建、统一估值、预算跨回合不变、旋转等价、共享与冲突、资源守恒、隐藏信息隔离、当前回合搜索、去重、日志零补算、超时保留完整方案、终局穷举对照，以及计分优化与原实现的等价性。
- 11 项浏览器测试：设置、多人交接、动物入住、撤销、保存恢复、Solo 整局结算、移动端操作，以及 AI 暂停、混合新旧日志导出与 IndexedDB 恢复。

## 代码结构

- `src/game/engine.ts`：纯规则引擎。`applyAction` 处理动作；`dispatch`/`undo` 管理本回合历史。UI 不决定规则。
- `src/game/cards.json`：完整卡牌的栖息地与计分数据。
- `src/App.tsx`：设置页、游戏桌、卡牌/规则对话框及结算。
- `src/game/ai/`：可见信息、项目估值、合法动作、搜索、Worker 与日志导出。
- `src/useAI.ts`：AI 回合调度、取消及逐步展示。
- `src/GameLog.tsx`：普通操作和 AI 决策日志界面。
- `tests/engine.test.ts`：规则与完整对局测试。
- `tests/browser/game.spec.ts`：浏览器交互验收。

AI 与人类共用同一 `Action` 和 `dispatch`；本版没有联机服务。对局存档为 `Session`，位于 localStorage 的 `harmonies.session.v1`；AI 详细日志另存 IndexedDB，刷新后自动恢复。

AI 的确定性估值公式、搜索近似、日志格式与验证方式见 [docs/ai.md](docs/ai.md)。默认每回合预算 3.5 秒；实际耗时、束宽和完整方案覆盖显示在界面及日志中。

历史 v2 的回归证据与完整对局耗时保留在 [docs/ai-v2-validation.md](docs/ai-v2-validation.md)；原搜索模块也保留，当前默认使用 v3。

运行 `npm run benchmark:compare-ai` 可与 Git 中保留的旧版 `f36c7cc` 做相同思考预算的完整对局比较。脚本独立提取未修改的旧版代码，串行运行交换先后手的双人局及相同种子的 Solo 局，记录最终分、实际耗时与完成深度；详见 [docs/ai-v3-time-comparison.md](docs/ai-time-comparison.md)。

规则核对及界面参考见 [docs/rules-and-sources.md](docs/rules-and-sources.md)。图像、卡牌数据及许可说明见 [docs/ATTRIBUTION.md](docs/ATTRIBUTION.md)。
