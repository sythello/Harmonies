# 来源与署名

## 原版游戏与插画

Harmonies 由 Johan Benvenuto 设计，Maëva Da Silva 绘制插画，Libellud 出版。名称、卡牌插画及原版视觉内容的权利属于相应权利人。本项目是本地桌游实现，不是 Libellud 或 Board Game Arena 官方产品。

`public/assets/animal-cards.webp` 为原版 42 张卡牌的参考图集，取自 [miles2542/harmonies-bot 的公开参考文件](https://github.com/miles2542/harmonies-bot/blob/main/docs/animalCards.webp)，该图集包含原版插画和卡牌图案。这里保留原图并用 CSS 定位显示，没有重新绘制或修改原版插画。该仓库的代码许可不代表原版游戏美术获得另行授权。

## 卡牌数据

`src/game/cards.json` 来自 [miles2542/harmonies-bot](https://github.com/miles2542/harmonies-bot/blob/main/docs/cards_database.json)，其仓库使用 Apache License 2.0。原始许可见 [cards-data-LICENSE](cards-data-LICENSE)。卡牌编号、图案、计分数据未改动；中文名称、说明、规则引擎和界面单独实现。

## 界面依赖

- React / React DOM：MIT。
- Vite：MIT。
- Lucide 图标：ISC。
- DM Sans、Noto Serif SC：通过 Google Fonts 加载，使用 SIL Open Font License；无法联网时使用本机字体回退，不影响游戏。
