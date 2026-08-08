# quant_desk

**完整投研 / 纸面交易 / 管理复盘** 工作台：选股 → 交易 → 收益评价 → 复盘。

```text
选股(selection) → 交易(paper book) → 评价(performance) → 复盘(review HTML)
```

## 目录

```text
quant_desk/
  config.yaml
  scripts/
    01_select.py          # 截面选股
    02_trade.py           # 按目标权重纸面调仓
    03_evaluate.py        # 收益率/风险评价
    04_review.py          # 中文复盘 HTML
    05_gates.py           # 门控 go/no-go 评价（quantkit.gates，fail-closed）
    run_lifecycle.py      # 历史模拟：整条链路一次跑完（出口自动接门控）
    run_today.py          # 今日：选股 + 调仓 + 快照 + 门控
  state/                  # 纸面账本 JSON（勿提交密钥）
  data/                   # 行情缓存
  output/                 # 报告与表格
  tests/                  # pytest：门控接线（fixtures/gate_metrics_go.json = GO 复现样例）
```

## 快速开始

```bash
cd ~/Private/quant-analysis/finance/projects/quant_desk

# A) 历史完整链路（推荐先跑这个）
quant-python scripts/run_lifecycle.py

# B) 分步
quant-python scripts/01_select.py
quant-python scripts/02_trade.py
quant-python scripts/03_evaluate.py
quant-python scripts/04_review.py
quant-python scripts/05_gates.py

# C) 今日例行
quant-python scripts/run_today.py

# 测试
quant-python -m pytest tests/ -q
```

## 门控（go/no-go，fail-closed）

生命周期出口（`run_lifecycle.py`）与今日例行（`run_today.py` → `05_gates.py`）都会调
`quantkit.gates.evaluate_gates` 并写 **`output/gate_report.json`**（verdict + 逐门
missing/failures + 入参快照）。规则：

- 本流水线只**实测** gate0 三腿：`turnover_annual`（fills 单边成交额/平均权益/年数，
  跨度 <28 天宁可缺测）、`aum_scale`（= `initial_cash`）、`friction_cost`/`cost_tier`
  （= config `cost_tier`，见下）；其余门（pbo/dsr/拥挤度/纸面/实盘）本地无证据 →
  **缺测即 NO-GO 并逐门点名**，绝不伪造。
- 外部证据（walk-forward 统计、拥挤度周评、纸面追踪月数……）写成 JSON 放
  `output/gate_metrics.json`（或 `05_gates.py --metrics PATH`）即合并复评；
  **实测键优先**于证据文件同名键。
- GO 复现样例（证据夹具 + 实测 gate0）：

  ```bash
  quant-python scripts/05_gates.py --metrics tests/fixtures/gate_metrics_go.json
  ```

## 成本纪律

config 用 `cost_tier: low|mid|high`（COST_TIERS 全口径双边 0.2%/0.4%/0.5%），
**禁止「万五」默认**；纸面账本单边费 = tier/2（`scripts/_common.resolve_fee_bps`）。
注意：`state/paper_book.json` 里已持久化的旧费率随账本走，`02_trade.py --reset` 后才按新档。

主报告：

- `output/lifecycle_report.html` — 历史生命周期复盘
- `output/review_latest.html` — 最新复盘
- `output/gate_report.json` — 门控 go/no-go（缺证据 fail-closed）
- 工具总览：`../../docs/toolkit_lifecycle_zh.html`

## 能力边界

| 有 | 暂无（可扩展） |
|----|----------------|
| 股票池截面打分选股 | 全市场实时扫单 |
| 因子过滤 + Top-N 权重 | 基本面三表深度因子 |
| 纸面成交账本 | 真实券商下单（见 vnpy_cloud_bridge） |
| 收益/回撤/贡献/复盘 HTML | 合规审计流水 |
