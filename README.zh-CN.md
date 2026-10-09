<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.png">
  <img alt="把 SPY 日收盘价画成一条世界线：类时段为青绿色实线，类空段为红色虚线，类时 K 线处画有光锥。这是 regime_validator 程序的真实输出。" src="assets/hero-light.png">
</picture>

<p align="center">
  <a href="https://special-relativity-finance.vercel.app/"><b>项目网站</b></a> &middot;
  <a href="#download">下载</a> &middot;
  <a href="#how-it-works">工作原理</a> &middot;
  <a href="#examples">示例</a> &middot;
  <a href="docs/REFERENCE.md">完整参考</a>
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-00809A.svg"></a>
</p>

# Special Relativity in Financial Modeling

[English](README.md) | 简体中文 | [日本語](README.ja.md) | [한국어](README.ko.md)

**一个 C++ 库和一组命令行工具，借用狭义相对论的数学，把股票或加密货币的每一根 K 线标记为普通波动（类时）或跳空（类空）。** 每根 K 线都会根据它的价格、成交量和动量，算出一个相对于市场“速度上限”的速度 β、一个洛伦兹因子 γ 和一个时空间隔 ds²。

面向想在真实 OHLCV 数据上探索这个物理类比的量化研究员、学生和好奇的交易者。**这是研究代码，不构成投资建议**：它并不声称市场遵循相对论，这里也没有任何经过检验的交易策略。

<a id="download"></a>
## 下载

<a href="https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases/permalink/latest/downloads/srfm-windows-x64.zip"><b>下载 Windows 版（含 .exe 工具的 .zip）</b></a>：解压后双击 `demo.bat`。

| 平台 | 获取方式 |
|---|---|
| Windows x64 | [`srfm-windows-x64.zip`](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases/permalink/latest/downloads/srfm-windows-x64.zip)：四个 `.exe` 工具、示例数据和 `demo.bat` |
| Linux x64 | [`srfm-linux-x64.tar.gz`](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases)：`tar xzf srfm-linux-x64.tar.gz && ./srfm/demo.sh` |
| macOS，或使用 C++ 库 | 用 CMake 从源码构建：[三条命令](docs/REFERENCE.md#build-from-source) |

下载包里有 `regime_validator`（给 K 线打标签）、`backtest_runner`、`srfm` 和 `lorentz_basics`，外加十个示例标的的日线数据（SPY、QQQ、AAPL、NVDA、BTC-USD 等）。每个工具加上 `--help` 都会打印它的选项和一个示例。

<a id="how-it-works"></a>
## 工作原理

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/how-it-works-dark.svg">
  <img alt="四个步骤：取一根 K 线（收盘价、成交量、动量）；把每一项换算成相对最近 20 根 K 线的标准差倍数；测量它相对上一根 K 线的步长，步长 = dP 的平方加 dV 的平方加 dM 的平方再开平方根；与速度上限 c = 每根 K 线 1 进行比较。低于上限为类时，位于光锥之内；超过上限为类空。" src="assets/how-it-works-light.svg" width="100%">
</picture>

**时空间隔**就是用速度上限衡量的步长：`ds² = step² − (c·Δt)²`。为负表示这根 K 线留在了自己的光锥之内（类时，普通波动）；为正表示它离开了光锥（类空，跳空）。对应代码是 `SpacetimeInterval::compute` 和 `MarketManifold::classify`。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/lorentz-factor-dark.svg">
  <img alt="洛伦兹因子 gamma 随 beta 变化的曲线：慢速 K 线处接近 1 且平坦，beta 为 0.9 时升到 2.29，beta 为 0.99 时升到 7.09。底部的刻度标出了 295 根类时 SPY K 线，beta 中位数为 0.72。gamma 从 1 到 3 的阴影带表示 srfm --backtest 使用的仓位范围。" src="assets/lorentz-factor-light.svg" width="100%">
</picture>

**洛伦兹因子** γ = 1/√(1 − β²) 把一根 K 线的速度变成一个乘数：平静的 K 线约为 1，接近上限时会变得很大。`srfm --backtest` 用它把仓位大小设在 1x 到 3x 之间。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/spy-last-60-dark.svg">
  <img alt="最近 60 根 SPY 日 K 线的柱状图，时间从 2025 年 12 月到 2026 年 2 月。每根 K 线的步长都与位于 1 的虚线速度上限对比：18 根低于上限（青绿色，类时），42 根超过上限（红色，类空）。" src="assets/spy-last-60-light.svg" width="100%">
</picture>

这三张图来自 [`scripts/figures/make_explainer.py`](scripts/figures/make_explainer.py)，它在 `validation/data/SPY_1m.csv` 上用纯 Python 重新计算 C++ 中的公式，只有每一个标签都与 `regime_validator` 的输出一致时才会画图（2026-09-28 全部 1,245 根 K 线都一致）。图中显示的 β 是第 3 步的步长；`regime_validator` 写出的 `beta` 列来自另一次单独归一化的计算，所以两者可能不同（对大多数日 K 线来说，它都停在 0.9999 的上限）。

<a id="examples"></a>
## 示例

真实输出，2026-09-28 使用 Windows 构建版本，在仓库和下载包自带的示例数据上运行。

**1. 给一个标的的每根 K 线打标签**（`demo.bat` 首先运行的就是这个）

```text
> regime_validator --input validation/data/SPY_1m.csv --output spy_regime.csv --ticker SPY
[SPY] Loaded 1256 bars
[SPY] Classified 1245 bars
  TIMELIKE:  295  (23.6948%)
  SPACELIKE: 950  (76.3052%)
  LIGHTLIKE: 0  (0%)
[SPY] Output written to spy_regime.csv
```

`spy_regime.csv` 每根 K 线一行：`ticker, bar_index, interval_type, next_bar_abs_return, next_bar_return, beta, geodesic_deviation`。比特币的情况类似：`--input validation/data/BTC_USD_1m.csv` 把 1,816 根 K 线中的 441 根标为类时，1,375 根标为类空。

**2. 手算 β、γ 和时空间隔**（[`examples/lorentz_basics.cpp`](examples/lorentz_basics.cpp)）

```text
> lorentz_basics
beta = 0.00   gamma = 1.0000
beta = 0.50   gamma = 1.1547
beta = 0.90   gamma = 2.2942
beta = 0.99   gamma = 7.0888
dP = +0.4   ds2 = -0.84   Timelike
dP = +3.0   ds2 = +8.00   Spacelike
```

**3. 基于标签比较策略**（`backtest_runner` 读取示例 1 生成的 CSV）

```text
> backtest_runner --input spy_regime.csv --output-dir out --ticker SPY
  Strategy            Sharpe   Sortino      MDD    γ-IR
  RAW                   -0.3315  -0.4599  0.4554  -0.0209
  RELATIVISTIC          -0.3827  -0.5320  0.8610  -0.3710
  GEODESIC_DEV          0.2887  0.4238  0.1624  0.0182
[SPY] Backtest CSV written to out/SPY_backtest.csv
```

单一标的、样本内、不计交易成本：这只是演示工具会报告什么，不能证明存在超额收益（edge）。

## 三步上手

1. 在上面下载适合你平台的压缩包（或者[从源码构建](docs/REFERENCE.md#build-from-source)）。
2. 运行 `demo.bat`（Windows）或 `./demo.sh`（Linux），给自带的 SPY K 线打标签。
3. 把 `regime_validator --input` 指向你自己的 CSV：任何带表头行、并且至少有 `close` 和 `volume` 两列的文件都可以。

## 这个标签有意义吗？

假设是：类空 K 线之后的收益方差比类时 K 线之后更大。把十个示例标的合并起来看，类空 K 线之后的方差约为类时 K 线之后的 **1.26x**。Bartlett 检验认为这高度显著，但对肥尾稳健的 Levene 检验并不认为显著（p = 0.098），而且效应量很小（Cohen's d = 0.034）。请把它当作一个开放的研究问题，而不是一个优势。完整表格、各标的图表以及复现命令见：[docs/REFERENCE.md，“The empirical question”](docs/REFERENCE.md#the-empirical-question)。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | 其余所有内容：从源码构建、构建产物、C++ API、数学推导、测试、基准测试、功能指南（Python、Rust）、FAQ、论文 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 模块布局和数据流 |
| [BUILD.md](BUILD.md) | 构建选项和平台说明 |
| [RESULTS.md](RESULTS.md) | 已提交的验证结果 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | 常见的构建和运行问题 |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 |
| [Paper 1.1.pdf](Paper%201.1.pdf) | 这个想法背后的论文 |

## SRFM 系列项目

| 仓库 | 简介 |
|---|---|
| **Special-Relativity-in-Financial-Modeling**（本仓库） | C++20 核心：在 OHLCV K 线上计算 β、γ、时空间隔标签、克里斯托费尔符号和测地线偏离，外加 Python 验证脚本 |
| [srfm-lab](https://gitlab.com/mattbusel/srfm-lab)（[网站](https://srfm-lab.vercel.app/)） | 基于这一思路构建的多语言研究实验室：黑洞信号、蒙特卡洛回测、模拟盘交易程序和策略想法引擎 |
| [srfm-python](https://gitlab.com/mattbusel/srfm-python) | 纯 Python SDK：pandas 的 `df.srfm` 访问器，以及用于洛伦兹因子流水线的 Polars 封装 |
| [srfm-paper-impl](https://gitlab.com/mattbusel/srfm-paper-impl) | 论文（PDF）、重新生成论文图表的脚本和一个 notebook，以及核心公式的小型 Rust 参考实现 |

## 许可证与引用

MIT，见 [LICENSE](LICENSE)。

```bibtex
@software{busel_srfm,
  author = {Busel, Matthew},
  title  = {Special Relativity in Financial Modeling},
  year   = {2025},
  url    = {https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling}
}
```
