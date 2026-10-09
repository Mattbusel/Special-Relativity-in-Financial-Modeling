<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.png">
  <img alt="SPY の日足終値を世界線として描いた図：時間的な区間はティールの実線、空間的な区間は赤の破線で、時間的な足には光円錐が描かれている。regime_validator バイナリの実際の出力。" src="assets/hero-light.png">
</picture>

<p align="center">
  <a href="https://special-relativity-finance.vercel.app/"><b>プロジェクトサイト</b></a> &middot;
  <a href="#download">ダウンロード</a> &middot;
  <a href="#how-it-works">仕組み</a> &middot;
  <a href="#examples">実行例</a> &middot;
  <a href="docs/REFERENCE.md">詳細リファレンス</a>
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT ライセンス" src="https://img.shields.io/badge/license-MIT-00809A.svg"></a>
</p>

# Special Relativity in Financial Modeling

[English](README.md) | [简体中文](README.zh-CN.md) | 日本語 | [한국어](README.ko.md)

**特殊相対性理論の数学を借りて、株式や暗号資産のローソク足 1 本 1 本を「通常の値動き（時間的）」か「ジャンプ（空間的）」かに分類する、C++ ライブラリとコマンドラインツールです。** 各足には、価格、出来高、モメンタムから計算される、市場の「速度上限」に対する速度 β、ローレンツ因子 γ、時空間隔 ds² が与えられます。

この物理のアナロジーを実際の OHLCV データで試してみたいクオンツ、学生、好奇心旺盛なトレーダーに向けたプロジェクトです。**研究用のコードであり、投資助言ではありません**。市場が相対性理論に従うと主張するものではなく、ここにあるものはどれも検証済みのトレード戦略ではありません。

<a id="download"></a>
## ダウンロード

<a href="https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases/permalink/latest/downloads/srfm-windows-x64.zip"><b>Windows 版をダウンロード（.exe ツール入りの .zip）</b></a>：解凍して `demo.bat` をダブルクリックします。

| プラットフォーム | 入手方法 |
|---|---|
| Windows x64 | [`srfm-windows-x64.zip`](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases/permalink/latest/downloads/srfm-windows-x64.zip)：4 つの `.exe` ツール、サンプルデータ、`demo.bat` |
| Linux x64 | [`srfm-linux-x64.tar.gz`](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases)：`tar xzf srfm-linux-x64.tar.gz && ./srfm/demo.sh` |
| macOS、または C++ ライブラリとして | CMake でソースからビルド：[コマンド 3 つ](docs/REFERENCE.md#build-from-source) |

ダウンロードには `regime_validator`（足にラベルを付ける）、`backtest_runner`、`srfm`、`lorentz_basics` と、10 銘柄分の日足サンプルデータ（SPY、QQQ、AAPL、NVDA、BTC-USD など）が入っています。どのツールも `--help` でオプションと使用例を表示します。

<a id="how-it-works"></a>
## 仕組み

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/how-it-works-dark.svg">
  <img alt="4 つのステップ：1 本の足（終値、出来高、モメンタム）を取り出す。それぞれを直近 20 本に対する標準偏差の単位に変換する。前の足からのステップを測る（ステップ = dP の 2 乗 + dV の 2 乗 + dM の 2 乗の平方根）。それを速度上限 c = 1 本あたり 1 と比べる。上限未満なら時間的で光円錐の内側、上限を超えれば空間的。" src="assets/how-it-works-light.svg" width="100%">
</picture>

**時空間隔**は、ステップを速度上限と比べて測ったものです：`ds² = step² − (c·Δt)²`。負なら足は自分の光円錐の内側にとどまった（時間的、通常の値動き）、正なら外に出た（空間的、ジャンプ）ことを意味します。コードは `SpacetimeInterval::compute` と `MarketManifold::classify` です。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/lorentz-factor-dark.svg">
  <img alt="beta に対するローレンツ因子 gamma の曲線：ゆっくりした足では 1 付近で平ら、beta 0.9 で 2.29、beta 0.99 で 7.09 まで上がる。下端の目盛りは時間的な SPY の足 295 本を示し、beta の中央値は 0.72。gamma 1 から 3 の帯は srfm --backtest が使うポジションサイズの範囲。" src="assets/lorentz-factor-light.svg" width="100%">
</picture>

**ローレンツ因子** γ = 1/√(1 − β²) は、足の速度を倍率に変換します。落ち着いた足ではほぼ 1、上限に近づくと大きくなります。`srfm --backtest` はこれを使ってポジションサイズを 1x から 3x の間で決めます。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/spy-last-60-dark.svg">
  <img alt="2025 年 12 月から 2026 年 2 月までの、直近 60 本の SPY 日足の棒グラフ。各足のステップの大きさを、1 の位置にある破線の速度上限と比べている。18 本は上限の下（ティール、時間的）、42 本は上限を超える（赤、空間的）。" src="assets/spy-last-60-light.svg" width="100%">
</picture>

この 3 つの図は [`scripts/figures/make_explainer.py`](scripts/figures/make_explainer.py) で作っています。このスクリプトは `validation/data/SPY_1m.csv` に対して C++ の数式を素の Python で計算し直し、すべてのラベルが `regime_validator` の出力と一致しない限り図を描きません（2026-09-28 時点で 1,245 本すべてが一致）。図に示した β はステップ 3 のステップです。`regime_validator` が書き出す `beta` 列は別途正規化した計算から来ているため、値が異なることがあります（ほとんどの日足では上限の 0.9999 に張り付いています）。

<a id="examples"></a>
## 実行例

2026-09-28 に Windows ビルドで、リポジトリとダウンロードに同梱されているサンプルデータを使って実行した実際の出力です。

**1. 1 銘柄のすべての足にラベルを付ける**（`demo.bat` が最初に実行するもの）

```text
> regime_validator --input validation/data/SPY_1m.csv --output spy_regime.csv --ticker SPY
[SPY] Loaded 1256 bars
[SPY] Classified 1245 bars
  TIMELIKE:  295  (23.6948%)
  SPACELIKE: 950  (76.3052%)
  LIGHTLIKE: 0  (0%)
[SPY] Output written to spy_regime.csv
```

`spy_regime.csv` は足 1 本につき 1 行です：`ticker, bar_index, interval_type, next_bar_abs_return, next_bar_return, beta, geodesic_deviation`。ビットコインも同じような結果で、`--input validation/data/BTC_USD_1m.csv` では 1,816 本中 441 本が時間的、1,375 本が空間的と判定されます。

**2. β、γ、時空間隔を手で計算する**（[`examples/lorentz_basics.cpp`](examples/lorentz_basics.cpp)）

```text
> lorentz_basics
beta = 0.00   gamma = 1.0000
beta = 0.50   gamma = 1.1547
beta = 0.90   gamma = 2.2942
beta = 0.99   gamma = 7.0888
dP = +0.4   ds2 = -0.84   Timelike
dP = +3.0   ds2 = +8.00   Spacelike
```

**3. ラベルを使って戦略を比較する**（`backtest_runner` が実行例 1 の CSV を読み込む）

```text
> backtest_runner --input spy_regime.csv --output-dir out --ticker SPY
  Strategy            Sharpe   Sortino      MDD    γ-IR
  RAW                   -0.3315  -0.4599  0.4554  -0.0209
  RELATIVISTIC          -0.3827  -0.5320  0.8610  -0.3710
  GEODESIC_DEV          0.2887  0.4238  0.1624  0.0182
[SPY] Backtest CSV written to out/SPY_backtest.csv
```

1 銘柄、インサンプル、コストなし。ツールが何を出力するかの例示であり、エッジがある証拠ではありません。

## 3 ステップで使う

1. 上からお使いのプラットフォーム向けの zip をダウンロードします（または[ソースからビルド](docs/REFERENCE.md#build-from-source)します）。
2. `demo.bat`（Windows）または `./demo.sh`（Linux）を実行して、同梱の SPY の足にラベルを付けます。
3. `regime_validator --input` を自分の CSV に向けます。ヘッダー行があり、少なくとも `close` と `volume` の列があるファイルなら何でも使えます。

## このラベルに意味はあるのか？

仮説は「空間的な足のあとには、時間的な足のあとよりもリターンの分散が大きくなる」というものです。10 銘柄のサンプルをまとめると、空間的な足のあとの分散は時間的な足のあとの約 **1.26x** です。バートレット検定ではこれは高度に有意とされますが、ファットテールに頑健なルビーン検定では有意になりません（p = 0.098）。効果量も小さいです（Cohen's d = 0.034）。エッジではなく、未解決の研究課題として扱ってください。完全な表、銘柄ごとのチャート、再現用のコマンドは [docs/REFERENCE.md の「The empirical question」](docs/REFERENCE.md#the-empirical-question)にあります。

## ドキュメント

| ドキュメント | 内容 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | その他すべて：ソースからのビルド、ビルドされるもの、C++ API、数学、テスト、ベンチマーク、機能ガイド（Python、Rust）、FAQ、論文 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | モジュール構成とデータの流れ |
| [BUILD.md](BUILD.md) | ビルドオプションとプラットフォームごとの注意点 |
| [RESULTS.md](RESULTS.md) | コミット済みの検証結果 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | ビルドや実行でよくある問題 |
| [CHANGELOG.md](CHANGELOG.md) | バージョン履歴 |
| [Paper 1.1.pdf](Paper%201.1.pdf) | このアイデアの元になった論文 |

## SRFM プロジェクトファミリー

| リポジトリ | 内容 |
|---|---|
| **Special-Relativity-in-Financial-Modeling**（このリポジトリ） | C++20 のコア：OHLCV の足に対する β、γ、時空間隔のラベル、クリストッフェル記号、測地線偏差。Python の検証スクリプト付き |
| [srfm-lab](https://gitlab.com/mattbusel/srfm-lab)（[サイト](https://srfm-lab.vercel.app/)） | このアイデアを土台にした多言語の研究ラボ：ブラックホールシグナル、モンテカルロ・バックテスト、ペーパートレーダー、アイデアエンジン |
| [srfm-python](https://gitlab.com/mattbusel/srfm-python) | ピュア Python の SDK：pandas の `df.srfm` アクセサと、ローレンツ因子パイプライン用の Polars ラッパー |
| [srfm-paper-impl](https://gitlab.com/mattbusel/srfm-paper-impl) | 論文（PDF）、その図を再生成するスクリプトとノートブック、コアとなる数式の小さな Rust リファレンス実装 |

## ライセンスと引用

MIT。[LICENSE](LICENSE) を参照してください。

```bibtex
@software{busel_srfm,
  author = {Busel, Matthew},
  title  = {Special Relativity in Financial Modeling},
  year   = {2025},
  url    = {https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling}
}
```
