<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/hero-dark.png">
  <img alt="SPY 일봉 종가를 세계선으로 그린 그림: 시간꼴 구간은 청록색 실선, 공간꼴 구간은 빨간 점선이며, 시간꼴 봉에는 광원뿔이 그려져 있다. regime_validator 바이너리의 실제 출력." src="assets/hero-light.png">
</picture>

<p align="center">
  <a href="https://special-relativity-finance.vercel.app/"><b>프로젝트 사이트</b></a> &middot;
  <a href="#download">다운로드</a> &middot;
  <a href="#how-it-works">작동 원리</a> &middot;
  <a href="#examples">예제</a> &middot;
  <a href="docs/REFERENCE.md">전체 레퍼런스</a>
</p>

<p align="center">
  <a href="LICENSE"><img alt="MIT 라이선스" src="https://img.shields.io/badge/license-MIT-00809A.svg"></a>
</p>

# Special Relativity in Financial Modeling

[English](README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | 한국어

**특수상대성이론의 수학을 빌려 주식이나 암호화폐의 가격 봉 하나하나를 평범한 움직임(시간꼴) 또는 점프(공간꼴)로 분류하는 C++ 라이브러리와 명령줄 도구입니다.** 각 봉에는 가격, 거래량, 모멘텀으로 계산한 값, 즉 시장의 "속도 한계"에 대한 속도 β, 로런츠 인자 γ, 시공간 간격 ds²가 매겨집니다.

이 물리학 비유를 실제 OHLCV 데이터로 탐구해 보고 싶은 퀀트, 학생, 호기심 많은 트레이더를 위한 프로젝트입니다. **연구용 코드이며 투자 조언이 아닙니다.** 시장이 상대성이론을 따른다고 주장하지 않으며, 여기 있는 어떤 것도 검증된 트레이딩 전략이 아닙니다.

<a id="download"></a>
## 다운로드

<a href="https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases/permalink/latest/downloads/srfm-windows-x64.zip"><b>Windows용 다운로드(.exe 도구가 든 .zip)</b></a>: 압축을 풀고 `demo.bat`을 더블클릭하세요.

| 플랫폼 | 받는 방법 |
|---|---|
| Windows x64 | [`srfm-windows-x64.zip`](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases/permalink/latest/downloads/srfm-windows-x64.zip): `.exe` 도구 네 개, 샘플 데이터, `demo.bat` |
| Linux x64 | [`srfm-linux-x64.tar.gz`](https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling/-/releases): `tar xzf srfm-linux-x64.tar.gz && ./srfm/demo.sh` |
| macOS, 또는 C++ 라이브러리 | CMake로 소스에서 빌드: [명령어 세 개](docs/REFERENCE.md#build-from-source) |

다운로드에는 `regime_validator`(봉에 라벨을 붙임), `backtest_runner`, `srfm`, `lorentz_basics`와 함께 열 개 종목의 일봉 샘플 데이터(SPY, QQQ, AAPL, NVDA, BTC-USD 등)가 들어 있습니다. 모든 도구는 `--help`를 붙이면 옵션과 사용 예를 출력합니다.

<a id="how-it-works"></a>
## 작동 원리

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/how-it-works-dark.svg">
  <img alt="네 단계: 봉 하나(종가, 거래량, 모멘텀)를 가져온다. 각각을 최근 20개 봉 기준 표준편차 단위로 바꾼다. 이전 봉에서의 걸음을 잰다(걸음 = dP 제곱 더하기 dV 제곱 더하기 dM 제곱의 제곱근). 이를 속도 한계 c = 봉당 1과 비교한다. 한계 미만이면 시간꼴로 광원뿔 안, 한계를 넘으면 공간꼴." src="assets/how-it-works-light.svg" width="100%">
</picture>

**시공간 간격**은 걸음을 속도 한계에 견주어 잰 값입니다: `ds² = step² − (c·Δt)²`. 음수면 봉이 자기 광원뿔 안에 머문 것(시간꼴, 평범한 움직임)이고, 양수면 광원뿔을 벗어난 것(공간꼴, 점프)입니다. 코드는 `SpacetimeInterval::compute`와 `MarketManifold::classify`입니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/lorentz-factor-dark.svg">
  <img alt="beta에 따른 로런츠 인자 gamma 곡선: 느린 봉에서는 1 근처에서 평평하고, beta 0.9에서 2.29, beta 0.99에서 7.09까지 올라간다. 아래쪽 눈금은 시간꼴 SPY 봉 295개를 표시하며, beta 중앙값은 0.72. gamma 1에서 3까지 칠해진 띠는 srfm --backtest가 쓰는 포지션 크기 범위." src="assets/lorentz-factor-light.svg" width="100%">
</picture>

**로런츠 인자** γ = 1/√(1 − β²)는 봉의 속도를 배수로 바꿉니다. 잔잔한 봉에서는 약 1이고, 한계에 가까워지면 커집니다. `srfm --backtest`는 이 값으로 포지션 크기를 1x에서 3x 사이로 정합니다.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/spy-last-60-dark.svg">
  <img alt="2025년 12월부터 2026년 2월까지 최근 60개 SPY 일봉의 막대 그래프. 각 봉의 걸음 크기를 1에 있는 점선 속도 한계와 비교했다. 18개는 한계 아래(청록색, 시간꼴), 42개는 한계 위(빨간색, 공간꼴)." src="assets/spy-last-60-light.svg" width="100%">
</picture>

이 세 그림은 [`scripts/figures/make_explainer.py`](scripts/figures/make_explainer.py)에서 나옵니다. 이 스크립트는 `validation/data/SPY_1m.csv`에서 C++ 수식을 순수 Python으로 다시 계산하고, 모든 라벨이 `regime_validator`의 출력과 일치하지 않으면 그림을 그리지 않습니다(2026-09-28에 1,245개 봉 전부 일치). 그림에 나온 β는 3단계의 걸음입니다. `regime_validator`가 쓰는 `beta` 열은 따로 정규화한 계산에서 나오기 때문에 값이 다를 수 있습니다(대부분의 일봉에서 상한 0.9999에 붙어 있습니다).

<a id="examples"></a>
## 예제

2026-09-28에 Windows 빌드로, 저장소와 다운로드에 들어 있는 샘플 데이터를 사용해 실행한 실제 출력입니다.

**1. 한 종목의 모든 봉에 라벨 붙이기**(`demo.bat`이 가장 먼저 실행하는 것)

```text
> regime_validator --input validation/data/SPY_1m.csv --output spy_regime.csv --ticker SPY
[SPY] Loaded 1256 bars
[SPY] Classified 1245 bars
  TIMELIKE:  295  (23.6948%)
  SPACELIKE: 950  (76.3052%)
  LIGHTLIKE: 0  (0%)
[SPY] Output written to spy_regime.csv
```

`spy_regime.csv`에는 봉마다 한 행이 들어갑니다: `ticker, bar_index, interval_type, next_bar_abs_return, next_bar_return, beta, geodesic_deviation`. 비트코인도 비슷합니다. `--input validation/data/BTC_USD_1m.csv`는 1,816개 봉 중 441개를 시간꼴로, 1,375개를 공간꼴로 분류합니다.

**2. β, γ, 시공간 간격 직접 계산하기**([`examples/lorentz_basics.cpp`](examples/lorentz_basics.cpp))

```text
> lorentz_basics
beta = 0.00   gamma = 1.0000
beta = 0.50   gamma = 1.1547
beta = 0.90   gamma = 2.2942
beta = 0.99   gamma = 7.0888
dP = +0.4   ds2 = -0.84   Timelike
dP = +3.0   ds2 = +8.00   Spacelike
```

**3. 라벨로 전략 비교하기**(`backtest_runner`가 예제 1의 CSV를 읽음)

```text
> backtest_runner --input spy_regime.csv --output-dir out --ticker SPY
  Strategy            Sharpe   Sortino      MDD    γ-IR
  RAW                   -0.3315  -0.4599  0.4554  -0.0209
  RELATIVISTIC          -0.3827  -0.5320  0.8610  -0.3710
  GEODESIC_DEV          0.2887  0.4238  0.1624  0.0182
[SPY] Backtest CSV written to out/SPY_backtest.csv
```

종목 하나, 표본 내, 비용 없음. 도구가 무엇을 보고하는지 보여 주는 예시일 뿐, 엣지가 있다는 증거가 아닙니다.

## 3단계로 사용하기

1. 위에서 내 플랫폼에 맞는 zip을 다운로드합니다(또는 [소스에서 빌드](docs/REFERENCE.md#build-from-source)합니다).
2. `demo.bat`(Windows) 또는 `./demo.sh`(Linux)를 실행해 함께 들어 있는 SPY 봉에 라벨을 붙입니다.
3. `regime_validator --input`을 내 CSV로 지정합니다. 헤더 행이 있고 최소한 `close`와 `volume` 열이 있는 파일이면 무엇이든 됩니다.

## 이 라벨에 의미가 있을까?

가설: 공간꼴 봉 다음에는 시간꼴 봉 다음보다 수익률 분산이 더 크다. 열 개 샘플 종목을 합쳐 보면, 공간꼴 봉 다음의 분산은 시간꼴 봉 다음 분산의 약 **1.26x**입니다. 바틀렛 검정은 이를 매우 유의하다고 보지만, 두꺼운 꼬리에 강건한 레빈 검정은 그렇지 않으며(p = 0.098) 효과 크기도 작습니다(Cohen's d = 0.034). 엣지가 아니라 열려 있는 연구 질문으로 봐 주세요. 전체 표, 종목별 차트, 재현 명령어: [docs/REFERENCE.md, "The empirical question"](docs/REFERENCE.md#the-empirical-question).

## 문서

| 문서 | 내용 |
|---|---|
| [docs/REFERENCE.md](docs/REFERENCE.md) | 그 밖의 전부: 소스에서 빌드, 빌드 결과물, C++ API, 수학, 테스트, 벤치마크, 기능 가이드(Python, Rust), FAQ, 논문 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 모듈 구성과 데이터 흐름 |
| [BUILD.md](BUILD.md) | 빌드 옵션과 플랫폼별 참고 사항 |
| [RESULTS.md](RESULTS.md) | 커밋된 검증 결과 |
| [docs/troubleshooting.md](docs/troubleshooting.md) | 자주 겪는 빌드 및 실행 문제 |
| [CHANGELOG.md](CHANGELOG.md) | 버전 기록 |
| [Paper 1.1.pdf](Paper%201.1.pdf) | 이 아이디어의 바탕이 된 논문 |

## SRFM 프로젝트 군

| 저장소 | 설명 |
|---|---|
| **Special-Relativity-in-Financial-Modeling**(이 저장소) | C++20 코어: OHLCV 봉에서 β, γ, 시공간 간격 라벨, 크리스토펠 기호, 측지선 편차 계산, 그리고 Python 검증 스크립트 |
| [srfm-lab](https://gitlab.com/mattbusel/srfm-lab)([사이트](https://srfm-lab.vercel.app/)) | 이 아이디어를 바탕으로 만든 다중 언어 연구 랩: 블랙홀 신호, 몬테카를로 백테스트, 모의투자 트레이더, 아이디어 엔진 |
| [srfm-python](https://gitlab.com/mattbusel/srfm-python) | 순수 Python SDK: pandas `df.srfm` 접근자와 로런츠 인자 파이프라인용 Polars 래퍼 |
| [srfm-paper-impl](https://gitlab.com/mattbusel/srfm-paper-impl) | 논문(PDF), 논문 그림을 다시 만드는 스크립트와 노트북, 핵심 수식의 작은 Rust 레퍼런스 구현 |

## 라이선스와 인용

MIT, [LICENSE](LICENSE) 참고.

```bibtex
@software{busel_srfm,
  author = {Busel, Matthew},
  title  = {Special Relativity in Financial Modeling},
  year   = {2025},
  url    = {https://gitlab.com/mattbusel/Special-Relativity-in-Financial-Modeling}
}
```
