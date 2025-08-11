# forex_indicators — Indicadores Técnicos para Forex em Python

Biblioteca simples e transparente (pandas/numpy) com uma coleção abrangente de indicadores técnicos usados no mercado Forex. Foco em clareza, compatibilidade com `pandas.Series/DataFrame` e API estável.

## Visão do projeto

- **Mais completo que bibliotecas populares (ex.: pandas-ta)**: nossa ambição é cobrir um conjunto amplo e coerente de indicadores, com API estável, validações explícitas e docstrings em português que facilitam busca semântica.
- **Independência do MetaTrader**: reproduza, estenda e combine indicadores nativamente em Python, sem depender de MT4/MT5. Ideal para fluxos de análise de dados e pesquisa quantitativa.
- **Código simples e transparente**: implementações diretas com `pandas`/`numpy`, priorizando legibilidade e reprodutibilidade para estudos, protótipos e produção.
- **Open source de verdade**: uma base de código aberta, clara e extensível — convidamos a comunidade a evoluir o ecossistema de indicadores com qualidade e testes.
- **Para análise de mercado e trading quantitativo**: pensado para pipelines de dados, notebooks, backtests e integrações com bibliotecas de modelagem/ML.

## Instalação

```bash
pip install -r requirements.txt
```

Requer: `numpy` e `pandas` (ver versões em `requirements.txt`).

## Estrutura do repositório

```
forex_indicators/        # Pacote principal (implementações e exportações)
examples/
  ├── example_usage.py  # Script de exemplo com dados sintéticos
  └── outputs/          # Artefatos gerados pelos exemplos (CSV, etc.)
notebooks/               # Notebooks de visualização e mini backtests
tests/                   # Testes automatizados
requirements.txt         # Dependências mínimas de runtime
LICENSE                  # Licença
README.md                # Este arquivo
```

## Uso rápido

```python
import pandas as pd
from forex_indicators import ema, rsi, macd

# df deve conter colunas: open, high, low, close, volume (minúsculas)
ema_20 = ema(df["close"], 20)
rsi_14 = rsi(df["close"], 14)
macd_df = macd(df["close"], 12, 26, 9)
```

Para ver vários indicadores sendo calculados de uma vez, execute o exemplo completo:

```bash
python examples/example_usage.py
# saída CSV será escrita em examples/outputs/indicators_output.csv
```

## Indicadores incluídos (amostra)

- Médias móveis: SMA, EMA, WMA, HMA, DEMA, TEMA, TRIMA, KAMA
- Osciladores: RSI, Stochastic (%K/%D), Williams %R, TRIX, CMO, TSI, Stochastic RSI, Ultimate Oscillator, Fisher Transform, SMI, CRSI
- Volatilidade/Bandas: Bollinger Bands (+ %B e Bandwidth), Keltner Channels, Donchian, Chandelier Exit, Ulcer Index, Z-Score
- Tendência/Direcionalidade: MACD (linha/sinal/hist), ADX (+DI/-DI), Aroon, Vortex, Ichimoku, Parabolic SAR, Moving Average Envelopes
- Volume/Fluxo: OBV, MFI, VWAP, CMF, ADL, Chaikin Oscillator, KVO, PVO
- Outros: ROC, Pivot Points, KST, BOP, Coppock Curve, Ease of Movement, Force Index, Elder Ray, Mass Index, Qstick, RVI, Alligator, Gator Oscillator, Fractals

A lista completa está disponível via `from forex_indicators import ...` e no arquivo `forex_indicators/__init__.py`.

## Contratos de entrada/saída

- As funções aceitam `pandas.Series` para preços/volumes ou `pandas.DataFrame` (quando aplicável), com colunas em minúsculas.
- As séries retornam `NaN` iniciais conforme o tamanho da janela (comportamento esperado).
- O cálculo segue definições clássicas; pequenas diferenças podem ocorrer por arredondamento/inicialização.

## Compatibilidade de dados

- As funções aceitam `Series`/`DataFrame` com colunas em minúsculas.
- Para dados vindos de provedores diversos (ex.: `Close`, `Adj Close`, `VOL`, `Last`), use o helper:

```python
from forex_indicators import normalize_ohlcv

df = normalize_ohlcv(df)
# Agora df terá colunas padronizadas: open/high/low/close/volume
```

Isso aumenta a interoperabilidade sem exigir renomeações manuais.

## Notebooks

- `notebooks/forex_visualizations_v2.ipynb`: carrega EURUSD via `yfinance` (se disponível) ou gera dados sintéticos; plota EMA, Bollinger, Keltner, Supertrend, ADX, ATR etc.; inclui 2 mini backtests (EMA 20/50 e Supertrend).
- Há também `notebooks/forex_visualizations.ipynb` como versão anterior/simplificada.

Abra em seu ambiente Jupyter/VS Code. Se `yfinance` não estiver instalado, o notebook usará dados sintéticos.

## Testes

```bash
python -m pytest -q
```

Os testes cobrem a correção básica das fórmulas e a integridade das saídas. Contribuições com novos testes são bem-vindas.

## Boas práticas e performance

- Utilize dados com índices ordenados cronologicamente.
- Prefira colunas minúsculas (`open/high/low/close/volume`).
- Em séries curtas, espere `NaN` nos primeiros `period` elementos.
- Operações vetorizadas em `pandas/numpy` priorizam legibilidade; para volumes massivos, avalie otimizações conforme a necessidade.

## Contribuindo

- Abra issues com descrições claras do problema/requisição.
- Ao propor novos indicadores, inclua referência da fórmula e testes.
- Padrões sugeridos: PEP8, nomes explícitos, funções puras, validação de entradas, retornos `Series/DataFrame` consistentes.

## Licença

Este projeto é licenciado sob os termos da licença incluída em `LICENSE`.