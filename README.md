# Indicadores Técnicos para Forex (Python)

Este pacote oferece implementações simples e transparentes, em pandas/numpy, dos 20 principais indicadores técnicos amplamente usados no mercado Forex.

## Instalação

```bash
pip install -r requirements.txt
```

## Estrutura

- `forex_indicators/indicators.py`: implementação dos indicadores
- `forex_indicators/__init__.py`: exporta funções para importação direta
- `examples/example_usage.py`: exemplo de uso com dados sintéticos

## Uso rápido

```python
import pandas as pd
from forex_indicators import ema, rsi, macd

# df deve conter colunas: open, high, low, close, volume
ema_20 = ema(df["close"], 20)
rsi_14 = rsi(df["close"], 14)
macd_df = macd(df["close"], 12, 26, 9)
```

## Lista dos 20 indicadores

1. SMA (Simple Moving Average)
2. EMA (Exponential Moving Average)
3. WMA (Weighted Moving Average)
4. MACD (Moving Average Convergence Divergence)
5. RSI (Relative Strength Index)
6. Stochastic Oscillator (%K, %D)
7. Bollinger Bands
8. ATR (Average True Range)
9. ADX (+DI, -DI, ADX)
10. CCI (Commodity Channel Index)
11. OBV (On-Balance Volume)
12. MFI (Money Flow Index)
13. Parabolic SAR
14. Ichimoku Cloud
15. Pivot Points (Clássico)
16. VWAP (Volume Weighted Average Price)
17. ROC (Rate of Change)
18. Williams %R
19. TRIX
20. Donchian Channels

## Detalhes e parâmetros

- SMA: média simples de `series` em `period`.
- EMA: média exponencial com `span=period`.
- WMA: média ponderada, pesos lineares mais recentes maiores.
- MACD: `ema(fast) - ema(slow)`, linha de sinal = ema do MACD, hist = macd - sinal.
- RSI (Wilder): usa médias móveis de Wilder de ganhos e perdas.
- Stochastic: `%K = 100*(close - LL_n)/(HH_n - LL_n)`, `%D` = SMA de `%K`.
- Bollinger: média `SMA(period)` e desvio padrão; bandas `± num_std * std`.
- ATR (Wilder): média de `True Range` com suavização de Wilder (`alpha=1/period`).
- ADX: calcula +DI, -DI e ADX (suavização de Wilder).
- CCI: `(TP - SMA(TP)) / (0.015 * mean_dev)` com `TP=(H+L+C)/3`.
- OBV: soma cumulativa do volume ponderado pelo sinal da variação do preço.
- MFI: fluxo monetário positivo/negativo baseado em `TP`, retorna índice 0–100.
- Parabolic SAR: implementação clássica com `step` e `max_step`.
- Ichimoku: Tenkan(9), Kijun(26), Senkou A/B (shift 26), Chikou (shift -26).
- Pivot Points: pontos P, S1–S3, R1–R3 (fórmulas clássicas).
- VWAP: preço médio ponderado pelo volume cumulativo (não reseta por sessão).
- ROC: retorno relativo a `period` atrás; se `as_percent=True`, em %.
- Williams %R: oscilador em [-100, 0].
- TRIX: ROC de 1 período da EMA tripla.
- Donchian: maior alta e menor baixa da janela, com linha média.

## Exemplo completo

Execute o exemplo com dados sintéticos:

```bash
python examples/example_usage.py
```

Isso irá salvar `examples/indicators_output.csv` com todas as colunas calculadas.

## Observações

- Para séries com poucos dados, os primeiros valores serão `NaN` devido às janelas.
- Resultados podem variar marginalmente de outras bibliotecas por diferenças de arredondamento e inicialização.
- Todas as funções operam sobre `pandas.Series` e retornam `Series` ou `DataFrame` conforme indicado.

## +20 indicadores adicionados

21. Keltner Channels
22. Supertrend
23. PPO (Percentage Price Oscillator)
24. KAMA (Kaufman Adaptive Moving Average)
25. TSI (True Strength Index)
26. DPO (Detrended Price Oscillator)
27. Aroon (Up/Down/Oscillator)
28. CMF (Chaikin Money Flow)
29. Chaikin Oscillator
30. ADL (Accumulation/Distribution Line)
31. Elder Ray Index (Bull/Bear Power)
32. Force Index
33. Ultimate Oscillator
34. Ease of Movement (EOM)
35. Mass Index
36. Qstick
37. Vortex Indicator (VI+/VI-)
38. Stochastic RSI
39. PVO (Percentage Volume Oscillator)
40. KST (Know Sure Thing)
41. BOP (Balance of Power)

### Breve descrição dos novos
- Keltner: EMA ± ATR*multiplicador
- Supertrend: bandas por ATR com lógica de tendência (±1)
- PPO/PVO: osciladores percentuais de preço/volume + sinal + hist
- KAMA: média adaptativa de Kaufman
- TSI: dupla EMA do momentum e da magnitude
- DPO: remove tendência comparando close deslocado e SMA
- Aroon: proximidade da última máxima/mínima
- CMF/ADL/Chaikin Osc: fluxo monetário e sua variação
- Elder Ray: bull/bear power relativos à EMA
- Force Index: momentum ponderado por volume (bruto e EMA)
- Ultimate: combinações de BP/TR em 3 janelas
- EOM: facilidade de movimento normalizado pelo volume
- Mass Index: razão de EMAs do range somada
- Qstick: SMA de close - open
- Vortex: direcionalidade baseada em VMs e ATR
- StochRSI: estocástico aplicado ao RSI
- KST: somatória ponderada de ROCs suavizados
- BOP: relação de força de fechamento vs abertura no range

## +outros 20 indicadores

42. HMA (Hull Moving Average)
43. DEMA (Double EMA)
44. TEMA (Triple EMA)
45. TRIMA (Triangular MA)
46. CMO (Chande Momentum Oscillator)
47. Fisher Transform (Ehlers)
48. Coppock Curve
49. Connors RSI (CRSI)
50. SMI (Stochastic Momentum Index)
51. RVI (Relative Vigor Index)
52. Alligator (Jaw/Teeth/Lips)
53. Gator Oscillator
54. Fractals (Bill Williams)
55. Bollinger %B
56. Bollinger Bandwidth
57. Chandelier Exit (long/short)
58. KVO (Klinger Volume Oscillator)
59. Moving Average Envelopes
60. Ulcer Index
61. Z-Score do preço

## Notebook de visualização e mini backtest

Um notebook foi criado em `notebooks/forex_visualizations_v2.ipynb` que:
- Carrega EURUSD via `yfinance` (se disponível) ou gera dados sintéticos.
- Plota EMA(20/50), Bandas de Bollinger, Keltner, Supertrend, ADX, ATR e outros.
- Executa dois mini backtests: cruzamento de médias (EMA 20/50) e Supertrend.

Como abrir:
- No seu ambiente com Jupyter/VS Code, abra `notebooks/forex_visualizations_v2.ipynb`.
- Se não tiver `yfinance`, o notebook usará dados sintéticos automaticamente.