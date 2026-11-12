# MLP — classificação de emissão acústica (mestrado)

Experimentos de aprendizado de máquina do mestrado sobre dados de emissão
acústica (EA). Repositório só com fontes: datasets, ambientes Python e saídas
de ferramentas ficam fora (ver abaixo).

| Pasta | Conteúdo |
|---|---|
| `MOMEA/` | classificação das condições de uma bomba (8 frequências × 7 configurações de válvula) a partir dos HITs do MOMEA: MLP, Random Forest e Gradient Boosting (`mlp_ea.py`), figuras e artigo (`artigo/`, `ARTIGO_SIM_EMBARCADO-10.11.2026/`), implementação embarcada (`embarcado/`: C, HLS e simulação no MicroBlaze). Resumo em `RESUMO_RESULTADOS.md`; uso do embarcado em `embarcado/README.md` |
| `CWRU/` | diagnóstico de rolamentos com a base pública CWRU (48 kHz, drive end): `cwru_mlp.py`, resultados e relatório (`RELATORIO_CWRU-10.11.2026/`) |
| `IrisDataSet/` | o mesmo modelo MLP aplicado ao conjunto Iris de Fisher (`mlp_iris.py`, dados do próprio scikit-learn) e relatório técnico didático sobre perceptrons multicamadas (`artigo/`, LaTeX com figuras e tabelas geradas pela execução). Ver `IrisDataSet/README.md` |

## Ambiente

```bat
cd MOMEA
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe mlp_ea.py
```

(`CWRU/cwru_mlp.py` usa o mesmo ambiente: `..\MOMEA\.venv\Scripts\python.exe -I cwru_mlp.py`.)

## O que não está no repositório

| Item | Por quê / onde obter |
|---|---|
| `MOMEA/Dados/` (52 MB) | aquisições do MOMEA (56 corridas, `hits.csv` + `config.json` + `summary.json` por condição). **Dados originais do experimento, só no disco de trabalho — mantenha um backup.** Estrutura esperada em `RESUMO_RESULTADOS.md`, seção 1 |
| `CWRU/Dados/` (283 MB) | 41 arquivos `.mat` da base pública *Case Western Reserve University Bearing Data Center* (48 kHz, drive end) |
| `*/embarcado/gerado/*.bin` | vetores de teste gerados por `embarcado/exportar_hits.py` a partir de `Dados/` |
| `*/embarcado/hls/proj_*`, logs do HLS | saída do Vitis HLS (`run_hls.tcl` recria) |
| `*.elf`, `*.exe` | binários compilados da simulação/testes (`embarcado/teste_pc.bat`, `microblaze/sim/rodar_sim_mb.py`) |
| `.venv/`, `__pycache__/` | ambiente Python (recriar com `requirements.txt`) |
| `*.zip`, auxiliares de LaTeX | pacotes e intermediários gerados |

Os `.h` em `embarcado/gerado/` (pesos da rede, classes, configuração) **estão**
no repositório: são o modelo treinado, e sem `Dados/` não daria para recriá-los.
