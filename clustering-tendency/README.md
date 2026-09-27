# Clustering tendency and entropy-based measures

CS-E4650 Methods of Data Mining homework: a visual and entropy-based study of
the clustering tendency of `data/clustering-tendency.csv` (1000
three-dimensional vectors with values in [0, 1]).

| Path | Content |
| --- | --- |
| `data/clustering-tendency.csv` | the data set |
| `code/clustering_tendency.py` | computes everything: data check, histograms, scatter plots, entropies (Aggarwal eq. 6.2, m = 64), greedy backward/forward selection, extra experiments |
| `figures/` | generated histograms and scatter plots (PDF) |
| `results/summary.txt`, `results/entropies.csv` | generated numbers |
| `results/latex/` | generated tables and number macros used by the report |
| `report/report.tex`, `report/report.pdf` | the report |

## Reproducing

```sh
pip install -r requirements.txt
python3 code/clustering_tendency.py      # figures/ and results/
cd report && latexmk -pdf report.tex     # report/report.pdf
```

Tested with Python 3.11, NumPy 2.4, Matplotlib 3.11 and TeX Live 2023.
The report reads all numbers from `results/latex/`, so re-running the script
updates the tables and the numbers in the text.

Before submitting, fill in the homework number (`\hwnumber`) and the team
members (`\author`) at the top of `report/report.tex`.
