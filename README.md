# Reservoir Routing in Ephemeral (Arid-Region) Streams — Generalized Empty/Full Model

A semi-analytical, empty-or-full reservoir routing model driven by the
Williams–Hann inflow hydrograph, built on top of Kamis, Bahrawi & Elfeki
(2018), *Arab. J. Geosci.* 11:106 (the "reference paper").

## Files

| File | What it is |
|---|---|
| [`theory_paper.md`](theory_paper.md) | The write-up: full derivation, validation, and results. Start here. |
| [`theory_paper.docx`](theory_paper.docx) | The same paper as a Word document. |
| [`reservoir_routing_colab.ipynb`](reservoir_routing_colab.ipynb) | Google Colab notebook, pre-run. Reproduces every number and plot in the paper. **[Open in Colab](https://colab.research.google.com/github/asepmhidayatulloh/Flood-Routing-Arid-Dam/blob/main/reservoir_routing_colab.ipynb)** — update the repo name in this link if you use a different one. |
| [`routing_core.py`](routing_core.py) | The core engine as a standalone, importable module (numerical benchmark + semi-analytical solution). |
| [`index.html`](index.html) | Interactive browser tool — adjust reservoir/outlet/inflow parameters and watch both routing curves recompute live. Enable **GitHub Pages** (Settings → Pages → deploy from `main` / root) to serve it directly. |
| `routing_summary.csv` | The Table 1/2 numbers from the paper, as CSV. |

## Quick start

- **Just want to explore the model?** Open `index.html` (or the GitHub Pages
  link once deployed) — no installation needed.
- **Want to reproduce the paper or adapt it to a real dam?** Open
  `reservoir_routing_colab.ipynb` in Google Colab (Runtime → Run all), then
  edit the reservoir/outlet/inflow parameters in Part 4.
- **Want to use the engine in your own code?** `from routing_core import *`
  — see the docstrings in `routing_core.py`.

## Scope and status

This is a first-generation model built on **synthetic** Williams-Hann
hydrographs (no real inflow data or site survey has been substituted yet —
see the theory paper's Section 8 for what's needed to apply it to a real
dam). The full-reservoir semi-analytical solution matches the numerical
benchmark well (~5–13% of peak). The empty-reservoir solution is accurate
when the flood never reaches the spillway crest, and has a documented,
flagged limitation when the crest is reached before the inflow has passed
its own peak — see the theory paper, Sections 4.6 and 5.3, for the exact
validity condition and honest error numbers.

## Related project

[**Storm Hydrograph — a Raindrop's Journey to the Gauge**](https://asepmhidayatulloh.github.io/Rainfall-journey-to-Outlet/)
— a separate, general-purpose φ-index / Nash-cascade rainfall-to-runoff
simulator (not dam-specific). [Repository](https://github.com/asepmhidayatulloh/Rainfall-journey-to-Outlet).

