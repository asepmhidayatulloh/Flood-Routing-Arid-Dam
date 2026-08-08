# Reservoir Routing in Arid regions — Empty/Full Dam Model

A semi-analytical, empty-or-full reservoir routing model driven by the
Williams–Hann inflow hydrograph

## Quick start

- **Just want to explore the model?** Open `index.html` (or the GitHub Pages
  link once deployed) — no installation needed.
- **Want to use the engine in your own code?** `from routing_core import *`
  — see the docstrings in `routing_core.py`.

## 🌊 Live Simulation
[**Launch the Flood Routing – Arid Dam Simulation →**](https://asepmhidayatulloh.github.io/Flood-Routing-Arid-Dam/)

## Scope and status

This is a first-generation model built on **synthetic** Williams-Hann
hydrographs. The full-reservoir semi-analytical solution matches the numerical
benchmark well (~5–13% of peak). The empty-reservoir solution is accurate
when the flood never reaches the spillway crest, and has a documented,
flagged limitation when the crest is reached before the inflow has passed
its own peak.

