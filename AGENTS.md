# Repository agent guide

Conventions for AI agents (Claude Code, Codex, etc.) working in this repo.
`CLAUDE.md` imports this file, so it applies to Claude Code as well.

## Reports & figures (`reports/`)

`reports/sample/` is the **template** for how reports are written here; mirror its
structure for new reports. The conventions below apply to **all reports**.

### Figure plotting conventions

- **Ask before labeling.** Whenever the user asks you to make / generate / update
  a figure, **first ask the user which labels to use** (axis labels, colorbar
  label, legend, and whether any title) and **confirm the choice before producing
  the figure**. Do not pick label text unilaterally.
- **Units are mandatory.** Every axis and colorbar label must carry its unit
  (e.g. `x (µm)`, `Height (nm)`, `dz/dx (nm/px)`). Never drop the unit, even when
  shortening a label.
- **Keep labels short and concise.** Capitalize the leading word of a label
  (e.g. `Height (nm)`); single-symbol axes like `x (µm)` / `y (µm)` stay bare.
- **No figure title** unless the user explicitly asks for one.
- **Font sizes (finalized):** axis label = 15, tick = 13 (colorbar label & ticks
  match; ~1.5× the matplotlib default).
- **Figures are vector PDF** (for crisp LaTeX inclusion via `pdflatex`).
- **Self-contained `images/` folder.** Each report's `images/` directory holds
  **both** the plotting script (`.py`) that regenerates the figures **and** the
  output `.pdf` files. The script should read the source data and **reuse the
  repo's existing processing modules** rather than reimplementing the analysis.
  See `reports/sample/images/make_figures.py` for the reference example (it reuses
  `export_leveled.masked_polyfit` and `measure_nid.pixel_nm` from `26 고물실 1조/`).

### LaTeX build

- `reports/sample/sample.tex` is REVTeX 4.2; the bibliography is embedded via a
  `filecontents` block (no external `.bib`).
- Build order: `pdflatex → bibtex → pdflatex → pdflatex` (see `.vscode/settings.json`).
- Build leftovers (`*.aux`, `*.bbl`, generated `apssamp.bib`, etc.) are gitignored
  under `reports/.gitignore`; the `.pdf` is tracked.

## General

- **Write code, comments, plots, and docs in English**, even though the repo is
  otherwise Korean.
