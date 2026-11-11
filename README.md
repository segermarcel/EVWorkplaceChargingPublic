# EVWorkplaceChargingPublic

Code for https://ev-workplace-charging.streamlit.app/

The repository contains the original Streamlit application and a static version
of the paper's precomputed scenarios, ready for GitHub Pages. Neither the
Streamlit URL nor the public site is changed by building locally.

## Local setup

1. Install dependencies with uv

Install uv by following the install instructions for your OS on their [website](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

2. Download input data Excel file and put it under `data/input_data`

3. Install gurobi solver: [Getting Started with Gurobi Optimizer](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer)

## Run streamlit dashboard locally

```bash
uv run streamlit run ev_workplace_charging/dashboard.py
```

The dashboard renders charts in memory and releases them after every render,
including interrupted renders. It no longer exports publication-resolution PNGs
on every interaction. Explicit exports remain available by passing a path to
`save_and_write_fig(fig, path)`.
Legends are positioned directly with Matplotlib to avoid an upstream
Seaborn/Matplotlib introspection cache retaining old figures.

## Static dashboard

Build and preview from the repository root:

```bash
uv sync --frozen
uv run --frozen python scripts/build_pages.py
uv run --frozen python -m http.server 8765 --bind 127.0.0.1 --directory build/pages
```

Open http://localhost:8765/. Use an HTTP server; opening `index.html` directly
does not allow the browser to fetch the scenario data.

The static dashboard needs no Python server, database, API keys or solver after
deployment. It includes all precomputed runs of 1-7 February, daily and weekly views,
relative and absolute metrics, electricity/carbon overlays, shareable selection
URLs and CSV profile downloads. Shift and battery assumptions are fixed; this
version does not run new optimizations. The existing Streamlit input behavior is
outside the scope of this migration.

Plots are PNGs generated during the build by the same Matplotlib/Seaborn functions
as Streamlit, with the same theme, axes, colors, labels, legends and 100-DPI export.
There is no separate browser implementation of the plots. The page also uses
Streamlit's Source Sans font (bundled locally under its SIL Open Font License).
Numeric tables and downloads are available under each plot's "Values & download"
section; clicking a plot opens the full-size image.

A full build renders 3,080 plot variants. It takes several minutes the first time
and reuses unchanged images on subsequent builds. Use `--jobs 4` to control the
number of rendering processes. Generated PNGs and their cache remain under the
ignored `build/` directory, not in Git. The browser loads only the six images for
the selected scenario.

Weekly plots show the mean daily profile, not a continuous seven-day timeline.
Weekly metrics average the daily absolute or percentage changes, matching the
original dashboard. Negative values indicate a reduction. Weekly bands use the
same Seaborn 95% percentile bootstrap (1,000 resamples), with seed 42 shared by
the local Streamlit and static renderers to make plots reproducible. The deployed
Streamlit version predating this change still uses a random seed and can therefore
show slightly different weekly bands. Absolute costs are in euros (day-ahead
prices in EUR/MWh), emissions in kg CO₂, and peak changes in kW.

The build validates every expected result/metric pair before writing the site.
It publishes only the browser assets and processed scenario results, excluding
the raw research inputs. All asset paths are relative, so the same build works
under the repository path used by GitHub Pages.

## Publishing after the paper review

1. Review and merge the changes into the repository used for deployment. A push
   to the branch watched by Streamlit also deploys the memory fix there.
2. In GitHub **Settings → Pages**, select **GitHub Actions** as the publishing
   source.
3. In **Actions → Publish GitHub Pages**, run the workflow on `main`. It validates
   the data, builds the site and deploys the resulting artifact.
4. Verify the URL shown by the deployment before changing links in the paper.
   For the current upstream repository the expected address is
   https://segermarcel.github.io/EVWorkplaceChargingPublic/.

Publishing is deliberately manual during the paper review. The Checks workflow
builds and tests changes without publishing; it does not replace the existing
Streamlit URL. Keep the old address available during the transition.

## Validation

```bash
uv sync --frozen --group dev
uv run --frozen pytest -q
uv run --frozen pre-commit run --all-files
node --check web/app.js
```

Tests cover repeated Streamlit renders, cleanup on render failure, complete CSV
coverage, numerical parity of all scenario profiles and metrics, and PNG byte
parity against Streamlit's actual image renderer. CI runs the hooks, tests and
static build. Dependabot checks Python dependencies and GitHub Actions weekly.
