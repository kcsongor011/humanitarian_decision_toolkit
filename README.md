# Humanitarian Decision Toolkit

Operational decision-support tools for humanitarian settings using probability, statistics, simulation, forecasting, and transparent uncertainty modelling.

This is a living project. The aim is to build small, interpretable notebooks and helper modules that show how uncertainty can be made explicit in humanitarian planning decisions.

## Arrival Dynamics

Forecasting and simulation of refugee arrivals and operational flows under uncertainty.

### Version 1 — Short-term border arrivals

Implemented:

- [Open in Colab](https://colab.research.google.com/github/kcsongor011/humanitarian_decision_toolkit/blob/main/notebooks/forecasting/arrival_dynamics_1_short_term_border_arrivals.ipynb)
- [View notebook source on GitHub](notebooks/forecasting/arrival_dynamics_1_short_term_border_arrivals.ipynb)

This notebook follows a synthetic 30-day border-arrivals scenario through progressive reveal, short-term probabilistic forecasting, preparedness translation, registration-capacity analysis, backlog risk, and final operational synthesis.

The notebook uses synthetic data and simplified assumptions. It is intended to demonstrate the decision logic, not to model a real emergency operation.

### Version 2 — Country-level longer-term asylum applications

Implemented:

- [Open in Colab](https://colab.research.google.com/github/kcsongor011/humanitarian_decision_toolkit/blob/main/notebooks/forecasting/arrival_dynamics_2_long_term_country_level_arrivals.ipynb)
- [View notebook source on GitHub](notebooks/forecasting/arrival_dynamics_2_long_term_country_level_arrivals.ipynb)

This notebook uses monthly first-time asylum application data for Italy from Eurostat to demonstrate a longer-term country-level forecasting workflow. It covers exploratory time-series analysis, train/test evaluation, naive baselines, SARIMA model selection, a SARIMAX extension with descriptive event-period indicators, and a 12-month projection.

The notebook uses a real administrative time series, but it is intended as a transparent modelling and decision-support demonstration rather than an operational forecast product.

## Other planned components

### Population Estimation Under Uncertainty

Sampling, overlap estimation, and approaches for incomplete or uncertain humanitarian datasets.

### Inference & Explanation

Application of regression and statistical learning methods to humanitarian and human rights problems.

## Stack

- Python
- Jupyter / Colab
- pandas
- NumPy
- matplotlib
- statsmodels

## How to use

The notebooks can be run locally in Jupyter or VS Code after cloning the repository. Public notebooks can also be opened in Google Colab from the links above.

Notebook outputs are intentionally cleared in the repository version so readers can execute the notebooks themselves.

GitHub’s notebook preview may fail for larger notebooks. For the most reliable experience, open the notebook in Colab and run it there.

## Current status

Version 1 and Version 2 of the Arrival Dynamics module are implemented and linked above.

Additional toolkit components are planned for population estimation under uncertainty and inference/explanation workflows.
