# GitHub Copilot Instructions for IRdetection in QTLab2425

## Focus Area

Only suggest and complete code related to the `IRdetection/` folder. Ignore `2Dqubit/` and `3Dqubit/` — we do not work in those directories.

## Project Purpose

This module is used for Infrared (IR) detection experiments in a quantum lab setting. It includes tools for configuring and running experiments, analyzing results, and visualizing data from resonator sweeps and TR measurements.

## Directory Overview

- `IRdetection/src/experiment/`
  - Contains core classes: `Experiment`, `Resonator`, `TRSwipe`
- `IRdetection/src/utils/`
  - Helper modules for data loading, file handling, math utilities, and plotting
- `IRdetection/Analysis/`
  - Scripts for analyzing and visualizing data (e.g. VNA fits)
- `IRdetection/Experiments/`
  - YAML and JSON files describing experimental setups
- `IRdetection/Tests/`
  - Organized tests for `experiment` and `analysis` modules

## Style Guide

- Follow **PEP8**
- Use **type hints** in all function definitions
- Use **docstrings** following the **Google style**
- Use **f-strings** for formatting
- Organize imports with **isort** and **black** formatting style

## Copilot Instructions

- Only suggest completions within the context of files in `IRdetection/`
- When writing new code:
  - For logic related to experiments: extend classes in `src/experiment/`
  - For data handling: use functions in `src/utils/data_loader.py` or `file_manager.py`
  - For math or fitting routines: reuse or extend `math_utils.py`
  - For plotting: use `plot_utils.py` or extend `plot_results.py`
- When writing tests:
  - Use `pytest`
  - Follow the structure in `Tests/test_analysis/` and `Tests/test_experiment/`

## Naming and Conventions

- Use **snake_case** for function and variable names
- Use **CamelCase** for class names
- Avoid generic names like `temp`, `data1`, etc. Use context-specific terms
- Prefer readability and clarity over brevity

## Behavior Expectations

- Suggest function implementations that:
  - Are modular and reusable
  - Include meaningful inline comments
  - Avoid hardcoded paths — use config loaders or arguments
- When completing experiment YAMLs, ensure proper nesting and match the structure of existing files
