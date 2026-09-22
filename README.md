# Nav Arena

A lightweight, modular navigation evaluation framework built on top of NVIDIA Isaac Sim 6.1.0 and Isaac Lab 3.0.0.

## Overview
`nav_arena` provides a benchmarking and evaluation testbed for autonomous navigation policies. It supports:
- **Modular Embodiments:** Extensible robot base and sensor definitions (Nova Carter, wheeled, legged).
- **Scene Ingestion:** Native OpenUSD scene loading (e.g., InteriorAgent).
- **ROS 2 Integration:** Passive simulation execution supporting external Nav2 stacks via standard ROS 2 interfaces.
- **Task Evaluation:** Modular task definitions and metrics for navigation benchmarks.

## Repository Structure
```text
nav_arena/
├── pyproject.toml
├── README.md
└── nav_arena/
    ├── embodiments/     # Robot configurations and sensor factories
    ├── scenes/          # Scene and environment configurations
    ├── tasks/           # Navigation tasks, MDP terms, and evaluation metrics
    └── scripts/         # Execution scripts and runners
```

## Installation
In your activated Isaac Lab Python environment:
```bash
pip install -e .
```
