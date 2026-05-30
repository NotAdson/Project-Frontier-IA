# Benchmarks

This directory contains the testing and benchmarking suites used to evaluate the performance of different Battle Agents against each other.

## Scripts
- **`run_benchmark.py`**: The main entry point to run a tournament between loaded agents.
- **`simulate.py`**: Runs a single match between two specific agents and generates a visually playable `replay.html`.

## Benchmark Types
- **`round_robin.py`**: Every agent plays every other agent `N` times to gather comprehensive win rate statistics.
- **`tournament.py`**: A single-elimination bracket-style tournament.
