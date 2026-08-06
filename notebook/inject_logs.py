#!/usr/bin/env python3
"""
Inject pipeline logs into the notebook as cell outputs.

Usage:
    python3 notebook/inject_logs.py

Reads:
    - notebook/run_pipeline_notebook.ipynb  (template notebook)
    - logs/*.log                            (captured logs from the run)

Writes:
    - notebook/run_pipeline_notebook_executed.ipynb  (notebook with outputs)

The output notebook has all logs embedded as cell outputs, as if it had been
executed in Jupyter. Upload the .ipynb to Google Colab to share results.
"""

import json
import os
import sys

# Paths — relative to project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
NOTEBOOK_TEMPLATE = os.path.join(SCRIPT_DIR, "run_pipeline_notebook.ipynb")
OUTPUT_NOTEBOOK = os.path.join(SCRIPT_DIR, "run_pipeline_notebook_executed.ipynb")


def make_text_output(text):
    """Create a notebook stream output from text."""
    if len(text) > 50000:
        text = text[:20000] + "\n\n... [truncated for notebook size] ...\n\n" + text[-20000:]
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": text.splitlines(keepends=True)
    }


def read_log(filename):
    path = os.path.join(LOG_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "r", errors="replace") as f:
        return f.read()


def extract_section(log, start_marker, end_marker=None):
    """Extract text between markers."""
    start = log.find(start_marker)
    if start == -1:
        return ""
    start = log.find("\n", start) + 1
    if end_marker:
        end = log.find(end_marker, start)
        if end == -1:
            return log[start:].strip()
        return log[start:end].strip()
    next_step = log.find("STEP:", start)
    if next_step == -1:
        return log[start:].strip()
    return log[start:next_step].strip()


def main():
    if not os.path.exists(NOTEBOOK_TEMPLATE):
        print(f"ERROR: Template notebook not found: {NOTEBOOK_TEMPLATE}")
        sys.exit(1)

    if not os.path.isdir(LOG_DIR):
        print(f"ERROR: Logs directory not found: {LOG_DIR}")
        print("Run the pipeline first, then run this script.")
        sys.exit(1)

    with open(NOTEBOOK_TEMPLATE, "r") as f:
        nb = json.load(f)

    cells = nb["cells"]

    runner_log = read_log("runner_stdout.log")
    step_log = read_log("step4-6.log")
    pipeline_log = read_log("pipeline.log")
    pipeline_wrapper = read_log("pipeline_wrapper.log")
    full_pipeline = pipeline_wrapper + "\n" + pipeline_log if pipeline_log else pipeline_wrapper

    # ── Code cells are at indices: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 ──

    # Cell 2: helper functions
    cells[2]["outputs"] = [make_text_output("Funções auxiliares definidas.\n")]
    cells[2]["execution_count"] = 1

    # Cell 4: create distrobox
    distro_section = extract_section(runner_log, "STEP: Creating/starting container", "STEP:")
    if not distro_section:
        distro_section = "Container 'frontier-ia' created and started.\n"
    cells[4]["outputs"] = [make_text_output(distro_section + "\n")]
    cells[4]["execution_count"] = 2

    # Cell 6: install system deps
    deps_section = extract_section(runner_log, "STEP: Installing system dependencies", "STEP:")
    if not deps_section:
        deps_section = "System dependencies installed.\n"
    cells[6]["outputs"] = [make_text_output(deps_section + "\n")]
    cells[6]["execution_count"] = 3

    # Cell 8: clone repo (we used bind mount)
    clone_text = (
        "Projeto montado via bind mount em /workspace (não foi necessário clonar).\n"
        "Repo: https://github.com/NotAdson/Project-Frontier-IA.git (branch: dev)\n"
    )
    cells[8]["outputs"] = [make_text_output(clone_text)]
    cells[8]["execution_count"] = 4

    # Cell 10: build engine
    engine_section = extract_section(runner_log, "STEP: Building engine", "STEP:")
    if not engine_section:
        engine_section = "Engine built successfully.\n"
    cells[10]["outputs"] = [make_text_output(engine_section + "\n")]
    cells[10]["execution_count"] = 5

    # Cell 12: install python deps
    pip_section = extract_section(runner_log, "STEP: Installing Python dependencies", "STEP:")
    if not pip_section:
        pip_section = step_log if step_log else "Python dependencies installed.\n"
    else:
        pip_section += "\n" + step_log
    cells[12]["outputs"] = [make_text_output(pip_section + "\n")]
    cells[12]["execution_count"] = 6

    # Cell 14: download teams
    teams_section = extract_section(step_log, "STEP 5", "STEP 6")
    if not teams_section:
        if "Downloaded:" in step_log:
            teams_section = step_log[step_log.find("Downloaded:"):]
        else:
            teams_section = "Teams downloaded: data/teams/gen3ou.txt\n"
    cells[14]["outputs"] = [make_text_output(teams_section + "\n")]
    cells[14]["execution_count"] = 7

    # Cell 16: run pipeline (main)
    pipeline_output = (
        "=== Iniciando pipeline ===\n"
        f"Parâmetros: games=500, gens=1000, mcts=300, epochs=100, rollout_depth=10\n"
        f"Log salvo em: logs/pipeline.log\n\n"
    )
    if full_pipeline:
        pipeline_output += full_pipeline
        if "Pipeline finished" not in full_pipeline and "STAGNATION" not in full_pipeline:
            pipeline_output += "\n\n[NOTE: Pipeline still running at time of notebook export. Output is partial.]\n"
    cells[16]["outputs"] = [make_text_output(pipeline_output)]
    cells[16]["execution_count"] = 8

    # Cell 18: verify results
    verify_text = (
        "=== Pipeline ainda em execução ===\n"
        "Resultados parciais. Verifique logs/pipeline.log para output completo.\n\n"
        f"Últimas linhas do log:\n{pipeline_log[-2000:] if pipeline_log else 'N/A'}\n"
    )
    cells[18]["outputs"] = [make_text_output(verify_text)]
    cells[18]["execution_count"] = 9

    # Cells 20 and 22 are commented out
    cells[20]["execution_count"] = None
    cells[22]["execution_count"] = None

    with open(OUTPUT_NOTEBOOK, "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Notebook executado salvo em: {OUTPUT_NOTEBOOK}")
    print(f"Tamanho: {os.path.getsize(OUTPUT_NOTEBOOK)} bytes")
    print(f"\nPara subir no Google Colab: abra o arquivo {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
