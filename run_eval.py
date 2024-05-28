import sys
import subprocess as sp
import time
from typing import Tuple
import numpy as np


def generate_output(which: str, lazy: bool):
    args = ["python3", "main.py", which]
    if lazy:
        args.append("lazy")
    old_stdout = sys.stdout
    output = sp.run(args, stdout=sp.PIPE, stderr=sp.PIPE).stderr
    sys.stdout = old_stdout
    # find line starting with "Result: "
    res = []
    texts = output.decode().split("\n")
    for line in texts:
        if line.startswith("Result: "):
            res.append(line[8:])
            break
    if not res:
        raise ValueError("No output found")
    plugin_time = []
    for line in texts:
        if line.startswith("<PLUGIN_INFO>"):
            plugin_time.append(float(line.split(" ")[-1]))
    res.append(plugin_time)
    for line in texts:
        if line.startswith("<DECODE_INFO>"):
            res.append(float(line.split(" ")[-1]))
            break
    return res


def multi_run(
    which: str, lazy: bool, round: int = 5, interval: float = None
) -> Tuple[float, float]:
    results = []
    for _ in range(round):
        results.append(float(generate_output(which, lazy)[0]))
        if interval:
            time.sleep(interval)
    return np.mean(results), np.std(results)


def run_all():
    print("Type, Avg, Std")
    python_no_lazy = multi_run("python", False)
    print(f"Python (w/ partial), {python_no_lazy[0]}, {python_no_lazy[1]}")
    python_lazy = multi_run("python", True)
    print(f"Python (w/o partial), {python_lazy[0]}, {python_lazy[1]}")
    search_no_lazy = multi_run("search", False, interval=10)
    print(f"Search (w/ partial), {search_no_lazy[0]}, {search_no_lazy[1]}")
    search_lazy = multi_run("search", True, interval=10)
    print(f"Search (w/o partial), {search_lazy[0]}, {search_lazy[1]}")
    planning_no_lazy = multi_run("planning", False, interval=10)
    print(f"Planning (w/ partial), {planning_no_lazy[0]}, {planning_no_lazy[1]}")
    planning_lazy = multi_run("planning", True, interval=10)
    print(f"Planning (w/o partial), {planning_lazy[0]}, {planning_lazy[1]}")
    validation_no_lazy = multi_run("validation", False)
    print(f"Validation (w/ partial), {validation_no_lazy[0]}, {validation_no_lazy[1]}")
    validation_lazy = multi_run("validation", True)
    print(f"Validation (w/o partial), {validation_lazy[0]}, {validation_lazy[1]}")
    sqlite_no_lazy = multi_run("sqlite", False)
    print(f"SQLite (w/ partial), {sqlite_no_lazy[0]}, {sqlite_no_lazy[1]}")
    sqlite_lazy = multi_run("sqlite", True)
    print(f"SQLite (w/o partial), {sqlite_lazy[0]}, {sqlite_lazy[1]}")
    calculator_no_lazy = multi_run("calculator", False)
    print(f"Calculator (w/ partial), {calculator_no_lazy[0]}, {calculator_no_lazy[1]}")
    calculator_lazy = multi_run("calculator", True)
    print(f"Calculator (w/o partial), {calculator_lazy[0]}, {calculator_lazy[1]}")


def run_breakdown(which: str, lazy: bool, round: int, interval: float = None):
    for _ in range(round):
        results = generate_output(which, lazy)
        print(
            f"{which}, {'w/' if not lazy else 'w/o'} partial, {results[0]}, {results[2]}, {results[1]}",
            flush=True,
        )
        if interval:
            time.sleep(interval)


if __name__ == "__main__":
    for i in range(100):
        run_breakdown("search", True, 1)
        run_breakdown("search", False, 1)
        print(f"Finished {i}th round", file=sys.stderr)
    for i in range(100):
        run_breakdown("planning", True, 1)
        run_breakdown("planning", False, 1)
        print(f"Finished {i}th round", file=sys.stderr)
    for i in range(100):
        run_breakdown("python", True, 1)
        run_breakdown("sqlite", True, 1)
        run_breakdown("calculator", True, 1)
        run_breakdown("python", False, 1)
        run_breakdown("sqlite", False, 1)
        run_breakdown("calculator", False, 1)
        print(f"Finished {i}th round", file=sys.stderr)
