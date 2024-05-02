import os
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
    for line in output.decode().split("\n"):
        if line.startswith("Result: "):
            return line[8:]
    return output.decode()


def multi_run(
    which: str, lazy: bool, round: int = 3, interval: float = None
) -> Tuple[float, float]:
    results = []
    for _ in range(round):
        results.append(float(generate_output(which, lazy)))
        if interval:
            time.sleep(interval)
    return np.mean(results), np.std(results)


if __name__ == "__main__":
    print("Type, Avg, Std")
    python_no_lazy = multi_run("python", False)
    print(f"Python (w/ partial), {python_no_lazy[0]}, {python_no_lazy[1]}")
    python_lazy = multi_run("python", True)
    print(f"Python (w/o partial), {python_lazy[0]}, {python_lazy[1]}")
    search_no_lazy = multi_run("search", False, interval=10)
    print(f"Search (w/ partial), {search_no_lazy[0]}, {search_no_lazy[1]}")
    search_lazy = multi_run("search", True, interval=10)
    print(f"Search (w/o partial), {search_lazy[0]}, {search_lazy[1]}")
