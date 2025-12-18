# src/runner.py
import sys
import os
import inspect
import importlib.util
from pathlib import Path
from cockatoo.input_reader import check_xs, save_to_json
import json
import numpy as np

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, set):
        return list(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj  # fallback

def save_to_json(data, json_path, parent_folder, case_name):
    parent_folder = Path(parent_folder)
    case_folder = parent_folder / case_name
    case_folder.mkdir(exist_ok=True)
    json_filename = Path(json_path).name
    final_json_path = case_folder / json_filename
    with open(final_json_path, "w") as f:
        json.dump(to_jsonable(data), f, indent=4)

def run():
    """
    Main driver function.
    Auto-detects the input file that called run() and executes workflow.
    """

    # -----------------------------
    # 1. Detect input file automatically
    # -----------------------------
    frame = inspect.stack()[1].frame
    module = inspect.getmodule(frame)
    print(f"[RUNNER] Detected input file: {module.__file__}")

    # -----------------------------
    # 2. Load input file variables
    # -----------------------------
    ALLOWED_TYPES = (int, float, str, bool, list, dict, tuple)
    variables = {
        k: v for k, v in module.__dict__.items()
        if not k.startswith("_") and isinstance(v, ALLOWED_TYPES)
    }

    print("Loaded variables from input script:")
    for k, v in variables.items():
        print(f"  {k} = {v}")

    # -----------------------------
    # 3. Validate cross sections
    # -----------------------------
    check_xs(variables)
    print("[RUNNER] Input checks passed.")

    # -----------------------------
    # 4. Save JSON
    # -----------------------------
    case_name = variables.get("case_name")
    json_file = Path(module.__file__).with_suffix(".json")
    current_folder = Path(module.__file__).parent

    save_to_json(variables, json_file, current_folder, case_name)

    print(f"[RUNNER] Saved JSON → {json_file}")

    # Now you can continue running your simulation
    return variables