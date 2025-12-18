# src/xs_utils.py
import json
import numpy as np
import os
import sys

# Prevent .pyc file generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

def is_1d_list(v):
    return isinstance(v, list) and all(not isinstance(i, (list, np.ndarray)) for i in v)

def check_xs(data):
    if data["geom_type"] == "1D":
        if data["solve_type"] == "forward" or data["solve_type"] == "adjoint":
            required = ["a", "N", "D", "TOT", "SIGS", "NUFIS", "CHI"]
        elif data["solve_type"] == "noise":
            required = ["a", "N", "D", "TOT", "SIGS", "NUFIS", "CHI", "dTOT", "dSIGS", "dNUFIS"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required variable: {key}")
        if not isinstance(data[key], list) or len(data[key]) != 1:
            raise ValueError(f"{key} must be a list of 1 group: [list]")
        if not is_1d_list(data[key][0]):
            raise ValueError(f"{key}[0] must be a 1-D list")

def save_to_json(data, json_path):
    """JSON cannot store numpy arrays; convert first."""
    serializable = {}
    for key, val in data.items():
        if isinstance(val, np.ndarray):
            serializable[key] = val.tolist()
        else:
            serializable[key] = val
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=4)