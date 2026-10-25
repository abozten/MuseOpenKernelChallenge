"""Run every test module. python3 -m h200sim.test_all"""

import importlib
import time
import warnings

warnings.filterwarnings("ignore")

for name in ("test_sim", "test_calibration", "test_tensor", "test_ops", "test_system", "test_control", "test_litmus", "test_collectives"):
    mod = importlib.import_module(f".{name}", __package__)
    for fn_name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t0 = time.time()
        getattr(mod, fn_name)()
        print(f"PASS {name}.{fn_name}  ({time.time() - t0:.1f}s)")
