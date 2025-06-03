# handlers/__init__.py

import importlib
import inspect
import os

# Dynamically import all modules in the handlers directory
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith("_handler.py") and filename != "__init__.py":
        module_name = filename[:-3]  # Remove .py extension
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Import classes that match the naming convention
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith("Handler") or name.endswith("Config"):
                globals()[name] = obj
