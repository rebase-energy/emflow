import importlib
import pkgutil
import os
from pathlib import Path
import sys

def list_problems():
    problems = []
    base = "emflow.examples"
    
    try:
        # Get the correct examples path
        import emflow
        examples_path = [str(Path(emflow.__file__).parent / "examples")]
        
        # Iterate through modules in the examples directory
        for finder, modname, ispkg in pkgutil.iter_modules(examples_path):
            # Skip the data directory
            if modname == 'data':
                continue
            
            if ispkg:
                try:
                    module = importlib.import_module(f"{base}.{modname}.problem")
                    if hasattr(module, "list_problem_variants"):
                        for variant in module.list_problem_variants():
                            problems.append(f"{modname.replace('_', '-')}" + f":{variant}")
                    elif hasattr(module, "get_problem"):
                        problems.append(f"{modname.replace('_', '-')}")
                except Exception as e:
                    print(f"Error importing {modname}.problem: {type(e).__name__} - {str(e)}")
                    import traceback
                    print(traceback.format_exc())
    except Exception as e:
        print(f"Error: {e}")
        return []
        
    return sorted(problems)

def load_problem(name):
    if ":" in name:
        folder, variant = name.split(":")
    else:
        folder, variant = name, None

    modname = folder.replace("-", "_")  # match filesystem
    module = importlib.import_module(f"emflow.examples.{modname}.problem")

    if variant:
        func = getattr(module, f"get_problem_{variant}")
    else:
        func = getattr(module, "get_problem")

    return func()
