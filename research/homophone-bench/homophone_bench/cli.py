from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


COMMANDS = {
    "merge-generative": "merge_generative",
    "function-glue": "function_glue",
    "finalize": "finalize",
    "compose-lots": "compose_lots",
    "recursive-poet": "recursive_poet",
    "round-rabbit": "round_rabbit",
    "mapping-web": "mapping_web",
    "smoke": "smoke_test",
}


def run_module(module_name: str, args: list[str]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    sys.argv = [module_name, *args]
    module = importlib.import_module(module_name)
    module.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Homophone bench v5 release CLI.")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    run_module(COMMANDS[ns.command], ns.args)


if __name__ == "__main__":
    main()
