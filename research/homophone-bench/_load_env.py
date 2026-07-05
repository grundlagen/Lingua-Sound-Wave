"""Load .env.local (gitignored) for local work; Claude Code doesn't commit it."""
import os
from pathlib import Path

def load_keys():
    """Load API keys from .env.local (safe, gitignored)."""
    env_file = Path(__file__).parent / ".env.local"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ[key] = val
        return True
    return False

if __name__ == "__main__":
    if load_keys():
        print("✓ Keys loaded from .env.local")
    else:
        print("✗ .env.local not found")

