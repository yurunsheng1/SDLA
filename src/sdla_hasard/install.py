"""Register SDLA and its safety critic/buffers in a pinned HASARD checkout."""
import argparse
from pathlib import Path
import subprocess

UPSTREAM_COMMIT = "af9165f72ec6da96ffb60f0a9d3baebb8068c3a8"
RELATIVE_WORKER = "sample_factory/algo/learning/learner_worker.py"
IMPORT = "from sdla_hasard.learner import SDLALearner"


def desired_text(name, original):
    if name == RELATIVE_WORKER:
        marker = "        if cfg.algo == 'PPOLag':"
        if original.count(marker) != 1:
            raise RuntimeError("Unsupported learner-selection code")
        text = original.replace("from __future__ import annotations",
                                "from __future__ import annotations\n\n" + IMPORT, 1)
        return text.replace(marker, "        if cfg.algo == 'PPOCvx':\n            learner_cls = SDLALearner\n        elif cfg.algo == 'PPOLag':", 1)
    if name.endswith("actor_critic.py"):
        old = 'elif cfg.algo in ["PPOLag", "PPOPID", "P3O"]:'
        new = 'elif cfg.algo in ["PPOLag", "PPOPID", "P3O", "PPOCvx"]:'
    else:
        old = "if algo in ['PPOLag', 'TRPOLag', 'TRPOPID', 'PPOPID', 'CPO', 'P3O']:"
        new = "if algo in ['PPOLag', 'TRPOLag', 'TRPOPID', 'PPOPID', 'CPO', 'P3O', 'PPOCvx']:"
    if original.count(old) != 1:
        raise RuntimeError(f"Unsupported safety-model registration in {name}")
    return original.replace(old, new, 1)


def install(root):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"Expected HASARD commit {UPSTREAM_COMMIT}; found {head}")
    paths = [RELATIVE_WORKER, "sample_factory/model/actor_critic.py",
             "sample_factory/algo/utils/shared_buffers.py"]
    pending = []
    for name in paths:
        path = root / name
        original = subprocess.check_output(["git", "-C", str(root), "show", f"HEAD:{name}"], text=True)
        desired, current = desired_text(name, original), path.read_text()
        if current == desired:
            continue
        if current != original:
            raise RuntimeError(f"Unrelated changes in {name}; use a clean checkout")
        pending.append((path, desired))
    for path, text in pending:
        path.write_text(text)
    return bool(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hasard-root", type=Path, required=True)
    args = parser.parse_args()
    changed = install(args.hasard_root)
    print("SDLA learner, safety critic and buffers registered" if changed else "SDLA already registered")


if __name__ == "__main__":
    main()
