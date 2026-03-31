"""Export reviewed training artifacts into the final_agents area.

This script does not overwrite the deployable agent by default.
It creates a transparent manifest and, optionally, a runtime template that can
later be wired into `final_agents/drone_agent.py` after review.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import shutil
from pathlib import Path
import sys

import torch

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_lib.common import ensure_dir, save_json
from training_lib.models import export_student_torchscript, load_checkpoint, load_model_config, smoke_test_student_policy


AGENT_TEMPLATE = '''"""Generated deploy-time agent wrapper.

Review this file before promoting it into production.
It is intentionally explicit and non-obfuscated.
This wrapper assumes `policy.pt` is a TorchScript runtime artifact.
"""

from __future__ import annotations

from pathlib import Path

try:
    import torch
except ImportError as exc:
    raise ImportError("This generated agent template requires PyTorch at runtime.") from exc


class DroneFlightController:
    def __init__(self):
        self.runtime_model_path = Path(__file__).resolve().parent / "exported" / "{export_name}" / "policy.pt"
        self._model = None
        self._device = "cpu"

    def reset(self):
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()

    def _lazy_load(self):
        if self._model is not None:
            return
        self._model = torch.jit.load(str(self.runtime_model_path), map_location=self._device)
        self._model.eval()

    def act(self, observation):
        self._lazy_load()
        with torch.no_grad():
            depth = torch.as_tensor(observation["depth"], dtype=torch.float32, device=self._device).unsqueeze(0)
            state = torch.as_tensor(observation["state"], dtype=torch.float32, device=self._device).unsqueeze(0)
            action = self._model(depth, state)
            action = action[0].detach().cpu().numpy()
        return action
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--student-model-config", type=Path, required=True)
    parser.add_argument(
        "--runtime-model",
        type=Path,
        default=None,
        help="Optional scripted/compiled runtime artifact for real packaged deployment. "
        "If provided, it will be copied as policy.pt and can be used by the generated wrapper.",
    )
    parser.add_argument("--residual-checkpoint", type=Path, default=None)
    parser.add_argument("--export-name", type=str, default="candidate_student")
    parser.add_argument("--write-agent-template", action="store_true")
    return parser.parse_args()


def resolve_checkpoint_path(path: Path) -> Path:
    if path.suffix == ".json":
        import json

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if "checkpoint_path" in payload:
            return Path(payload["checkpoint_path"])
    return path


def main() -> None:
    args = parse_args()
    export_root = ensure_dir(DEFAULT_MODEL_ROOT / "final_agents" / "exported" / args.export_name)
    resolved_student_checkpoint = resolve_checkpoint_path(args.student_checkpoint)
    student_model, _ = load_checkpoint(resolved_student_checkpoint, map_location="cpu")
    student_config = load_model_config(args.student_model_config)
    if asdict(student_model.config) != asdict(student_config):
        raise ValueError(
            "Student checkpoint config does not match the supplied student_model_config.json."
        )
    shutil.copy2(resolved_student_checkpoint, export_root / "student_checkpoint.pt")
    shutil.copy2(args.student_model_config, export_root / "student_model_config.json")
    if args.runtime_model is not None:
        shutil.copy2(args.runtime_model, export_root / "policy.pt")
        runtime_export = {"source": "provided_runtime_model"}
    else:
        runtime_export = export_student_torchscript(
            resolved_student_checkpoint,
            export_root / "policy.pt",
            map_location="cpu",
        )
    if args.residual_checkpoint:
        shutil.copy2(args.residual_checkpoint, export_root / "residual_landing.pt")

    smoke_runtime = torch.jit.load(str(export_root / "policy.pt"), map_location="cpu")
    smoke_runtime.eval()
    smoke = smoke_test_student_policy(smoke_runtime, state_dim=student_model.config.state_dim)

    manifest = {
        "export_name": args.export_name,
        "student_checkpoint": str(export_root / "student_checkpoint.pt"),
        "student_model_config": str(export_root / "student_model_config.json"),
        "runtime_model": str(export_root / "policy.pt"),
        "runtime_model_source": "provided" if args.runtime_model is not None else "exported_from_student_checkpoint",
        "residual_checkpoint": str(export_root / "residual_landing.pt") if args.residual_checkpoint else None,
        "runtime_export": runtime_export,
        "runtime_smoke": smoke,
        "review_required": True,
        "note": "This export is transparent by design. For real packaged submission, prefer a self-contained top-level "
        "drone_agent.py plus a scripted runtime artifact such as policy.pt.",
    }
    save_json(export_root / "export_manifest.json", manifest)
    save_json(export_root / "export_smoke.json", smoke)

    if args.write_agent_template:
        template_path = export_root / "drone_agent_template.py"
        template_path.write_text(AGENT_TEMPLATE.format(export_name=args.export_name), encoding="utf-8")
        print(f"Wrote runtime template to {template_path}")

    print(f"Exported artifacts to {export_root}")


if __name__ == "__main__":
    main()
