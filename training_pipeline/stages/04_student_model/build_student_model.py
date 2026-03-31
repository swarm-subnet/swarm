"""Create the deployable student architecture and an initial checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (DEFAULT_MODEL_ROOT, REPO_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from training_lib.common import ensure_dir, save_json, seed_everything
from training_lib.models import (
    StudentModelConfig,
    StudentPolicy,
    build_student_runtime,
    export_student_torchscript,
    save_checkpoint,
    save_model_config,
    smoke_test_student_policy,
    validate_student_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_ROOT / "artifacts" / "04_student_model")
    parser.add_argument("--state-dim", type=int, default=141)
    parser.add_argument("--teacher-state-dim", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--gru-hidden-dim", type=int, default=256)
    parser.add_argument("--disable-aux-heads", action="store_true")
    parser.add_argument("--random-seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.random_seed)
    ensure_dir(args.output_dir)

    config = StudentModelConfig(
        state_dim=args.state_dim,
        teacher_state_dim=args.teacher_state_dim,
        hidden_dim=args.hidden_dim,
        gru_hidden_dim=args.gru_hidden_dim,
        use_aux_heads=not args.disable_aux_heads,
    )
    validate_student_contract(config)
    model = StudentPolicy(config)

    smoke = smoke_test_student_policy(model, state_dim=config.state_dim)

    save_model_config(args.output_dir / "student_model_config.json", config)
    save_checkpoint(
        args.output_dir / "student_init.pt",
        model,
        extra={"stage": "04_student_model", "note": "Randomly initialized student checkpoint."},
    )
    runtime = build_student_runtime(model)
    runtime_path = args.output_dir / "student_runtime_preview.pt"
    runtime_smoke = smoke_test_student_policy(runtime, state_dim=config.state_dim)
    torchscript_summary = export_student_torchscript(
        args.output_dir / "student_init.pt",
        runtime_path,
    )
    exported_runtime = torch.jit.load(str(runtime_path), map_location="cpu")
    exported_smoke = smoke_test_student_policy(exported_runtime, state_dim=config.state_dim)
    save_json(
        args.output_dir / "student_model_summary.json",
        {
            "state_dim": config.state_dim,
            "teacher_state_dim": config.teacher_state_dim,
            "hidden_dim": config.hidden_dim,
            "gru_hidden_dim": config.gru_hidden_dim,
            "use_aux_heads": config.use_aux_heads,
            "smoke_test": smoke,
            "runtime_smoke_test": runtime_smoke,
            "exported_runtime_smoke_test": exported_smoke,
            "runtime_preview_path": str(runtime_path),
            "runtime_preview_export": torchscript_summary,
        },
    )
    save_json(args.output_dir / "student_smoke_test.json", smoke)
    print(f"Saved student config to {args.output_dir / 'student_model_config.json'}")
    print(f"Saved initial checkpoint to {args.output_dir / 'student_init.pt'}")
    print(f"Saved runtime preview to {runtime_path}")


if __name__ == "__main__":
    main()
