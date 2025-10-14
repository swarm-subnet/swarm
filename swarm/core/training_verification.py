#!/usr/bin/env python3

import sys
import os
import json
import gc
import ast
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile, BadZipFile
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from swarm.constants import (
    TRAINING_CODE_MAX_SIZE,
    TRAINING_CODE_MAX_FILES,
    VERIFICATION_TEST_TASKS,
    VERIFICATION_CORRELATION_THRESHOLD,
    VERIFICATION_MEAN_DIFF_THRESHOLD,
    VERIFICATION_PASS_THRESHOLD,
    SIM_DT,
    HORIZON_SEC,
)

FORBIDDEN_IMPORTS = {
    "subprocess", "os.system", "eval", "exec",
    "socket", "urllib", "requests", "__import__",
    "importlib.import_module", "pty", "atexit",
}

ALLOWED_EXTENSIONS = {".py", ".json", ".yaml", ".yml", ".txt", ".pkl", ".md"}

DANGEROUS_PATTERNS = [
    b"os.system", b"subprocess.", b"socket.",
    b"urllib.", b"requests.", b"__import__",
    b"eval(", b"exec(", b"compile(",
]


def extract_training_code(model_path: Path) -> Optional[Path]:
    try:
        with ZipFile(model_path, 'r') as zf:
            if 'training_code.zip' not in zf.namelist():
                return None

            tmpdir = Path(tempfile.mkdtemp())
            training_code_path = tmpdir / "training_code.zip"

            with zf.open('training_code.zip') as src:
                with open(training_code_path, 'wb') as dst:
                    dst.write(src.read())

            return training_code_path
    except Exception:
        return None


def validate_zip_safety(path: Path) -> bool:
    try:
        with ZipFile(path) as zf:
            total_size = 0
            for info in zf.infolist():
                if info.filename.startswith(("/", "\\")) or ".." in Path(info.filename).parts:
                    return False
                total_size += info.file_size
                if total_size > TRAINING_CODE_MAX_SIZE:
                    return False
        return True
    except Exception:
        return False


def validate_training_code(training_code_path: Path) -> Tuple[bool, float, List[str]]:
    findings = []

    try:
        if not training_code_path.exists():
            return False, 0.0, ["Training code file does not exist"]

        if training_code_path.stat().st_size > TRAINING_CODE_MAX_SIZE:
            return False, 0.0, [f"Training code exceeds {TRAINING_CODE_MAX_SIZE} bytes"]

        if not validate_zip_safety(training_code_path):
            return False, 0.0, ["Training code ZIP is unsafe"]

        with ZipFile(training_code_path, 'r') as zf:
            file_list = zf.namelist()

            if "train.py" not in file_list:
                return False, 0.0, ["Missing required train.py"]

            if len(file_list) > TRAINING_CODE_MAX_FILES:
                return False, 0.0, [f"Too many files: {len(file_list)} > {TRAINING_CODE_MAX_FILES}"]

            for filename in file_list:
                ext = Path(filename).suffix
                if ext and ext not in ALLOWED_EXTENSIONS:
                    findings.append(f"Disallowed file extension: {filename}")

            train_py_content = zf.read("train.py")

            for pattern in DANGEROUS_PATTERNS:
                if pattern in train_py_content:
                    findings.append(f"Dangerous pattern found: {pattern.decode('utf-8', errors='ignore')}")

            try:
                train_py_str = train_py_content.decode('utf-8')
                tree = ast.parse(train_py_str)

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if any(forbidden in alias.name for forbidden in FORBIDDEN_IMPORTS):
                                findings.append(f"Forbidden import: {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and any(forbidden in node.module for forbidden in FORBIDDEN_IMPORTS):
                            findings.append(f"Forbidden import from: {node.module}")

            except SyntaxError as e:
                findings.append(f"Syntax error in train.py: {e}")
            except Exception as e:
                findings.append(f"Failed to parse train.py: {e}")

        if findings:
            critical_count = sum(1 for f in findings if any(x in f.lower() for x in ["forbidden", "dangerous"]))
            if critical_count > 0:
                return False, 0.0, findings

            safety_score = max(0.0, 1.0 - (len(findings) * 0.1))
            return True, safety_score, findings

        return True, 1.0, []

    except BadZipFile:
        return False, 0.0, ["Corrupted ZIP file"]
    except Exception as e:
        return False, 0.0, [f"Validation error: {e}"]


def train_model(training_code_path: Path, seed: int, timesteps: int) -> Optional[Path]:
    try:
        code_dir = Path(tempfile.mkdtemp())
        with ZipFile(training_code_path, 'r') as zf:
            zf.extractall(code_dir)

        train_script = code_dir / "train.py"
        if not train_script.exists():
            return None

        output_dir = Path(tempfile.mkdtemp())
        output_model = output_dir / "trained_model.zip"

        sys.path.insert(0, str(code_dir))

        try:
            import train as training_module
            if hasattr(training_module, 'train'):
                training_module.train(
                    seed=seed,
                    timesteps=timesteps,
                    output_path=str(output_model)
                )
            else:
                return None

            if output_model.exists():
                return output_model
            else:
                return None

        finally:
            if str(code_dir) in sys.path:
                sys.path.remove(str(code_dir))

    except Exception:
        return None


def load_policy_weights(model_path: Path) -> Optional[Dict]:
    try:
        with ZipFile(model_path, 'r') as zf:
            if 'policy.pth' not in zf.namelist():
                return None

            with zf.open('policy.pth') as f:
                state_dict = torch.load(f, map_location='cpu', weights_only=False)
                return state_dict
    except Exception:
        return None


def compare_architectures(uploaded_model: Path, trained_model: Path) -> float:
    try:
        uploaded_policy = load_policy_weights(uploaded_model)
        trained_policy = load_policy_weights(trained_model)

        if uploaded_policy is None or trained_policy is None:
            return 0.0

        uploaded_layers = list(uploaded_policy.keys())
        trained_layers = list(trained_policy.keys())

        if uploaded_layers != trained_layers:
            return 0.5

        total_params_uploaded = sum(p.numel() for p in uploaded_policy.values())
        total_params_trained = sum(p.numel() for p in trained_policy.values())

        if total_params_uploaded != total_params_trained:
            return 0.5

        shape_matches = sum(1 for k in uploaded_layers if uploaded_policy[k].shape == trained_policy[k].shape)
        shape_score = shape_matches / len(uploaded_layers) if uploaded_layers else 0.0

        return shape_score
    except Exception:
        return 0.0


def evaluate_model(model_path: Path, test_seeds: List[int]) -> List[float]:
    try:
        from swarm.validator.task_gen import random_task
        from swarm.core.secure_loader import secure_load_ppo
        from swarm.utils.env_factory import make_env
        from swarm.validator.reward import flight_reward

        scores = []

        for seed in test_seeds:
            task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=seed)
            env = make_env(task, gui=False)

            try:
                model = secure_load_ppo(model_path, env=env, device="cpu")

                obs = env._computeObs()
                if isinstance(obs, dict):
                    obs = obs[next(iter(obs))]

                t_sim = 0.0
                success = False

                while t_sim < task.horizon:
                    act, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(act[None, :])
                    t_sim += SIM_DT

                    if terminated or truncated:
                        success = info.get("success", False)
                        break

                score = flight_reward(success=success, t=t_sim, horizon=task.horizon, task=task)
                scores.append(score)

            finally:
                env.close()
                del model
                gc.collect()

        return scores
    except Exception:
        return []


def compare_performance(uploaded_model: Path, trained_model: Path, test_seeds: List[int]) -> Tuple[float, Dict]:
    try:
        uploaded_scores = evaluate_model(uploaded_model, test_seeds)
        trained_scores = evaluate_model(trained_model, test_seeds)

        if len(uploaded_scores) == 0 or len(trained_scores) == 0:
            return 0.0, {'error': 'No evaluation results'}

        uploaded_arr = np.array(uploaded_scores, dtype=np.float64)
        trained_arr = np.array(trained_scores, dtype=np.float64)

        correlation = np.corrcoef(uploaded_arr, trained_arr)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0

        mean_diff = np.abs(uploaded_arr - trained_arr).mean()

        correlation_pass = correlation >= VERIFICATION_CORRELATION_THRESHOLD
        mean_diff_pass = mean_diff <= VERIFICATION_MEAN_DIFF_THRESHOLD

        if correlation_pass and mean_diff_pass:
            score = 1.0
        elif correlation_pass or mean_diff_pass:
            score = 0.5
        else:
            score = 0.0

        details = {
            'correlation': float(correlation),
            'mean_difference': float(mean_diff),
            'uploaded_scores': [float(s) for s in uploaded_scores],
            'trained_scores': [float(s) for s in trained_scores],
            'correlation_pass': correlation_pass,
            'mean_diff_pass': mean_diff_pass,
        }

        return score, details
    except Exception as e:
        return 0.0, {'error': str(e)}


def main():
    if len(sys.argv) != 4:
        print("Usage: training_verification.py <model_path> <uid> <result_file>")
        sys.exit(1)

    model_path = Path(sys.argv[1])
    uid = int(sys.argv[2])
    result_file = Path(sys.argv[3])

    result = {
        "uid": uid,
        "safety_score": 0.0,
        "training_completed": False,
        "architecture_match": 0.0,
        "performance_correlation": 0.0,
        "performance_mean_diff": 1.0,
        "final_score": 0.0,
        "passed": False,
        "error": None
    }

    try:
        training_code_path = extract_training_code(model_path)
        if not training_code_path:
            result["error"] = "No training code found"
            with open(result_file, 'w') as f:
                json.dump(result, f)
            sys.exit(0)

        is_safe, safety_score, findings = validate_training_code(training_code_path)
        result["safety_score"] = safety_score
        result["safety_findings"] = findings

        if not is_safe:
            result["error"] = f"Code safety failed: {findings}"
            with open(result_file, 'w') as f:
                json.dump(result, f)
            sys.exit(0)

        seed = 42
        timesteps = 50000
        trained_model_path = train_model(training_code_path, seed, timesteps)
        result["training_completed"] = trained_model_path is not None

        if not trained_model_path:
            result["error"] = "Training failed"
            with open(result_file, 'w') as f:
                json.dump(result, f)
            sys.exit(0)

        arch_score = compare_architectures(model_path, trained_model_path)
        result["architecture_match"] = arch_score

        test_seeds = [42 + i * 1000 for i in range(VERIFICATION_TEST_TASKS)]
        perf_score, perf_details = compare_performance(model_path, trained_model_path, test_seeds)
        result["performance_correlation"] = perf_details.get('correlation', 0.0)
        result["performance_mean_diff"] = perf_details.get('mean_difference', 1.0)
        result["performance_details"] = perf_details

        output_trained_model = result_file.parent / f"trained_model_uid_{uid}.zip"
        shutil.copy2(trained_model_path, output_trained_model)
        result["trained_model_saved"] = str(output_trained_model.name)

        final_score = (
            0.20 * safety_score +
            0.10 * (1.0 if result["training_completed"] else 0.0) +
            0.20 * arch_score +
            0.50 * perf_score
        )
        result["final_score"] = final_score
        result["passed"] = final_score >= VERIFICATION_PASS_THRESHOLD

    except Exception as e:
        result["error"] = str(e)

    with open(result_file, 'w') as f:
        json.dump(result, f)

    sys.exit(0)


if __name__ == "__main__":
    main()
