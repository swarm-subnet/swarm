from __future__ import annotations

from ._shared import (
    Dict,
    List,
    Optional,
    Path,
    Tuple,
    _BatchStat,
    asyncio,
    deque,
    json,
    sys,
    time,
    traceback,
)
from .config import (
    _active_runtime_overrides,
    _apply_relaxed_overrides,
    _parse_args,
    _resolve_run_options,
    _Tee,
    _ts,
)
from .reporting import _print_results
from .seeds import (
    _find_seeds,
    _infer_uid_from_model_path,
    _load_type_seeds,
    _save_type_seeds,
)
from .workers import _run_benchmark


def main() -> None:
    import swarm.benchmark.engine as engine

    args = _parse_args()
    run_opts = _resolve_run_options(args)

    requested_workers = max(1, int(args.workers))
    effective_workers = requested_workers

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    inferred_uid = _infer_uid_from_model_path(model_path)
    if args.uid is None:
        uid = inferred_uid if inferred_uid is not None else 0
    else:
        uid = int(args.uid)

    effective_log_out = args.log_out
    if effective_log_out is None and run_opts.default_log_out:
        effective_log_out = Path(run_opts.default_log_out)

    log_fh = None
    run_error: Optional[BaseException] = None
    report_error: Optional[BaseException] = None
    task_meta: List[Dict[str, object]] = []
    results: list = []
    seed_times: List[float] = []
    seed_wall_by_key: Dict[Tuple[int, int], deque[float]] = {}
    full_wall_by_key: Dict[Tuple[int, int], float] = {}
    seed_status_by_key: Dict[Tuple[int, int], deque[str]] = {}
    batch_stats: List[_BatchStat] = []
    elapsed = 0.0
    eval_start = time.time()
    launched_workers = effective_workers
    overrides: Dict[str, object] = {}
    summary: Optional[Dict[str, object]] = None

    try:
        if effective_log_out:
            log_fh = open(effective_log_out, "w")
            sys.stdout = _Tee(sys.__stdout__, log_fh)
            sys.stderr = _Tee(sys.__stderr__, log_fh)

        if args.relax_timeouts:
            overrides = _apply_relaxed_overrides()

        print(f"[{_ts()}] === FULL EVALUATION BENCHMARK ===")
        print(f"[{_ts()}] Model: {model_path}")
        if args.uid is None and inferred_uid is not None:
            print(f"[{_ts()}] UID: {uid} (inferred from model filename)")
        elif args.uid is None:
            print(f"[{_ts()}] UID: {uid} (default fallback)")
        else:
            print(f"[{_ts()}] UID: {uid} (from --uid)")
        print(f"[{_ts()}] Workers requested: {requested_workers}")
        print(f"[{_ts()}] Workers effective:  {effective_workers}")
        print(f"[{_ts()}] Profile: debug")
        print(f"[{_ts()}] RPC verbosity: {args.rpc_verbosity}")
        print(f"[{_ts()}] Host parallelism: process")
        print(f"[{_ts()}] Container mode: single-seed")
        if overrides:
            print(f"[{_ts()}] Relaxed timeouts: {overrides}")
        runtime_overrides = _active_runtime_overrides()
        if runtime_overrides:
            print(f"[{_ts()}] Runtime overrides: {runtime_overrides}")
        from ._shared import BENCH_GROUP_ORDER
        print(f"[{_ts()}] Map types selected: {', '.join(BENCH_GROUP_ORDER)}")
        if effective_log_out:
            print(f"[{_ts()}] Log file: {effective_log_out}")
        print()

        if args.seed_file is not None:
            print(f"[{_ts()}] Loading seeds from {args.seed_file}...")
            type_seeds = engine._load_type_seeds(args.seed_file)
        else:
            if args.seed_search_rng is not None:
                print(f"[{_ts()}] Seed search RNG: {args.seed_search_rng}")
                import random
                random.seed(args.seed_search_rng)
            print(f"[{_ts()}] Finding {args.seeds_per_group} seeds per group...")
            type_seeds = engine._find_seeds(args.seeds_per_group)
        if args.save_seed_file is not None:
            engine._save_type_seeds(args.save_seed_file, type_seeds)
            print(f"[{_ts()}] Saved seed file: {args.save_seed_file}")
        total_seeds = sum(len(v) for v in type_seeds.values())
        for group, seeds in type_seeds.items():
            print(f"  {group}: {seeds}")
        print(f"  Total: {total_seeds}")
        print()

        (
            task_meta,
            results,
            seed_times,
            seed_wall_by_key,
            seed_status_by_key,
            full_wall_by_key,
            batch_stats,
            elapsed,
            eval_start,
            launched_workers,
        ) = asyncio.run(
            engine._run_benchmark(
                model_path,
                uid,
                type_seeds,
                effective_workers,
                run_opts=run_opts,
            )
        )
    except BaseException as exc:
        run_error = exc
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        out_stream = sys.__stdout__
        err_stream = sys.__stderr__
        if out_stream is None or getattr(out_stream, "closed", False):
            out_stream = err_stream if err_stream is not None else sys.stdout
        if err_stream is None or getattr(err_stream, "closed", False):
            err_stream = out_stream if out_stream is not None else sys.stderr
        sys.stdout = out_stream
        sys.stderr = err_stream

    final_out_stream = _Tee(out_stream, log_fh) if log_fh else out_stream
    final_err_stream = _Tee(err_stream, log_fh) if log_fh else err_stream
    sys.stdout = final_out_stream
    sys.stderr = final_err_stream
    try:
        print(f"\n[{_ts()}] === RESULTS ===", flush=True)
        if task_meta and results:
            if run_error is not None:
                print(f"[{_ts()}] Printing partial results collected before failure.", flush=True)
            try:
                summary = _print_results(
                    task_meta,
                    results,
                    seed_times,
                    seed_wall_by_key,
                    seed_status_by_key,
                    full_wall_by_key,
                    batch_stats,
                    elapsed,
                    eval_start,
                    launched_workers,
                    host_parallelism="process",
                )
            except BaseException as exc:
                report_error = exc
                print(f"[{_ts()}] Report generation failed: {type(exc).__name__}: {exc}", flush=True)

        if run_error is not None:
            print(
                f"[{_ts()}] Benchmark failed before report generation: {type(run_error).__name__}: {run_error}",
                flush=True,
            )
            traceback.print_exception(type(run_error), run_error, run_error.__traceback__)
            if report_error is not None:
                traceback.print_exception(type(report_error), report_error, report_error.__traceback__)
            print(f"[{_ts()}] === BENCHMARK FAILED ===", flush=True)
        elif report_error is not None:
            traceback.print_exception(type(report_error), report_error, report_error.__traceback__)
            print(f"[{_ts()}] === BENCHMARK FAILED ===", flush=True)
        else:
            print(f"[{_ts()}] === BENCHMARK COMPLETE ===", flush=True)
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        sys.stdout = out_stream
        sys.stderr = err_stream

    if log_fh:
        try:
            log_fh.flush()
        except Exception:
            pass
        finally:
            log_fh.close()

    if args.summary_json_out is not None and summary is not None:
        args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if run_error is not None:
        raise run_error
    if report_error is not None:
        raise report_error
