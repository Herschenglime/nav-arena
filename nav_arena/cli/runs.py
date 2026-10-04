# Copyright (c) 2026, nav_arena developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The 'nav_arena runs' subcommand: list, show, compare, and rerun recorded runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from nav_arena.benchmarks.manifest import BatchManifest, RunStatus, _utcnow_iso
from nav_arena.benchmarks.results import append_results_row
from nav_arena.benchmarks.sweep import check_preflight_processes
from nav_arena.benchmarks.tracking import compare_batch, find_run_spec_for_rerun, list_runs, show_run
from nav_arena.cli.run import execute_single_run_process
from nav_arena.cli._help import print_subcommand_help
from nav_arena.utils.logger import get_logger
from nav_arena.utils.paths import RUNS_DIR

logger = get_logger("nav_arena.cli")


def handle_runs_list(args: argparse.Namespace) -> int:
    """Handle 'runs list' subcommand."""
    try:
        data = list_runs(RUNS_DIR, batch_id=args.batch)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if not data:
        if args.batch:
            print(f"No runs found in batch '{args.batch}'.")
        else:
            print("No benchmark batches found in runs directory.")
        return 0

    if args.batch:
        headers = ["Run ID", "Method", "Robot", "Route", "Seed", "Status", "Terminal Cause", "Success", "TTG (s)"]
        rows = []
        for r in data:
            ttg_str = f"{r['time_to_goal']:.2f}" if r.get("time_to_goal") is not None else "-"
            rows.append([
                str(r.get("run_id", "")),
                str(r.get("method", "")),
                str(r.get("robot", "")),
                str(r.get("route", "")),
                str(r.get("seed", "")),
                str(r.get("status", "")),
                str(r.get("terminal_cause", "") or "-"),
                str(r.get("success", False)),
                ttg_str,
            ])
    else:
        headers = ["Batch ID", "Total", "Completed", "Success Rate", "Status"]
        rows = []
        for b in data:
            rows.append([
                str(b.get("batch_id", "")),
                str(b.get("total_runs", b.get("total", 0))),
                str(b.get("completed", 0)),
                f"{b.get('success_rate', 0.0):.1f}%",
                str(b.get("status", "")),
            ])

    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "|-" + "-|-".join("-" * widths[i] for i in range(len(headers))) + "-|"
    print(header_line)
    print(sep_line)
    for row in rows:
        row_str = "| " + " | ".join(val.ljust(widths[i]) for i, val in enumerate(row)) + " |"
        print(row_str)

    return 0


def handle_runs_show(args: argparse.Namespace) -> int:
    """Handle 'runs show' subcommand."""
    try:
        data = show_run(args.id, RUNS_DIR)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if data.get("type") == "batch":
        print(f"Batch: {data.get('batch_id')}")
        print(f"Directory: {data.get('batch_dir')}")
        print(f"Status: {data.get('status')}")
        print(f"Success Rate: {data.get('success_rate', 0.0):.1f}%")
        progress = data.get("progress", {})
        print("Progress:")
        print(f"  Total:   {progress.get('total', 0)}")
        print(f"  Done:    {progress.get('done', 0)}")
        print(f"  Running: {progress.get('running', 0)}")
        print(f"  Queued:  {progress.get('queued', 0)}")
        print(f"  Failed:  {progress.get('failed', 0)}")
        print(f"  Timeout: {progress.get('timeout', 0)}")
        print(f"  Skipped: {progress.get('skipped', 0)}")
        duration = data.get("duration", {})
        print("Duration:")
        print(f"  Started:  {duration.get('started_at') or '-'}")
        print(f"  Ended:    {duration.get('ended_at') or '-'}")
        dur_s = duration.get("duration_s")
        print(f"  Duration: {f'{dur_s:.1f}s' if dur_s is not None else '-'}")
        spec = data.get("spec", {})
        if spec:
            print("Spec Summary:")
            for k, v in spec.items():
                print(f"  {k}: {v}")
    else:
        print(f"Run: {data.get('run_id')}")
        print(f"Directory: {data.get('run_dir')}")
        print(f"Status: {data.get('status')}")
        outcome = data.get("outcome", {})
        print("Outcome:")
        print(f"  Success:        {outcome.get('success')}")
        print(f"  Terminal Cause: {outcome.get('terminal_cause') or '-'}")
        print(f"  Exit Code:      {outcome.get('exit_code') if outcome.get('exit_code') is not None else '-'}")
        metrics = data.get("metrics", {})
        print("Metrics:")
        ttg = metrics.get("time_to_goal_s")
        print(f"  Time to Goal:     {f'{ttg:.2f} s' if ttg is not None else '-'}")
        pl = metrics.get("path_length_m")
        print(f"  Path Length:      {f'{pl:.2f} m' if pl is not None else '-'}")
        inf = metrics.get("mean_inference_ms")
        print(f"  Mean Inference:   {f'{inf:.1f} ms' if inf is not None else '-'}")
        st = metrics.get("steps")
        print(f"  Steps:            {st if st is not None else '-'}")
        fgd = metrics.get("final_goal_distance_m")
        print(f"  Final Goal Dist:  {f'{fgd:.2f} m' if fgd is not None else '-'}")
        settings = data.get("settings", {})
        if settings:
            print("Settings:")
            for k in ("method", "robot", "scene", "route", "seed"):
                if k in settings:
                    print(f"  {k}: {settings[k]}")
        print(f"Log: {data.get('log_path') or '-'}")

    return 0


def handle_runs_compare(args: argparse.Namespace) -> int:
    """Handle 'runs compare' subcommand."""
    try:
        table = compare_batch(args.batch, RUNS_DIR, by=args.by)
        print(table)
        return 0
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1


def handle_runs_rerun(args: argparse.Namespace) -> int:
    """Handle 'runs rerun' subcommand."""
    if not args.force:
        conflicts = check_preflight_processes()
        if conflicts:
            logger.error("Active simulation or verification processes detected:")
            for pid, cmd in conflicts:
                logger.error("  [PID %d] %s", pid, cmd)
            logger.error("Refusing to rerun. Use --force to proceed anyway.")
            return 1

    try:
        run_dir, spec = find_run_spec_for_rerun(args.run_id, RUNS_DIR)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    if args.gui is not None:
        spec.viz.gui = args.gui

    batch_id = ""
    if run_dir.parent.is_dir():
        if (run_dir.parent / "manifest.json").is_file() or (run_dir.parent / "batch.yaml").is_file():
            batch_id = run_dir.parent.name

    exit_code = execute_single_run_process(
        spec=spec,
        run_dir=run_dir,
        quiet=args.quiet,
        batch_id=batch_id,
    )

    # Sync back to batch manifest and batch results.csv if part of a batch
    if batch_id and (run_dir.parent / "manifest.json").is_file():
        try:
            m_path = run_dir.parent / "manifest.json"
            manifest = BatchManifest.load(m_path)
            for r in manifest.runs:
                if r.id == run_dir.name or r.run_dir == run_dir.name:
                    r.status = RunStatus.DONE.value if exit_code in (0, 2) else RunStatus.FAILED.value
                    r.exit_code = exit_code
                    sum_path = run_dir / "summary.json"
                    if sum_path.is_file():
                        try:
                            s_data = json.loads(sum_path.read_text(encoding="utf-8"))
                            r.terminal_cause = s_data.get("terminal_cause")
                        except Exception:
                            pass
                    r.ended = _utcnow_iso()
                    break
            manifest.save(m_path)
        except Exception as exc:
            logger.warning("Failed updating batch manifest after rerun: %s", exc)

        batch_csv = run_dir.parent / "results.csv"
        run_csv = run_dir / "results.csv"
        if run_csv.is_file():
            try:
                with run_csv.open("r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        append_results_row(batch_csv, rows[-1])
            except Exception as exc:
                logger.warning("Failed appending to batch results.csv after rerun: %s", exc)

    return exit_code


def handle_runs(args: argparse.Namespace) -> int:
    """Dispatch 'runs' subcommands."""
    action = getattr(args, "runs_action", None)
    if not action:
        return print_subcommand_help("runs")

    if action == "list":
        return handle_runs_list(args)
    elif action == "show":
        return handle_runs_show(args)
    elif action == "compare":
        return handle_runs_compare(args)
    elif action == "rerun":
        return handle_runs_rerun(args)
    else:
        logger.error("Unknown runs action '%s'", action)
        return 1


def add_runs_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the 'runs' subcommand."""
    runs_parser = subparsers.add_parser(
        "runs",
        help="Inspect, list, compare, or rerun recorded benchmark episodes.",
    )
    runs_sub = runs_parser.add_subparsers(dest="runs_action", metavar="ACTION")

    # runs list [--batch ID]
    list_p = runs_sub.add_parser("list", help="List batches or runs.")
    list_p.add_argument("--batch", type=str, default=None, help="Batch ID or directory to list runs for.")

    # runs show <run-or-batch-id>
    show_p = runs_sub.add_parser("show", help="Show details for a run or batch.")
    show_p.add_argument("id", type=str, help="Run ID, batch ID, or directory path.")

    # runs compare <batch-id> [--by method|route|robot]
    compare_p = runs_sub.add_parser("compare", help="Compare metrics across methods, routes, or robots.")
    compare_p.add_argument("batch", type=str, help="Batch ID or directory to compare.")
    compare_p.add_argument(
        "--by",
        choices=["method", "route", "robot"],
        default="method",
        help="Grouping dimension (default: method).",
    )

    # runs rerun <run-id> [--gui] [--quiet] [--force]
    rerun_p = runs_sub.add_parser("rerun", help="Re-execute a previously recorded run.")
    rerun_p.add_argument("run_id", type=str, help="Run ID or path to re-execute.")
    rerun_p.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Kit GUI viewport (--gui / --no-gui).",
    )
    rerun_p.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Suppress worker stdout unless an error occurs.",
    )
    rerun_p.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Bypass pre-flight conflict check.",
    )
