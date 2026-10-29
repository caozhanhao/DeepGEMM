#!/usr/bin/env python3
"""Per-rank NCU wrapper, launched by torchrun after a successful plain replay.

Prints the command by default. --execute explicitly starts NCU. Each rank gets
its own report, with a TCP lockstep communicator across ALL EP ranks.
"""

import argparse
import os
import shlex
import sys
from pathlib import Path


def build_command(args, env):
    rank, size = int(env["RANK"]), int(env["WORLD_SIZE"])
    if args.port == int(env.get("MASTER_PORT", "-1")):
        raise ValueError(
            "NCU communicator needs a port distinct from torchrun MASTER_PORT"
        )
    command = [
        args.ncu,
        "--config-file",
        "off",
        "--nvtx",
        "--nvtx-include",
        "megamoe-replay/",
        "--kernel-name",
        "regex:sm100_fp4_fp4_mega_moe_impl",
        "--launch-count",
        "1",
        "--replay-mode",
        "application",
        "--lockstep-kernel-launch",
        "--lockstep-nvtx-include",
        "megamoe-replay/",
        "--communicator",
        "tcp",
        "--communicator-tcp-hostname",
        args.host,
        "--communicator-tcp-port",
        str(args.port),
        "--communicator-tcp-num-peers",
        str(size),
        "--clock-control",
        "none",
        "--cache-control",
        "none",
        "--import-source",
        "yes",
    ]
    if args.section:
        for section in args.section:
            command.extend(["--section", section])
    else:
        command.extend(["--metrics", "gpu__time_duration.sum"])
    command.extend(
        [
            "-o",
            str(args.output / f"rank-{rank:04d}"),
            sys.executable,
            str(Path(__file__).with_name("replay_mega_moe.py")),
            str(args.capture),
            "--output",
            str(args.output / "replay-attempts"),
            "--profile",
            "ncu",
            "--iterations",
            "1",
            "--warmup",
            str(args.warmup),
            "--timeout-seconds",
            str(args.timeout_seconds),
        ]
    )
    if args.deep_gemm_path:
        command.extend(["--deep-gemm-path", str(args.deep_gemm_path)])
    if args.allow_native_change:
        command.append("--allow-native-change")
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--host",
        required=True,
        help="Explicit reachable NCU communicator host, also for multi-node runs",
    )
    parser.add_argument("--port", type=int, default=29513)
    parser.add_argument("--ncu", default="ncu")
    parser.add_argument(
        "--section",
        action="append",
        help="Add only sections supported by this NCU/GPU; default duration only",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--deep-gemm-path", type=Path)
    parser.add_argument("--allow-native-change", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.warmup < 1 or args.timeout_seconds < 1:
        parser.error("Invalid port/warmup/timeout")
    command = build_command(args, os.environ)
    print(shlex.join(command), flush=True)
    if args.execute:
        args.output.mkdir(parents=True, exist_ok=True)
        # This changes compiler flags/JIT output; keep the comparison run's
        # lineinfo setting the same and record any native identity override.
        os.environ.setdefault("DG_JIT_WITH_LINEINFO", "1")
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
