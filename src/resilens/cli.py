import argparse
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> None:
    pass


def cmd_benchmark(args: argparse.Namespace) -> None:
    pass


def cmd_report(args: argparse.Namespace) -> None:
    pass


def cmd_build_scenarios(args: argparse.Namespace) -> None:
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resilens")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="process one video")
    run.add_argument("video", type=Path)
    run.add_argument("--mode", default="A", choices=["A", "B", "C", "D", "E"])
    run.add_argument("--profile", default="minimal")
    run.add_argument("--annotations", type=Path)
    run.add_argument("--replay", default="offline", choices=["offline", "realtime"])
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("benchmark", help="compare modes on scenarios")
    bench.add_argument("--modes", nargs="+", default=["A", "B", "C", "D", "E"])
    bench.add_argument("--profile", default="minimal")
    bench.add_argument("--repeats", type=int, default=3)
    bench.set_defaults(func=cmd_benchmark)

    report = sub.add_parser("report", help="build reports")
    report.add_argument("run_ids", nargs="+")
    report.set_defaults(func=cmd_report)

    scen = sub.add_parser("build-scenarios", help="generate degraded control clips")
    scen.add_argument("source_dir", type=Path)
    scen.add_argument("--seed", type=int, default=0)
    scen.set_defaults(func=cmd_build_scenarios)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
