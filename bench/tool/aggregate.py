#!/usr/bin/env python3
"""Aggregate BENCH_JSON lines from bench run logs into a markdown table.

Usage:
  flutter test integration_test -d <device> | tee run.log
  python3 bench/tool/aggregate.py run.log [more.log ...]
"""

import json
import sys
from pathlib import Path

MARKER = "BENCH_JSON "


def read_text_any(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16")  # PowerShell Tee-Object writes UTF-16
    return raw.decode("utf-8", errors="replace")


def load(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        for line in read_text_any(Path(p)).splitlines():
            idx = line.find(MARKER)
            if idx < 0:
                continue
            try:
                rows.append(json.loads(line[idx + len(MARKER):]))
            except json.JSONDecodeError:
                print(f"warning: bad json line in {p}", file=sys.stderr)
    return rows


def fmt_config(cfg: dict) -> str:
    skip = {"width", "height", "frames"}
    return " ".join(f"{k}={v}" for k, v in cfg.items() if k not in skip)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    rows = load(sys.argv[1:])
    if not rows:
        raise SystemExit("no BENCH_JSON lines found")
    rows.sort(key=lambda r: (r["suite"], r["app"], fmt_config(r["config"])))

    print("| suite | app | config | mean ms | median | p95 | extra |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        s = r["stats"]
        extra = {
            k: v
            for k, v in r.items()
            if k not in {"app", "suite", "config", "stats", "samples"}
        }
        extra_s = " ".join(
            f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in extra.items()
        )
        print(
            f"| {r['suite']} | {r['app']} | {fmt_config(r['config'])} "
            f"| {s['mean']:.2f} | {s['median']:.2f} | {s['p95']:.2f} "
            f"| {extra_s} |"
        )


if __name__ == "__main__":
    main()
