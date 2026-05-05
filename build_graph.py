#!/usr/bin/env python3
"""
build_graph.py  —  Build wiki_graph.pkl from the Danker Wikipedia link dataset.

The Danker dataset is a pre-parsed TSV of the English Wikipedia link graph
maintained by Benjamin Danker. Each line is simply:

    source_title<TAB>target_title

No SQL parsing, no page-ID resolution — just read and build the dict.

Dataset URL:
    https://danker.s3.amazonaws.com/index.html

Download the most recent file listed there, e.g.:
    2024-09-01.allwiki.links.rank  (or .tsv / .txt depending on the release)

Usage:
    python3 build_graph.py 2024-09-01.allwiki.links.rank
    python3 build_graph.py links.tsv --out /data/wiki
    python3 build_graph.py links.tsv --limit 5000000   # quick test run

Output:
    wiki_graph.pkl   — {source_title: [target_title, ...]}
    graph_stats.json — metadata

Runtime:  2-5 minutes (pure Python TSV parsing).
RAM:      ~4-6 GB peak, ~3 GB for the final dict.
Disk:     ~3 GB for wiki_graph.pkl.
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional


def log(msg: str):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def build(tsv_path: str, out_dir: str, limit: Optional[int]):
    if not os.path.exists(tsv_path):
        print(f"ERROR: file not found: {tsv_path}", file=sys.stderr)
        sys.exit(1)

    file_size = os.path.getsize(tsv_path)
    log(f"Input  : {tsv_path}  ({file_size / 1e9:.2f} GB)")
    log(f"Output : {out_dir}")
    if limit:
        log(f"Limit  : {limit:,} lines (test mode)")

    graph: dict = defaultdict(list)
    n_lines  = 0
    n_bad    = 0
    t0       = time.time()
    report   = 1_000_000  # print progress every 1M lines

    # The Danker file may be plain TSV or may include a rank/score column.
    # We only care about the first two tab-separated fields.
    with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")

            # Skip blank lines and comment lines
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                n_bad += 1
                continue

            src = parts[0].strip()
            tgt = parts[1].strip()

            if not src or not tgt:
                n_bad += 1
                continue

            graph[src].append(tgt)
            n_lines += 1

            if n_lines % report == 0:
                elapsed = time.time() - t0
                rate    = n_lines / elapsed / 1e6
                print(f"\r  {n_lines/1e6:.1f}M edges  {rate:.2f}M/s  "
                      f"{len(graph)/1e6:.1f}M articles …",
                      end="", flush=True)

            if limit and n_lines >= limit:
                log(f"\n  Limit of {limit:,} reached — stopping early")
                break

    print()  # newline after progress line

    elapsed    = time.time() - t0
    n_articles = len(graph)

    log(f"Parsed {n_lines:,} edges across {n_articles:,} articles in {elapsed:.1f}s")
    if n_bad:
        log(f"Skipped {n_bad:,} malformed lines")

    # Convert defaultdict → plain dict to reduce pickle size slightly
    graph = dict(graph)

    os.makedirs(out_dir, exist_ok=True)
    graph_path = os.path.join(out_dir, "wiki_graph.pkl")
    stats_path = os.path.join(out_dir, "graph_stats.json")

    log(f"Writing {graph_path} …")
    t_write = time.time()
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    write_s  = time.time() - t_write
    size_mb  = os.path.getsize(graph_path) / 1e6
    log(f"  Written {size_mb:.0f} MB in {write_s:.1f}s")

    stats = {
        "source_file":   os.path.abspath(tsv_path),
        "built_at":      datetime.utcnow().isoformat(),
        "articles":      n_articles,
        "edges":         n_lines,
        "bad_lines":     n_bad,
        "elapsed_s":     round(elapsed + write_s),
        "graph_file":    graph_path,
        "partial_build": bool(limit),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    log("=" * 60)
    log(f"Done in {(elapsed + write_s)/60:.1f} min")
    log(f"Articles : {n_articles:,}")
    log(f"Edges    : {n_lines:,}")
    log(f"Graph    : {graph_path}  ({size_mb:.0f} MB)")
    log("=" * 60)


def main():
    ap = argparse.ArgumentParser(
        description="Build wiki_graph.pkl from a Danker TSV link file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 build_graph.py 2024-09-01.allwiki.links.rank
  python3 build_graph.py links.tsv --out /data/wiki
  python3 build_graph.py links.tsv --limit 2000000
        """,
    )
    ap.add_argument("tsv", help="Path to Danker TSV file")
    ap.add_argument("--out", default=".",
                    help="Output directory (default: current dir)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many edges (useful for testing)")
    args = ap.parse_args()
    build(args.tsv, args.out, args.limit)


if __name__ == "__main__":
    main()
