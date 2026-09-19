import argparse, asyncio, hashlib, json, os, subprocess, time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

from src.rag_pipeline import RAGPipeline
from src.reranker import RerankingRetriever
from evals import eval_retriever, eval_generator, eval_rag_pipeline, eval_application, eval_safety, eval_operations

BASELINE_PATH = "baselines/baseline.json"
CANDIDATE_PATH = "baselines/candidate.json"

def _slug(name):
    return name.strip().lower().replace(" ", "_")

def flatten_nested(namespace, summary):
    out = {}
    for metric, stats in summary.items():
        slug = _slug(metric)
        for stat, val in stats.items():
            out[f"{namespace}.{slug}.{stat}"] = val
    return out

def prefix_flat(namespace, flat):
    return {f"{namespace}.{key}": val for key, val in flat.items()}

def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"

def _prompt_hash():
    try:
        from src.generator import prompt
        return hashlib.sha256(str(prompt).encode()).hexdigest()[:12]
    except Exception:
        return "unknown"

def build_metadata(label):
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "prompt_hash": _prompt_hash(),
        "label": label
    }

def run_suite(label="", quiet=False):
    verbose = not quiet
    t_start = time.perf_counter()

    print("Building pipeline (once)...")
    rag = asyncio.run(RAGPipeline.create(fetch_k=20, top_k=5))
    retriever = getattr(rag, "retriever", None) or asyncio.run(RerankingRetriever.create(fetch_k=20, top_k=5))

    metrics = {}

    print("\n[1/6] retriever eval...")
    metrics.update(flatten_nested("retriever", eval_retriever.run(retriever)))

    print("\n[2/6] generator eval...")
    metrics.update(flatten_nested("generator", eval_generator.run()))

    print("\n[3/6] pipeline (triad) eval...")
    metrics.update(flatten_nested("pipeline", eval_rag_pipeline.run(rag)))

    print("\n[4/6] application quality eval...")
    metrics.update(flatten_nested("application", eval_application.run(rag)))

    print("\n[5/6] safety evals...")
    metrics.update(prefix_flat("safety", eval_safety.run_safety(rag, verbose=verbose)))

    print("\n[6/6] operational evals...")
    metrics.update(prefix_flat("ops", eval_operations.run_operations(rag, verbose=verbose)))

    elapsed = time.perf_counter() - t_start

    return {
        "metadata": {
            **build_metadata(label),
            "suite_seconds": round(elapsed, 1),
            "n_metrics": len(metrics)
        },
        "metrics": metrics
    }

def write_snapshot(snapshot, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
    return path

def print_snapshot(snapshot):
    meta = snapshot["metadata"]
    metrics = snapshot["metrics"]
    print("\n" + "=" * 74)
    print("SNAPSHOT")
    print("=" * 74)
    print(f"label       : {meta.get('label') or '(none)'}")
    print(f"created_at  : {meta['created_at']}")
    print(f"git_sha     : {meta['git_sha']}    prompt_hash: {meta['prompt_hash']}")
    print(f"suite_time  : {meta['suite_seconds']}s     metrics: {len(metrics)}")
    print("-" * 74)
    for key in sorted(metrics):
        val = metrics[key]
        shown = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"  {key:<44} {shown}")
    print("=" * 74)

def main():
    parser = argparse.ArgumentParser(description="Run the full eval suite and write a snapshot.")
    parser.add_argument("--baseline", action="store_true",
                        help=f"write to {BASELINE_PATH} (bless this run as the baseline)")
    parser.add_argument("--out", default=None, help="custom output path")
    parser.add_argument("--label", default="", help="describe the change this snapshot represents")
    parser.add_argument("--quiet", action="store_true", help="suppress per-eval ops/safety chatter")
    args = parser.parse_args()

    out = args.out or (BASELINE_PATH if args.baseline else CANDIDATE_PATH)

    snapshot = run_suite(label=args.label, quiet=args.quiet)
    print_snapshot(snapshot)
    path = write_snapshot(snapshot, out)
    print(f"\nwrote snapshot -> {path}")

if __name__ == "__main__":
    main()