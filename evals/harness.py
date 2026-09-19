import json

def load_goldens(path):
    with open(path) as f:
        return json.load(f)

def summarize_by_metric(result):
    test_results = getattr(result, "test_results", None)
    if test_results is None:
        test_results = result if isinstance(result, list) else []

    buckets = {}
    for tr in test_results:
        metrics = getattr(tr, "metrics_data", None) or getattr(tr, "metrics", None) or []
        for m in metrics:
            name = getattr(m, "name", "unknown")
            b = buckets.setdefault(name, {"scores": [], "passed": 0, "total": 0})
            b["total"] += 1
            score = getattr(m, "score", None)
            if score is not None:
                b["scores"].append(score)
            if getattr(m, "success", False):
                b["passed"] += 1

    summary = {}
    for name, b in buckets.items():
        scores = b["scores"]
        summary[name] = {
            "n": b["total"],
            "pass_rate": (100 * b["passed"] / b["total"]) if b["total"] else 0.0,
            "avg_score": (sum(scores) / len(scores)) if scores else float("nan"),
            "min_score": min(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
        }
    return summary

def print_summary(title, summary):
    print("\n" + "=" * 60)
    print(f"{title}  (per-metric summary)")
    print("=" * 60)
    for name, s in summary.items():
        avg = f"{s['avg_score']:.2f}" if s["avg_score"] == s["avg_score"] else "nan"
        print(f"  {name:<26} pass_rate={s['pass_rate']:5.0f}%  avg={avg}  n={s['n']}")
    print("=" * 60)