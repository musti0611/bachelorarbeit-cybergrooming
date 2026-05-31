"""Per-label Precision, Recall, F1 evaluation."""

from collections import defaultdict
from parse_goldstandard import LABELS


def compute_metrics(results: list) -> dict:
    """
    results: [{"gold": label, "pred": label, ...}, ...]
    Returns per-label and macro Precision/Recall/F1.
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for r in results:
        gold = r["gold"]
        pred = r["pred"]
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    metrics = {}
    for label in LABELS:
        t = tp[label]
        p_denom = tp[label] + fp[label]
        r_denom = tp[label] + fn[label]
        precision = t / p_denom if p_denom > 0 else 0.0
        recall = t / r_denom if r_denom > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "support": r_denom,
            "tp": t,
            "fp": fp[label],
            "fn": fn[label],
        }

    active = [m for m in metrics.values() if m["support"] > 0]
    metrics["MACRO"] = {
        "precision": round(sum(m["precision"] for m in active) / len(active), 3),
        "recall": round(sum(m["recall"] for m in active) / len(active), 3),
        "f1": round(sum(m["f1"] for m in active) / len(active), 3),
        "support": sum(m["support"] for m in active),
    }

    return metrics


def print_report(metrics: dict) -> None:
    col = 32
    header = f"{'Label':<{col}} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for label in LABELS:
        m = metrics[label]
        print(f"{label:<{col}} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8}")
    print(sep)
    m = metrics["MACRO"]
    print(f"{'MACRO':<{col}} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} {m['support']:>8}")
