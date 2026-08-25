"""Corpus construction and the duplicate audit.

get_corpus(name) returns (sequences, labels, class_names, info) for:
    "raw"           the corpus exactly as distributed
    "dedup"         exact duplicates collapsed, majority label per sequence
    "dedup_strict"  as dedup, but sequences carrying conflicting labels dropped

Why this matters. A corpus of biological sequences can contain a large fraction
of exact duplicates. Under a random train/test split those duplicates put copies
of training rows into the test set, and the reported score then measures partly
memorisation rather than generalisation. Deduplicating first removes that
inflation; exact_duplicate_leakage() quantifies it for any given split so the
size of the effect can be reported rather than assumed.

Also provides truncate(seqs, n) for length-matched comparisons against models
with a fixed context window.
"""
from __future__ import annotations

import collections
import hashlib

import numpy as np

DATA_ROOT = "data"


def load_covid_raw(root: str = DATA_ROOT):
    seqs = np.load(f"{root}/seq_data_7000.npy", allow_pickle=True)
    labs = np.load(f"{root}/seq_data_variant_names_7000.npy", allow_pickle=True)
    S = [str(s[0]) if np.ndim(s) else str(s) for s in seqs]
    Y = [str(l[0]) if np.ndim(l) else str(l) for l in labs]
    return S, Y


def corpus_stats(S, Y) -> dict:
    counts = collections.Counter(Y)
    by_seq = collections.defaultdict(collections.Counter)
    for s, y in zip(S, Y):
        by_seq[s][y] += 1
    conflicting = sum(1 for c in by_seq.values() if len(c) > 1)
    L = np.array([len(s) for s in S])
    return {
        "n": len(S),
        "n_unique_sequences": len(by_seq),
        "duplicate_fraction": round(1 - len(by_seq) / len(S), 4),
        "n_classes": len(counts),
        "majority_class": counts.most_common(1)[0][0],
        "majority_fraction": round(counts.most_common(1)[0][1] / len(S), 4),
        "rarest_class": counts.most_common()[-1][0],
        "rarest_count": counts.most_common()[-1][1],
        "n_sequences_with_conflicting_labels": conflicting,
        "len_min": int(L.min()), "len_max": int(L.max()),
        "frac_over_1022": float((L > 1022).mean()),
        "residues_kept_at_1022": float(np.minimum(L, 1022).sum() / L.sum()),
    }


def deduplicate(S, Y, policy: str = "majority", seed: int = 0):
    """Collapse identical sequences to one row.

    policy:
      'majority' -- keep the most frequent label, ties broken deterministically
                    by label name (reproducible, no RNG).
      'drop'     -- discard any sequence whose label is not unanimous.
      'first'    -- keep the first occurrence, matching a naive dedup.

    Returns (sequences, labels, info).
    """
    by_seq = collections.OrderedDict()
    for s, y in zip(S, Y):
        by_seq.setdefault(s, collections.Counter())[y] += 1

    out_s, out_y, dropped = [], [], 0
    for s, c in by_seq.items():
        if len(c) > 1 and policy == "drop":
            dropped += 1
            continue
        if policy == "first":
            lab = next(iter(c))
        else:
            top = max(c.values())
            lab = sorted(k for k, v in c.items() if v == top)[0]
        out_s.append(s)
        out_y.append(lab)
    info = {"policy": policy, "n_in": len(S), "n_out": len(out_s),
            "n_dropped_conflicting": dropped}
    return out_s, out_y, info


def encode_labels(Y):
    names = sorted(set(Y))
    to_i = {c: i for i, c in enumerate(names)}
    return np.array([to_i[y] for y in Y], dtype=np.int32), names


def get_corpus(name: str = "dedup", root: str = DATA_ROOT):
    """name in {'raw', 'dedup', 'dedup_strict'} -> (seqs, y, class_names, info)."""
    S, Y = load_covid_raw(root)
    if name == "raw":
        info = {"corpus": "raw"}
    elif name == "dedup":
        S, Y, info = deduplicate(S, Y, policy="majority")
        info["corpus"] = "dedup"
    elif name == "dedup_strict":
        S, Y, info = deduplicate(S, Y, policy="drop")
        info["corpus"] = "dedup_strict"
    else:
        raise ValueError(f"unknown corpus: {name}")
    y, names = encode_labels(Y)
    info.update(corpus_stats(S, Y))
    return S, y, names, info


def truncate(S, max_len: int = 1022):
    """Length-matched control for the ESM-2 comparison (plm_frozen, issue B1)."""
    return [s[:max_len] for s in S]


def exact_duplicate_leakage(S, split) -> dict:
    tr, te = split
    seen = {hashlib.md5(S[i].encode()).hexdigest() for i in tr}
    dup = sum(1 for i in te if hashlib.md5(S[i].encode()).hexdigest() in seen)
    return {"n_test": len(te), "n_exact_dup": dup,
            "frac_exact_dup": dup / max(1, len(te))}


if __name__ == "__main__":
    import json
    for name in ("raw", "dedup", "dedup_strict"):
        S, y, names, info = get_corpus(name)
        print(name, json.dumps(info, indent=1), flush=True)
