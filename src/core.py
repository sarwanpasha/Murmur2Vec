"""Core library: hashed k-mer sketches, baselines, metrics and statistics.

Everything downstream imports this module, so every experiment shares one
implementation of the embedding, one metric code path and one statistical test.

Contents
--------
Hashing
    hash_fn(name, seed)      named hash family -> callable str -> int.
                             Supported: murmur (MurmurHash3), xxhash, fnv1a,
                             tabulation, sha256.
    sign_fn(seed)            independent sign hash for the signed estimator.

Embeddings
    murmur2vec(counts, m, signed=False, hash_family="murmur", ...)
                             fixed-size sketch of the k-mer spectrum. Each k-mer
                             is hashed to one of m buckets and its count added
                             there; with signed=True the contribution is
                             multiplied by an independent +-1 sign hash, which
                             makes the estimator unbiased under collisions.
                             force_collide={} deliberately maps chosen k-mers to
                             the same bucket, used by collision_mechanism.py.
    full_spectrum(counts, vocab)
                             the explicit k-mer count vector (the Spike2Vec-style
                             baseline). Dimension = number of observed k-mers.
    gaussian_sketch(X, m)    dense Gaussian random projection at matched
                             dimension, as a negative control.

Collision calibration
    collision_pct(keys, m)   1 - occupied_buckets / m at the given m.
    m_for_collision(keys, c) binary search for the smallest m meeting a target.
    m_heuristic(U, c)        closed form ceil((U-1)/(2c)) from balls-in-bins.
                             Accurate to a few percent for c <= 10%; it drifts to
                             about 1.4x at c = 40%.

Evaluation
    stratified_splits / grouped_splits / leave_one_class_out / few_shot_splits
    compute_metrics(...)     one metric path for every experiment. The headline
                             AUC is one-vs-rest macro; all_auc_variants=True also
                             emits OvR-weighted, OvO-macro and hard-label
                             variants, because these can differ by a lot on an
                             imbalanced problem and the choice must be explicit.
    fit_eval(clf, ...)       fit one classifier on one split and score it.

Statistics
    paired_t(a, b, corrected_for_resampling=True, test_frac=0.30)
                             Nadeau-Bengio corrected paired t-test. Repeated
                             random splits reuse data, so the naive paired t is
                             anti-conservative; the variance is inflated by
                             1/n + test_frac/(1 - test_frac).
    tost(a, b, margin)       two one-sided tests. A large p-value from a t-test
                             is NOT evidence that two methods are equivalent;
                             TOST at a pre-registered margin is.
    holm_bonferroni(pvals)   family-wise correction.
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy import sparse, stats


def _mmh3():
    import mmh3
    return mmh3


def hash_fn(name: str, seed: int):
    """Return f(str) -> non-negative int, for a named hash family."""
    name = name.lower()
    if name in ("murmur", "murmur3", "mmh3"):
        m = _mmh3()
        return lambda s: m.hash(s, seed, signed=False)
    if name == "xxhash":
        import xxhash
        return lambda s: xxhash.xxh64_intdigest(s, seed=seed)
    if name == "fnv1a":
        def f(s: str) -> int:
            h = 0xcbf29ce484222325 ^ (seed & 0xFFFFFFFF)
            for b in s.encode():
                h ^= b
                h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
            return h
        return f
    if name in ("sha256", "crypto"):
        pre = str(seed).encode()
        return lambda s: int.from_bytes(
            hashlib.sha256(pre + s.encode()).digest()[:8], "little")
    if name == "tabulation":
        rng = np.random.default_rng(seed)
        tab = rng.integers(0, 2 ** 63 - 1, size=(64, 256), dtype=np.int64)

        def f(s: str) -> int:
            h = 0
            for i, b in enumerate(s.encode()):
                h ^= int(tab[i % 64, b])
            return h & 0x7FFFFFFFFFFFFFFF
        return f
    raise ValueError(f"unknown hash family: {name}")


HASH_FAMILIES = ("murmur", "xxhash", "fnv1a", "tabulation", "sha256")


def sign_fn(seed: int):
    """+/-1 sign hash, independent of the bucket hash (different seed stream)."""
    m = _mmh3()
    return lambda s: 1.0 if (m.hash(s, seed, signed=False) & 1) else -1.0


def kmers(seq: str, k: int):
    return (seq[i:i + k] for i in range(len(seq) - k + 1))


def kmer_counts(seq: str, k: int) -> Counter:
    return Counter(kmers(seq, k))


def corpus_kmer_counts(seqs, k: int):
    """Per-sequence Counters plus the global vocabulary Counter."""
    per_seq, vocab = [], Counter()
    for s in seqs:
        c = kmer_counts(s, k)
        per_seq.append(c)
        vocab.update(c.keys())
    return per_seq, vocab


def unique_kmer_count(seqs, k: int) -> int:
    v = set()
    for s in seqs:
        v.update(kmers(s, k))
    return len(v)


def murmur2vec(per_seq_counts, m: int, signed: bool = False,
               hash_family: str = "murmur", seed: int = 0, sign_seed: int = 1,
               force_collide: dict | None = None) -> sparse.csr_matrix:
    """Hash k-mer counts into an m-dimensional sketch.

    force_collide: optional {kmer -> bucket} override, used by the collision_mechanism
    intervention experiment to deliberately collapse chosen k-mers together.
    """
    h = hash_fn(hash_family, seed)
    sg = sign_fn(sign_seed) if signed else None
    override = force_collide or {}
    cache = {}
    scache = {}

    indptr, indices, data = [0], [], []
    for counts in per_seq_counts:
        acc = {}
        for w, c in counts.items():
            j = override.get(w)
            if j is None:
                j = cache.get(w)
                if j is None:
                    j = h(w) % m
                    cache[w] = j
            if sg is not None:
                s = scache.get(w)
                if s is None:
                    s = sg(w)
                    scache[w] = s
                acc[j] = acc.get(j, 0.0) + s * c
            else:
                acc[j] = acc.get(j, 0.0) + c
        for j, v in acc.items():
            indices.append(j)
            data.append(v)
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(per_seq_counts), m))


def full_spectrum(per_seq_counts, vocab):
    """The exact k-mer spectrum -- this is the Spike2Vec baseline."""
    keys = sorted(vocab)
    idx = {w: i for i, w in enumerate(keys)}
    indptr, indices, data = [0], [], []
    for counts in per_seq_counts:
        for w, c in counts.items():
            indices.append(idx[w])
            data.append(c)
        indptr.append(len(indices))
    X = sparse.csr_matrix(
        (np.asarray(data, dtype=np.float32),
         np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(per_seq_counts), len(keys)))
    return X, keys


def spaced_kmer_counts(seq: str, pattern: str) -> Counter:
    """pattern is a mask like '1101' -- '1' keeps a position, '0' skips it."""
    span = len(pattern)
    keep = [i for i, ch in enumerate(pattern) if ch == "1"]
    out = Counter()
    for i in range(len(seq) - span + 1):
        out["".join(seq[i + j] for j in keep)] += 1
    return out


def gaussian_sketch(X, m: int, seed: int = 0):
    """Dense Gaussian JL projection -- collision_mechanism's negative control.

    Same target dimension as Murmur2Vec but no frequency-dependent collision
    structure, so it isolates whether the mechanism in Proposition 12 is what
    produces the SARS-CoV-2 advantage.
    """
    rng = np.random.default_rng(seed)
    R = rng.normal(0.0, 1.0 / np.sqrt(m), size=(X.shape[1], m)).astype(np.float32)
    return np.asarray(X @ R)


def collision_pct(vocab_keys, m: int, hash_family: str = "murmur",
                  seed: int = 0) -> float:
    """Definition of Eq. (1): 1 - (#occupied buckets)/(#unique k-mers)."""
    h = hash_fn(hash_family, seed)
    occupied = {h(w) % m for w in vocab_keys}
    return 1.0 - len(occupied) / len(vocab_keys)


def m_for_collision(vocab_keys, target_pct: float, hash_family: str = "murmur",
                    seed: int = 0, lo=None, hi=None) -> int:
    """Smallest m whose measured collision rate is <= target (binary search)."""
    U = len(vocab_keys)
    lo = lo or max(2, U // 200)
    hi = hi or max(lo * 2, 400 * U)
    if collision_pct(vocab_keys, hi, hash_family, seed) > target_pct:
        return hi
    while lo < hi:
        mid = (lo + hi) // 2
        if collision_pct(vocab_keys, mid, hash_family, seed) <= target_pct:
            hi = mid
        else:
            lo = mid + 1
    return lo


def m_heuristic(U: int, target_pct: float) -> int:
    """Closed-form safe bound for m: a rule instead of a grid search.

    Throwing U keys into m buckets, the expected number of occupied buckets is
    m(1 - (1 - 1/m)^U).  For U << m this expands to U - U(U-1)/(2m) + O(U^3/m^2),
    so the collision percentage of Eq. (1) is c ~= (U - 1) / (2m), giving

        m >= (U - 1) / (2c).

    One O(N) pass to count unique k-mers replaces the grid search entirely.
    """
    return int(np.ceil((U - 1) / (2.0 * target_pct)))


def m_exact(U: int, target_pct: float, lo: int = 2, hi=None) -> int:
    """Same target under the exact occupancy expectation, for comparison."""
    hi = hi or 500 * max(U, 2)

    def c(m):
        return 1.0 - (m * (1.0 - (1.0 - 1.0 / m) ** U)) / U
    while lo < hi:
        mid = (lo + hi) // 2
        if c(mid) <= target_pct:
            hi = mid
        else:
            lo = mid + 1
    return lo


CLASSIFIERS = ("NB", "KNN", "RF", "MLP", "LR", "DT")


def make_classifier(name: str, seed: int = 0, n_jobs: int = -1):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    if name == "NB":
        return GaussianNB()
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=n_jobs)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      n_jobs=n_jobs)
    if name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(100,), solver="adam",
                             random_state=seed, max_iter=300)
    if name == "LR":
        return LogisticRegression(penalty="l2", solver="lbfgs", max_iter=3000,
                                  random_state=seed, n_jobs=n_jobs)
    if name == "DT":
        return DecisionTreeClassifier(random_state=seed)
    raise ValueError(f"unknown classifier: {name}")


DENSE_REQUIRED = {"NB", "MLP"}

AUC_VARIANTS = ("ovr_macro", "ovr_weighted", "ovo_macro", "hardlabel_ovr_weighted")


def compute_metrics(y_true, y_pred, y_proba=None, n_classes=None,
                    all_auc_variants: bool = False) -> dict:
    """One definition of every metric, for every method, forever.

    The headline AUC is OvR macro: with B.1.1.7 at 48.1% of the SARS-CoV-2
    corpus, a weighted OvR average is dominated by the one easy class, which is
    how a method can look near-perfect on AUC while its macro-F1 says otherwise.
    all_auc_variants=True additionally reports the alternatives (including the
    hard-label version) so a discrepancy between two reported AUCs can be
    attributed rather than guessed at.
    """
    from sklearn import metrics as M

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    labels = list(range(n_classes))

    out = {
        "accuracy": float(M.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(M.balanced_accuracy_score(y_true, y_pred)),
        "precision_weighted": float(M.precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(M.recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(M.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(M.precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(M.recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(M.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(M.matthews_corrcoef(y_true, y_pred)),
        "kappa": float(M.cohen_kappa_score(y_true, y_pred)),
    }

    def _auc(proba, multi_class, average):
        try:
            return float(M.roc_auc_score(y_true, proba, multi_class=multi_class,
                                         average=average, labels=labels))
        except Exception:
            return float("nan")

    if y_proba is None:
        out["roc_auc_ovr_macro"] = float("nan")
        out["roc_auc_available"] = False
    else:
        P = np.asarray(y_proba, dtype=np.float64)
        row = P.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        P = P / row
        out["roc_auc_ovr_macro"] = _auc(P, "ovr", "macro")
        out["roc_auc_available"] = True
        if all_auc_variants:
            out["roc_auc_ovr_weighted"] = _auc(P, "ovr", "weighted")
            out["roc_auc_ovo_macro"] = _auc(P, "ovo", "macro")
            H = np.zeros_like(P)
            H[np.arange(len(y_pred)), y_pred] = 1.0
            out["roc_auc_hardlabel_ovr_weighted"] = _auc(H, "ovr", "weighted")
            out["roc_auc_hardlabel_ovr_macro"] = _auc(H, "ovr", "macro")
    return out


def per_class_report(y_true, y_pred, n_classes: int, class_names=None) -> dict:
    from sklearn import metrics as M
    labels = list(range(n_classes))
    p, r, f, s = M.precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    names = class_names if class_names is not None else [str(i) for i in labels]
    return {
        "class": list(names),
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f.tolist(),
        "support": s.tolist(),
        "confusion": M.confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def predict_with_proba(clf, X_test, n_classes: int):
    """Predictions plus a full n_classes-wide probability matrix, or None."""
    y_pred = clf.predict(X_test)
    proba = None
    if hasattr(clf, "predict_proba"):
        try:
            p = clf.predict_proba(X_test)
            proba = np.zeros((X_test.shape[0], n_classes), dtype=np.float64)
            for i, cls in enumerate(clf.classes_):
                proba[:, int(cls)] = p[:, i]
        except Exception:
            proba = None
    return y_pred, proba


def fit_eval(clf_name, X_train, X_test, y_train, y_test, n_classes,
             seed: int = 0, standardize: bool = False,
             all_auc_variants: bool = False) -> dict:
    from sklearn.preprocessing import StandardScaler

    Xtr, Xte = X_train, X_test
    if clf_name in DENSE_REQUIRED and sparse.issparse(Xtr):
        Xtr, Xte = Xtr.toarray(), Xte.toarray()
    if standardize:
        sc = StandardScaler(with_mean=not sparse.issparse(Xtr))
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)

    clf = make_classifier(clf_name, seed=seed)
    t0 = time.perf_counter()
    clf.fit(Xtr, y_train)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred, proba = predict_with_proba(clf, Xte, n_classes)
    test_time = time.perf_counter() - t0

    res = compute_metrics(y_test, y_pred, proba, n_classes,
                          all_auc_variants=all_auc_variants)
    res["train_time"] = train_time
    res["test_time"] = test_time
    res["classifier"] = clf_name
    return res


@dataclass
class TestResult:
    name: str
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    t: float
    df: int
    p: float
    cohens_d: float
    p_corrected: float = float("nan")
    correction: str = ""
    note: str = ""

    def as_dict(self):
        return asdict(self)


def paired_t(a, b, name: str = "", corrected_for_resampling: bool = True,
             test_frac: float = 0.30) -> TestResult:
    """Paired test on per-split scores.

    With repeated random train/test splits the observations share training
    data, so the naive paired t-test is anti-conservative.  Nadeau-Bengio
    inflates the variance by (1/n + test_frac/(1 - test_frac)) instead of 1/n.
    Reported by default; set corrected_for_resampling=False for the naive one.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    df = n - 1
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0:
        sd = 1e-12
    if corrected_for_resampling:
        factor = 1.0 / n + test_frac / (1.0 - test_frac)
        note = f"Nadeau-Bengio corrected (test_frac={test_frac})"
    else:
        factor = 1.0 / n
        note = "naive paired t"
    se = float(np.sqrt(factor) * sd)
    t = mean / se
    p = float(2 * stats.t.sf(abs(t), df))
    crit = stats.t.ppf(0.975, df)
    return TestResult(name=name, n=n, mean_diff=mean,
                      ci_low=mean - crit * se, ci_high=mean + crit * se,
                      t=float(t), df=df, p=p,
                      cohens_d=float(mean / sd), note=note)


def tost(a, b, margin: float, name: str = "", test_frac: float = 0.30,
         corrected_for_resampling: bool = True) -> dict:
    """Two one-sided tests for equivalence within +/- margin.

    Failing to reject a difference is not evidence of equivalence. This is the
    test that licenses a "tied" claim. The margin must be fixed before looking
    at the result; the defaults used here are 0.01 accuracy, 0.02 macro-F1.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = len(d)
    df = n - 1
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) or 1e-12
    factor = (1.0 / n + test_frac / (1.0 - test_frac)) if corrected_for_resampling else 1.0 / n
    se = float(np.sqrt(factor) * sd)
    t_low = (mean + margin) / se
    t_high = (mean - margin) / se
    p_low = float(stats.t.sf(t_low, df))
    p_high = float(stats.t.cdf(t_high, df))
    p = max(p_low, p_high)
    crit = stats.t.ppf(0.95, df)
    return {
        "name": name, "n": n, "margin": margin, "mean_diff": mean,
        "ci90_low": mean - crit * se, "ci90_high": mean + crit * se,
        "t_lower": float(t_low), "t_upper": float(t_high), "df": df,
        "p_lower": p_low, "p_upper": p_high, "p_tost": p,
        "equivalent": bool(p < 0.05),
    }


def holm_bonferroni(pvals, alpha: float = 0.05):
    """Returns (corrected p-values, reject flags) in the input order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    corrected = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        corrected[idx] = min(1.0, running)
    return corrected, corrected < alpha


@dataclass
class SplitSet:
    """A frozen, shareable collection of train/test index pairs."""
    kind: str
    dataset: str
    test_size: float
    splits: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def save(self, path):
        np.savez_compressed(
            path,
            kind=self.kind, dataset=self.dataset, test_size=self.test_size,
            n_splits=len(self.splits),
            **{f"tr{i}": np.asarray(tr) for i, (tr, _) in enumerate(self.splits)},
            **{f"te{i}": np.asarray(te) for i, (_, te) in enumerate(self.splits)},
            meta=np.array([repr(self.meta)], dtype=object))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=True)
        n = int(z["n_splits"])
        splits = [(z[f"tr{i}"], z[f"te{i}"]) for i in range(n)]
        meta = eval(str(z["meta"][0])) if "meta" in z else {}
        return SplitSet(kind=str(z["kind"]), dataset=str(z["dataset"]),
                        test_size=float(z["test_size"]), splits=splits, meta=meta)


def stratified_splits(y, n_splits: int = 20, test_size: float = 0.30,
                      seed: int = 0, dataset: str = "") -> SplitSet:
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size,
                                 random_state=seed)
    idx = np.arange(len(y))
    return SplitSet("stratified", dataset, test_size,
                    [(tr, te) for tr, te in sss.split(idx, y)],
                    {"seed": seed})


def grouped_splits(y, groups, n_splits: int = 20, test_size: float = 0.30,
                   seed: int = 0, dataset: str = "", kind: str = "grouped") -> SplitSet:
    """Group-aware split: no group (e.g. identity cluster) spans train and test."""
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                            random_state=seed)
    idx = np.arange(len(y))
    return SplitSet(kind, dataset, test_size,
                    [(tr, te) for tr, te in gss.split(idx, y, groups)],
                    {"seed": seed, "n_groups": int(len(set(groups)))})


def leave_one_class_out(y, classes=None, dataset: str = "") -> SplitSet:
    """Hold out one class entirely -- the 'a new lineage appears' stress test."""
    y = np.asarray(y)
    classes = classes if classes is not None else sorted(set(y.tolist()))
    idx = np.arange(len(y))
    splits = [(idx[y != c], idx[y == c]) for c in classes]
    return SplitSet("leave_one_class_out", dataset, float("nan"), splits,
                    {"classes": [int(c) for c in classes]})


def few_shot_splits(y, target_class, n_shots, n_repeats: int = 10,
                    test_size: float = 0.30, seed: int = 0,
                    dataset: str = "") -> SplitSet:
    """Cap one class at n_shots training examples; test set keeps the rest."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    tgt = idx[y == target_class]
    other = idx[y != target_class]
    splits = []
    for _ in range(n_repeats):
        rng.shuffle(other)
        cut = int(round(len(other) * (1 - test_size)))
        tr_other, te_other = other[:cut], other[cut:]
        pick = rng.permutation(tgt)
        tr = np.concatenate([tr_other, pick[:n_shots]])
        te = np.concatenate([te_other, pick[n_shots:]])
        splits.append((np.sort(tr), np.sort(te)))
    return SplitSet(f"few_shot_{n_shots}", dataset, test_size, splits,
                    {"target_class": int(target_class), "n_shots": n_shots})


def duplicate_audit(seqs, split) -> dict:
    """How much of the test set is an exact copy of something in training?

    This is the cheap version of the leakage check; the clustered split in
    similarity_splits.py is the expensive one.
    """
    tr, te = split
    train_hashes = {hashlib.md5(seqs[i].encode()).hexdigest() for i in tr}
    dup = [int(i) for i in te
           if hashlib.md5(seqs[i].encode()).hexdigest() in train_hashes]
    return {"n_test": len(te), "n_exact_dup": len(dup),
            "frac_exact_dup": len(dup) / max(1, len(te)),
            "dup_indices": dup}


DATA_ROOT = "data"


def load_covid(root: str = DATA_ROOT):
    """7,000 SARS-CoV-2 spike sequences with Pango lineage labels."""
    seqs = np.load(f"{root}/seq_data_7000.npy", allow_pickle=True)
    labs = np.load(f"{root}/seq_data_variant_names_7000.npy", allow_pickle=True)
    seqs = [str(s[0]) if np.ndim(s) else str(s) for s in seqs]
    labs = [str(l[0]) if np.ndim(l) else str(l) for l in labs]
    names = sorted(set(labs))
    to_i = {c: i for i, c in enumerate(names)}
    y = np.array([to_i[l] for l in labs], dtype=np.int32)
    return seqs, y, names


def clean_sequence(s: str, alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> str:
    """Uppercase, strip gaps, drop residues outside the standard 20."""
    keep = set(alphabet)
    return "".join(ch for ch in s.upper() if ch in keep)


def length_stats(seqs) -> dict:
    L = np.array([len(s) for s in seqs])
    return {
        "n": int(L.size), "min": int(L.min()), "max": int(L.max()),
        "mean": float(L.mean()), "median": float(np.median(L)),
        "frac_over_1022": float((L > 1022).mean()),
        "mean_frac_kept_at_1022": float(np.minimum(L, 1022).sum() / L.sum()),
    }
