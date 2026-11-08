"""Group counterparties within ONE statement.

Input is the cleaned rows of a single statement (clean.clean_statement). Credits and debits
are grouped separately, so a group is always credit-only or debit-only and a credit group
is never merged with a debit group because of a similar name.

Within one direction:
  1. Fixed names (atm, interest, self, tax, the empty name of GST-on-charge lines) are
     grouped by exact text only.
  2. Prefix names: a name of at least prefix_join_min characters that is the start of a
     longer name in the same statement, typically a cut-off ("nerana techno").
  3. The other names are clustered: distance = 1 - similarity, where similarity is
     fuzz_weight * RapidFuzz token_sort_ratio + (1 - fuzz_weight) * cosine of character
     n-gram TF-IDF vectors. Agglomerative clustering, merging while the distance is under
     the threshold. The distance matrix covers this statement's distinct names only.
  4. Each prefix name, longest first, joins the group of the longer names it starts if
     they are all in ONE group; if they span several groups it's ambiguous and it keeps a
     group of its own.

The group's name is its most frequent cleaned form, where a cut-off form counts towards
the longer form it starts (ties: the longest, then A-Z).
"""

from collections import Counter
from collections.abc import Sequence

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer

from statementscreen.patterns.clean import SELF
from statementscreen.patterns.config import DEFAULT, PatternConfig

EXACT_ONLY = frozenset({"", "atm", "interest", "gst payment", "advance tax", SELF})


def name_distances(names: Sequence[str], cfg: PatternConfig = DEFAULT) -> np.ndarray:
    """Symmetric distance matrix (0 = same, 1 = unrelated) between distinct names."""
    fz = process.cdist(names, names, scorer=fuzz.token_sort_ratio, workers=1) / 100.0
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(cfg.ngram_min, cfg.ngram_max))
    tfidf = vec.fit_transform(names)  # rows are L2-normalised, so X @ X.T is the cosine
    cos = (tfidf @ tfidf.T).toarray()
    sim = cfg.fuzz_weight * fz + (1 - cfg.fuzz_weight) * cos
    dist = np.clip(1.0 - sim, 0.0, 1.0)
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    return dist


def _is_prefix(short: str, long: str, cfg: PatternConfig) -> bool:
    return len(short) >= cfg.prefix_join_min and len(long) > len(short) and long.startswith(short)


def _is_cut_form(name: str, others: Sequence[str], was_cut: bool, cfg: PatternConfig) -> bool:
    """A cut-off form of a longer name: marked cut by cleaning, or ending mid-word in it.
    ("nerana technologies" is NOT a cut form of "nerana technologies limtied".)"""
    return any(_is_prefix(name, b, cfg) and (was_cut or b[len(name)] != " ") for b in others)


class NameClusterer:
    """Clusters one direction's distinct names; the distances are computed once, so several
    thresholds can be tried cheaply (the dev sweep)."""

    def __init__(self, names: Sequence[str], cfg: PatternConfig = DEFAULT,
                 cut_names: frozenset[str] = frozenset()) -> None:  # fmt: skip
        self.names = list(names)
        self.cfg = cfg
        self.prefix = [i for i, a in enumerate(self.names)
                       if _is_cut_form(a, self.names, a in cut_names, cfg)]  # fmt: skip
        prefix = set(self.prefix)
        self.full = [i for i in range(len(self.names)) if i not in prefix]
        self.dist = (name_distances([self.names[i] for i in self.full], cfg)
                     if len(self.full) > 1 else None)  # fmt: skip

    def labels(self, threshold: float) -> list[int]:
        labels = [-1] * len(self.names)
        if self.dist is not None:
            model = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                            linkage=self.cfg.linkage,
                                            distance_threshold=threshold)  # fmt: skip
            for i, lab in zip(self.full, model.fit_predict(self.dist), strict=True):
                labels[i] = int(lab)
        elif self.full:
            labels[self.full[0]] = 0
        nxt = max(labels, default=-1) + 1
        for i in sorted(self.prefix, key=lambda k: (-len(self.names[k]), self.names[k])):
            groups = {labels[j] for j, b in enumerate(self.names)
                      if labels[j] >= 0 and _is_prefix(self.names[i], b, self.cfg)}  # fmt: skip
            if len(groups) == 1:
                labels[i] = groups.pop()
            else:  # ambiguous: starts names in several groups
                labels[i], nxt = nxt, nxt + 1
        return labels


def _canonical(forms: list[str]) -> str:
    """The most frequent form, where a cut-off form ("shakapu technol") counts towards the
    longer form it starts ("shakapu technologies"). Ties: the longest, then A-Z."""
    counts = Counter(forms)
    full = [f for f in counts if not any(g != f and g.startswith(f) for g in counts)]
    total = {f: counts[f] for f in full}
    for f in counts:
        if f not in total:
            longer = [g for g in full if g.startswith(f)]
            total[min(longer, key=lambda g: (-counts[g], -len(g), g))] += counts[f]
    return min(total, key=lambda f: (-total[f], -len(f), f))


def group_ids(cleaned: pl.DataFrame, thresholds: Sequence[float],
              cfg: PatternConfig = DEFAULT) -> dict[float, list[str]]:  # fmt: skip
    """Group id per row (in the given row order) for each threshold, e.g. "C003", "D017"."""
    directions = cleaned["direction"].to_list()
    names = cleaned["counterparty"].to_list()
    cut = cleaned["counterparty_cut"].to_list()
    out = {t: [""] * len(names) for t in thresholds}
    for direction, tag in (("credit", "C"), ("debit", "D")):
        rows = [i for i, d in enumerate(directions) if d == direction]
        exact = sorted({names[i] for i in rows if names[i] in EXACT_ONLY})
        fuzzy = sorted({names[i] for i in rows if names[i] not in EXACT_ONLY})
        cut_names = frozenset(names[i] for i in rows if cut[i])
        clusterer = NameClusterer(fuzzy, cfg, cut_names) if fuzzy else None
        for t in thresholds:
            label = dict(zip(fuzzy, clusterer.labels(t), strict=True)) if clusterer else {}
            base = max(label.values(), default=-1) + 1
            label |= {name: base + k for k, name in enumerate(exact)}
            for i in rows:
                out[t][i] = f"{tag}{label[names[i]]:03d}"
    return out


def group_statement(cleaned: pl.DataFrame, cfg: PatternConfig = DEFAULT,
                    threshold: float | None = None) -> pl.DataFrame:  # fmt: skip
    """The cleaned rows plus group_id and group_name (the group's most frequent form)."""
    cleaned = cleaned.sort("row")
    t = cfg.group_threshold if threshold is None else threshold
    ids = group_ids(cleaned, [t], cfg)[t]
    out = cleaned.with_columns(pl.Series("group_id", ids, dtype=pl.String))
    names = out.group_by("group_id").agg(pl.col("counterparty"))
    canonical = {g: _canonical(forms) for g, forms in names.iter_rows()}
    return out.with_columns(
        pl.col("group_id").replace_strict(canonical, return_dtype=pl.String).alias("group_name")
    )
