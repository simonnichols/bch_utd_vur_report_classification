# =====================================================================
# VUR metrics from SAVED results only -- no earlier cells needed.
# Per-label Wilson CIs for precision/recall/F1/accuracy, macro + weighted
# aggregates with patient-level bootstrap CIs, and ordinal QWK/MAE with
# bootstrap CIs. Runs for BERT and both GPT models on the same test set.
# =====================================================================
import math
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

# ---------------- CONFIG ----------------
BERT_PRED_PATH = 'YOUR_PATH_HERE/test_predictions_v2_split_before_downsample.csv'
GPT_PRED_PATH = 'vur_3shot_predictions.csv'

MODELS = {   # name: (file, left prediction column, right prediction column)
    'BERT':           (BERT_PRED_PATH, 'BERT_Left', 'BERT_Right'),
    'GPT-4o (3-shot)': (GPT_PRED_PATH, 'gpt_4o_Left', 'gpt_4o_Right'),
    'GPT-5 (3-shot)':  (GPT_PRED_PATH, 'gpt_5_Left', 'gpt_5_Right'),
}
N_BOOT = 2000
SEED = 0
OUT_PER_LABEL = 'vur_metrics_per_label.csv'
OUT_AGGREGATE = 'vur_metrics_aggregate.csv'

GRADES = np.array([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
GRADE_RANK = {g: i for i, g in enumerate(GRADES)}
METRICS = ['precision', 'recall', 'f1', 'accuracy']


# ---------------- helpers ----------------
def wilson_ci(x, n, z=1.96):
    if n <= 0:
        return (np.nan, np.nan)
    phat = x / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def to_valid_grade(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if v in GRADE_RANK else np.nan


def confusion_counts(true_l, true_r, pred_l, pred_r):
    """One-vs-rest counts for all 20 labels (10 grades x 2 sides). Returns dict label -> (tp, fp, fn, tn)."""
    out = {}
    for side, t, p in [('Left', true_l, pred_l), ('Right', true_r, pred_r)]:
        for g in GRADES:
            ts, ps = (t == g), (p == g)
            tp = int(np.sum(ts & ps)); fp = int(np.sum(~ts & ps))
            fn = int(np.sum(ts & ~ps)); tn = int(np.sum(~ts & ~ps))
            out[f'{g}_{side}'] = (tp, fp, fn, tn)
    return out


def label_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0   # undefined (never predicted) counts as 0, sklearn convention
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    return precision, recall, f1, accuracy


def aggregate(counts):
    """Macro and support-weighted averages across the 20 labels.
    F1 is recomputed from the averaged precision/recall (keeps min(P,R) <= F1 <= max(P,R))."""
    per = {lab: label_metrics(*c) for lab, c in counts.items()}
    support = {lab: c[0] + c[2] for lab, c in counts.items()}
    total = sum(support.values())
    macro, weighted = {}, {}
    for i, m in [(0, 'precision'), (1, 'recall'), (3, 'accuracy')]:
        macro[m] = np.mean([per[lab][i] for lab in per])
        weighted[m] = sum(per[lab][i] * support[lab] for lab in per) / total
    for agg in (macro, weighted):
        p, r = agg['precision'], agg['recall']
        agg['f1'] = 2 * p * r / (p + r) if (p + r) else 0.0
    return macro, weighted


def ordinal(true_l, true_r, pred_l, pred_r):
    out = {}
    for side, t, p in [('Left', true_l, pred_l), ('Right', true_r, pred_r),
                       ('Both', np.concatenate([true_l, true_r]), np.concatenate([pred_l, pred_r]))]:
        tr = [GRADE_RANK[v] for v in t]; pr = [GRADE_RANK[v] for v in p]
        out[f'QWK {side}'] = cohen_kappa_score(tr, pr, weights='quadratic', labels=list(range(10)))
        out[f'MAE {side}'] = float(np.mean(np.abs(t - p)))
    return out


def evaluate(name, df, lcol, rcol):
    tl, tr = df['Left'].astype(float).values, df['Right'].astype(float).values
    pl, pr = df[lcol].values.astype(float), df[rcol].values.astype(float)

    # ---- per-label point estimates + Wilson CIs ----
    counts = confusion_counts(tl, tr, pl, pr)
    per_rows = []
    for lab, (tp, fp, fn, tn) in counts.items():
        p, r, f1, a = label_metrics(tp, fp, fn, tn)
        row = {'model': name, 'label': lab, 'true_n': tp + fn, 'predicted_n': tp + fp}
        for m, val, x, n in [('precision', p, tp, tp + fp), ('recall', r, tp, tp + fn),
                             ('f1', f1, 2 * tp, 2 * tp + fp + fn), ('accuracy', a, tp + tn, tp + fp + fn + tn)]:
            lo, hi = wilson_ci(x, n)
            row[m] = val if n > 0 else np.nan          # NaN = undefined (e.g. grade never predicted)
            row[f'{m}_lo'], row[f'{m}_hi'] = lo, hi
        per_rows.append(row)

    # ---- aggregates + ordinal, with patient-level bootstrap ----
    macro, weighted = aggregate(counts)
    ords = ordinal(tl, tr, pl, pr)
    rng = np.random.default_rng(SEED)
    n = len(df)
    boot = {k: [] for k in [f'macro {m}' for m in METRICS] + [f'weighted {m}' for m in METRICS] + list(ords)}
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        bm, bw = aggregate(confusion_counts(tl[idx], tr[idx], pl[idx], pr[idx]))
        bo = ordinal(tl[idx], tr[idx], pl[idx], pr[idx])
        for m in METRICS:
            boot[f'macro {m}'].append(bm[m]); boot[f'weighted {m}'].append(bw[m])
        for k, v in bo.items():
            boot[k].append(v)
    points = {**{f'macro {m}': macro[m] for m in METRICS},
              **{f'weighted {m}': weighted[m] for m in METRICS}, **ords}
    agg_rows = [{'model': name, 'metric': k, 'value': v,
                 'lo': np.percentile(boot[k], 2.5), 'hi': np.percentile(boot[k], 97.5)}
                for k, v in points.items()]
    return pd.DataFrame(per_rows), pd.DataFrame(agg_rows)


# ---------------- load, check, run ----------------
frames, id_sets = {}, {}
for name, (path, lcol, rcol) in MODELS.items():
    df = pd.read_csv(path)
    df[lcol] = df[lcol].map(to_valid_grade)
    df[rcol] = df[rcol].map(to_valid_grade)
    bad = df[[lcol, rcol]].isnull().any(axis=1)
    print(f'{name}: {len(df)} rows loaded, {bad.sum()} with missing/off-scale grades (excluded)')
    frames[name] = (df[~bad].reset_index(drop=True), lcol, rcol)
    id_sets[name] = set(df['Accession Number'])

# every model must have been scored on the same patients
ref = next(iter(id_sets.values()))
for name, ids in id_sets.items():
    assert ids == ref, f'{name} was not run on the same test set as the others'
h = hashlib.md5(','.join(sorted(map(str, ref))).encode()).hexdigest()
print(f'All models share the same {len(ref)}-patient test set (hash {h})\n')

all_per, all_agg = [], []
for name, (df, lcol, rcol) in frames.items():
    per, agg = evaluate(name, df, lcol, rcol)
    all_per.append(per); all_agg.append(agg)

per_label = pd.concat(all_per, ignore_index=True)
aggregate_tbl = pd.concat(all_agg, ignore_index=True)
per_label.to_csv(OUT_PER_LABEL, index=False)
aggregate_tbl.to_csv(OUT_AGGREGATE, index=False)

# ---------------- readable printout ----------------
fmt = lambda v, lo, hi: 'n/a (never predicted)' if pd.isna(v) else f'{v:.4f} [{lo:.4f}, {hi:.4f}]'
for name in MODELS:
    print('=' * 100); print(name); print('=' * 100)
    print('Per-label (Wilson 95% CI):')
    for _, r in per_label[per_label.model == name].iterrows():
        print(f"  {r.label:10s} true n={r.true_n:<4d} " +
              '  '.join(f"{m[:4]}={fmt(r[m], r[m + '_lo'], r[m + '_hi'])}" for m in METRICS))
    print('Aggregate (patient-level bootstrap 95% CI):')
    for _, r in aggregate_tbl[aggregate_tbl.model == name].iterrows():
        print(f"  {r.metric:22s} {r.value:.4f} [{r.lo:.4f}, {r.hi:.4f}]")
    print()
print(f'Saved {OUT_PER_LABEL} and {OUT_AGGREGATE}')
