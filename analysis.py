"""DP-SGD (momentum, tuned) vs DP-Adam (defaults) on UCI Adult at small epsilon."""
import json, os, sys, time
import numpy as np
from scipy.special import gammaln, logsumexp, expit
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED, DELTA, EPS_LIST = 42, 1e-5, [0.5, 1.0, 2.0]
STEPS, BATCH = 300, 1024
FINAL_SEEDS, TUNE_SEEDS = [42, 43, 44, 45, 46], [100, 101]
CLIPS = [0.1, 0.3, 1.0, 3.0]
GRIDS = {"sgdm": [0.03, 0.1, 0.3, 1.0, 3.0], "adam": [0.001, 0.003, 0.01, 0.03, 0.1]}
np.random.seed(SEED)

# ---------- data (same loading/preprocessing/split for all arms) ----------
for attempt in range(4):
    try:
        ds = fetch_openml(data_id=1590, as_frame=True, parser="auto"); break
    except Exception as e:
        print("download failed, retrying:", e); time.sleep(2 ** attempt)
else:
    sys.exit("could not download Adult")
X, y = ds.data.copy(), (ds.target.astype(str).str.contains(">")).astype(int).values
cat = [c for c in X.columns if X[c].dtype.name in ("category", "object")]
num = [c for c in X.columns if c not in cat]
for c in cat: X[c] = X[c].astype(str)
for c in num: X[c] = X[c].fillna(X[c].median())
print("rows:", len(X), "features:", X.shape[1])
Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.25, stratify=ytv, random_state=SEED)  # 60/20/20
pre = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                         ("n", StandardScaler(), num)]).fit(Xtr)
def prep(df): return np.hstack([pre.transform(df), np.ones((len(df), 1))])  # bias column
Xtr, Xva, Xte = prep(Xtr), prep(Xva), prep(Xte)
n, d = Xtr.shape
xnorm = np.linalg.norm(Xtr, axis=1)
Q = BATCH / n

# ---------- simple RDP accountant (Poisson-subsampled Gaussian, integer orders) ----------
ORDERS = np.arange(2, 257)
def epsilon(sigma):
    best = np.inf
    for a in ORDERS:
        k = np.arange(a + 1)
        lg = (gammaln(a + 1) - gammaln(k + 1) - gammaln(a - k + 1) + (a - k) * np.log1p(-Q)
              + k * np.log(Q) + (k * k - k) / (2 * sigma ** 2))
        best = min(best, STEPS * logsumexp(lg) / (a - 1) + np.log(1 / DELTA) / (a - 1))
    return best
def noise_multiplier(target):
    lo, hi = 0.3, 500.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if epsilon(mid) > target else (lo, mid)
    return hi  # eps(hi) <= target

# ---------- DP training: per-example clipping + Gaussian noise ----------
def dp_train(opt, lr, clip, sigma, seed):
    rng = np.random.default_rng(seed)
    w, m, v = np.zeros(d), np.zeros(d), np.zeros(d)
    for t in range(1, STEPS + 1):
        idx = np.flatnonzero(rng.random(n) < Q)  # Poisson batch
        r = expit(Xtr[idx] @ w) - ytr[idx]  # per-example grad = r_i * x_i
        f = np.minimum(1.0, clip / np.maximum(np.abs(r) * xnorm[idx], 1e-12))
        g = (Xtr[idx].T @ (r * f) + rng.normal(0, sigma * clip, d)) / BATCH
        if opt == "adam":
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            w -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        else:  # SGD with momentum 0.9
            m = 0.9 * m + g; w -= lr * m
    return w

def evaluate(w, Xe, ye):
    s = Xe @ w
    return float(np.mean((s > 0) == ye)), float(roc_auc_score(ye, s))

def tune(opt, sigma):  # 5x4 grid, selected on validation accuracy only
    best, cfg = -1, None
    for lr in GRIDS[opt]:
        for c in CLIPS:
            acc = np.mean([evaluate(dp_train(opt, lr, c, sigma, s), Xva, yva)[0] for s in TUNE_SEEDS])
            if acc > best: best, cfg = acc, (lr, c)
    return cfg

# ---------- experiment ----------
arms = ["baseline", "improved", "tuned_adam"]
res = {a: {"acc": np.zeros((len(EPS_LIST), len(FINAL_SEEDS))), "auc": np.zeros((len(EPS_LIST), len(FINAL_SEEDS)))} for a in arms}
chosen = {}
for i, eps in enumerate(EPS_LIST):
    sigma = noise_multiplier(eps)
    print(f"eps={eps}: sigma={sigma:.3f} (check eps={epsilon(sigma):.3f})", flush=True)
    cfgs = {"baseline": ("adam", 0.01, 1.0), "improved": ("sgdm", *tune("sgdm", sigma)),
            "tuned_adam": ("adam", *tune("adam", sigma))}
    for a, cf in cfgs.items():
        chosen[f"{a}_eps{eps}"] = list(cf)
        for j, s in enumerate(FINAL_SEEDS):
            res[a]["acc"][i, j], res[a]["auc"][i, j] = evaluate(dp_train(cf[0], cf[1], cf[2], sigma, s), Xte, yte)
        print(f"  {a} {cf}: test acc {res[a]['acc'][i].mean():.4f}", flush=True)

ref = LogisticRegression(max_iter=500).fit(Xtr[:, :-1], ytr)
metrics = {}
for a in arms:
    per_seed = res[a]["acc"].mean(axis=0)  # mean over eps, one value per seed
    key = {"baseline": "baseline", "improved": "improved", "tuned_adam": "tuned_adam"}[a]
    if a != "tuned_adam":
        metrics[f"{key}_score"] = float(per_seed.mean()); metrics[f"{key}_score_std"] = float(per_seed.std())
    metrics[f"{key}_accuracy"] = float(per_seed.mean())
    metrics[f"{key}_roc_auc"] = float(res[a]["auc"].mean())
    for i, eps in enumerate(EPS_LIST):
        metrics[f"{key}_acc_eps{eps}"] = float(res[a]["acc"][i].mean())
metrics["tuned_adam_accuracy_std"] = float(res["tuned_adam"]["acc"].mean(axis=0).std())
metrics["improvement"] = metrics["improved_score"] - metrics["baseline_score"]
metrics["improved_minus_tuned_adam_accuracy"] = metrics["improved_accuracy"] - metrics["tuned_adam_accuracy"]
metrics["majority_class_accuracy"] = float(max(yte.mean(), 1 - yte.mean()))
metrics["nonprivate_logreg_accuracy"] = float(ref.score(Xte[:, :-1], yte))
if not all(np.isfinite(v) for v in metrics.values()) or metrics["improved_score"] <= 0.5:
    sys.exit("no meaningful metrics")

os.makedirs("figures", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4))
for k, a in enumerate(arms):
    ax.bar(np.arange(3) + 0.27 * k, res[a]["acc"].mean(axis=1), 0.27,
           yerr=res[a]["acc"].std(axis=1), label=["DP-Adam default", "DP-SGD-M tuned", "DP-Adam tuned"][k])
ax.set_xticks(np.arange(3) + 0.27); ax.set_xticklabels([f"eps={e}" for e in EPS_LIST])
ax.set_ylim(0.7, 0.88); ax.set_ylabel("test accuracy"); ax.legend(); fig.tight_layout()
fig.savefig("figures/accuracy_by_epsilon.png", dpi=120)

out = {
    "question": "Does a carefully tuned non-adaptive DP-SGD (with momentum) match or beat a default-setting DP-Adam at the same high-privacy budgets (small epsilon) on a public tabular task?",
    "method": "Numpy logistic regression on one-hot/standardised Adult with per-example clipping, Gaussian noise, Poisson batches; RDP accountant sets sigma for eps in {0.5,1,2}, delta=1e-5; same steps/batch for all arms; tuned arms use a 5x4 (lr x clip) grid scored on a validation split.",
    "dataset": {"name": "UCI Adult income (OpenML 1590)", "source_url": "https://www.openml.org/d/1590", "n_rows": int(len(X)),
                "n_train": int(n), "n_test": int(len(yte)), "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "test accuracy (mean over eps in {0.5,1,2} and 5 noise seeds)",
    "higher_is_better": True,
    "baseline": "DP-Adam with default settings (lr 0.01, clip 1.0, betas 0.9/0.999)",
    "improved": "DP-SGD with momentum 0.9, learning rate and clip norm chosen per epsilon from a 5x4 grid on the validation set",
    "metrics": metrics,
    "notes": "Split: 20% test held out once, remaining 80% split into 60% train / 20% validation (of total); tuning used validation only, test used once at the end. Hyperparameter-search privacy cost is not accounted for. Tuning used 2 separate noise seeds (100,101); final runs use seeds 42-46 (std is across seeds of the eps-averaged accuracy). Accountant is a simple RDP bound with integer orders. Tuned DP-Adam is a reference arm. Chosen hyperparameters: " + json.dumps(chosen),
}
with open("results.json", "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(json.dumps(metrics, indent=2))
