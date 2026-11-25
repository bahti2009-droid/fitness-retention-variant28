"""Variant 28 — Fitness app retention. All observations and labels are SYNTHETIC.

Run: python -m streamlit run app.py
Recommended Python: 3.11/3.12. No API keys, external dataset or network needed at runtime.
Workflow: DATA -> SPLIT -> PREPARE -> TRAIN -> VALIDATE -> TEST -> DECIDE.
Reports and presentation are intentionally deferred to Step 2.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import zipfile
from importlib.metadata import version
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve, silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["workouts", "streak", "notifications"]
TARGET = "active_next_week"
GENERATOR_VERSION = "fitness-synthetic-v1"
FEATURE_INFO = pd.DataFrame([
    ["workouts", "Workouts in the past 7 days", "integer count", "0–14"],
    ["streak", "Consecutive active days as of the end of this week", "days", "0–120"],
    ["notifications", "Push notifications received in the past 7 days", "integer count", "0–60"],
    [TARGET, "At least one workout next week (simulated)", "0/1", "0–1"],
], columns=["Feature", "Definition", "Units", "Range"])
ASSUMPTIONS = """
**Source: an original simulation, not real fitness application records.**
The observation unit is one unique user at the end of a simulated week.
All inputs precede the following week. Each user appears once.
Three overlapping engagement regimes are used only inside the generator;
their IDs are never included in the dataset, models, or cluster evaluation.
Workouts follow a Poisson distribution, streak follows a Gamma distribution,
and notifications follow a Poisson distribution.
Unusual combinations are introduced into 3% of rows: few workouts with a long streak,
many notifications, or high activity. These are plausible unusual observations,
not verified or labeled real-world anomalies.
Next-week activity probability is defined by a nonlinear logistic function:
`z = -2 + 0.55*workouts + 0.9*log1p(streak) + 0.07*min(notifications,10)`
` - 0.06*max(notifications-10,0) - 0.10*max(workouts-9,0)^2 + Normal(0,0.75)`.
The label is sampled from Bernoulli(sigmoid(z)): it is related to the inputs
but is not perfectly determined by them. After generating the label, feature values
are independently masked with NaN at the selected probability (MCAR, a simplification).
All distributions and relationships are educational assumptions, not empirical findings.
The OpenML link in the worksheet is a catalog, not the source of these observations.
"""
LIMITATIONS = """
- Metrics describe performance on simulated data and do not establish real-world performance.
- A random split suits the independent simulated users. Real logs require a temporal split and separation by user_id.
- MCAR missingness is simpler than real missingness; seasonality, drift, and social factors are absent.
- Cluster counts are exploratory; silhouette is not accuracy. Seed stability does not establish stability under resampling.
- PCA is linear; t-SNE/UMAP distort distances, group sizes, and positions. Visual separation does not prove segments exist.
- DBSCAN noise and Isolation Forest flags do not establish fraud or measurement error.
- Notification relationships reflect generator assumptions. Causal campaign effects require an A/B test.
Improvement: collect consented, anonymized logs, validate metric definitions,
use external temporal evaluation, assess probability calibration, and test retention campaigns experimentally.
"""


@st.cache_data(max_entries=12, show_spinner=False)
def generate_data(n: int = 800, seed: int = 42, missing: float = 0.05) -> pd.DataFrame:
    """Generate reproducible synthetic inputs AND labels; never present as real logs."""
    if not 600 <= n <= 1000 or not 0 <= missing <= 0.20:
        raise ValueError("n must be 600–1000; missing must be 0–0.20")
    rng = np.random.default_rng(seed)
    mode = rng.choice(3, n, p=[0.38, 0.42, 0.20])
    workouts = np.clip(rng.poisson(np.array([0.8, 3.1, 6.0])[mode]), 0, 14).astype(float)
    streak = np.clip(np.rint(rng.gamma(1.7, np.array([1.0, 4.0, 12.0])[mode])), 0, 120)
    # Streak counts any app activity, not only workouts; zero workouts can coexist with streak.
    notifications = np.clip(rng.poisson(np.array([14., 8., 4.])[mode]), 0, 60).astype(float)
    unusual = rng.choice(n, max(1, int(n * 0.03)), replace=False)
    for i, idx in enumerate(unusual):
        if i % 3 == 0:
            workouts[idx], streak[idx] = rng.integers(0, 2), rng.integers(60, 121)
        elif i % 3 == 1:
            notifications[idx] = rng.integers(35, 61)
        else:
            workouts[idx], streak[idx] = rng.integers(10, 15), rng.integers(40, 91)
    z = (-2 + 0.55 * workouts + 0.9 * np.log1p(streak)
         + 0.07 * np.minimum(notifications, 10)
         - 0.06 * np.maximum(notifications - 10, 0)
         - 0.10 * np.maximum(workouts - 9, 0) ** 2 + rng.normal(0, 0.75, n))
    target = rng.binomial(1, 1 / (1 + np.exp(-np.clip(z, -30, 30))))
    values = np.column_stack([workouts, streak, notifications])
    values[rng.random(values.shape) < missing] = np.nan
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, "user_id", [f"SYN-{i + 1:04d}" for i in range(n)])
    frame[TARGET] = target
    return frame


def prepare(strategy: str = "median", scaled: bool = True) -> Pipeline:
    """Fresh preprocessing for each fit; unsupervised analysis never reuses supervised fit."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy=strategy, keep_empty_features=True)),
        ("scaler", StandardScaler() if scaled else "passthrough"),
    ])


def metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, prob),
    }


@st.cache_resource(max_entries=6, show_spinner=False)
def train_supervised(frame: pd.DataFrame, seed: int) -> dict:
    """Steps 4–11: select with train CV only; final test is never used for tuning."""
    X = frame[FEATURES]
    y = frame[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(Xtr, ytr))
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=3, n_jobs=1, random_state=seed),
    }
    models, rows = {}, []
    for name, estimator in candidates.items():
        model = Pipeline(prepare().steps + [("classifier", estimator)])
        scores = cross_val_score(model, Xtr, ytr, cv=folds, scoring="f1", n_jobs=1)
        rows.append({"model": name, **{f"fold_{i + 1}": s for i, s in enumerate(scores)},
                     "cv_f1_mean": scores.mean(), "cv_f1_sd": scores.std(ddof=0)})
        models[name] = model
    cv = pd.DataFrame(rows).set_index("model")
    winner = cv["cv_f1_mean"].idxmax()  # Deterministic tie-break: Logistic Regression first.
    selected = models[winner]
    # These predictions are strictly OOF on training rows, for the business threshold only.
    oof = cross_val_predict(clone(selected), Xtr, ytr, cv=folds,
                            method="predict_proba", n_jobs=1)[:, 1]
    # Selection is already frozen. Fit both for descriptive test comparison/interpretation.
    test_rows, probabilities, reports = [], {}, {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        probabilities[name] = p
        test_rows.append({"model": name, **metrics(yte, (p >= .5).astype(int), p)})
        reports[name] = pd.DataFrame(classification_report(
            yte, (p >= .5).astype(int), output_dict=True, zero_division=0)).T
    baseline = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    baseline_p = baseline.predict_proba(Xte)[:, list(baseline.classes_).index(1)]
    test_rows.append({"model": "Dummy: majority", **metrics(yte, baseline.predict(Xte), baseline_p)})
    return dict(models=models, winner=winner, cv=cv, test=pd.DataFrame(test_rows).set_index("model"),
                probs=probabilities, reports=reports, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, oof=oof)


def cost_threshold(result: dict, fp_cost: float, fn_cost: float) -> tuple[float, pd.DataFrame]:
    """Tune on training OOF predictions, never test. Costs are user assumptions."""
    y = result["ytr"].to_numpy()
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = result["oof"] >= threshold
        fp = int(((y == 0) & pred).sum())
        fn = int(((y == 1) & ~pred).sum())
        rows.append({"threshold": threshold, "FP": fp, "FN": fn,
                     "cost_per_user": (fp_cost * fp + fn_cost * fn) / len(y)})
    table = pd.DataFrame(rows)
    return float(table.loc[table.cost_per_user.idxmin(), "threshold"]), table


def safe_silhouette(values: np.ndarray, labels: np.ndarray) -> float:
    groups = len(np.unique(labels))
    return float(silhouette_score(values, labels)) if 1 < groups < len(values) else float("nan")


@st.cache_data(max_entries=8, show_spinner=False)
def unsupervised(frame: pd.DataFrame, seed: int) -> dict:
    """Exploratory full-cohort fitting: target and ID are excluded by explicit allowlist."""
    X = frame[FEATURES].copy()
    pre = prepare()
    Z = pre.fit_transform(X)
    rows = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
        rows.append({"K": k, "inertia": km.inertia_, "silhouette": safe_silhouette(Z, km.labels_)})
    search = pd.DataFrame(rows)
    suggested = int(search.loc[search.silhouette.idxmax(), "K"])
    comparison = []
    # Hold K AND seed fixed so only preprocessing changes.
    for strategy in ["median", "mean"]:
        for scaled in [False, True]:
            A = prepare(strategy, scaled).fit_transform(X)
            lab = KMeans(n_clusters=suggested, n_init=10, random_state=seed).fit_predict(A)
            comparison.append({"imputer": strategy, "scaled": scaled, "K": suggested,
                               "silhouette": safe_silhouette(A, lab)})
    pca = PCA().fit(Z)
    projected = pca.transform(Z)[:, :2]
    loadings = pd.DataFrame(pca.components_[:2], columns=FEATURES, index=["PC1", "PC2"])
    return dict(Z=Z, search=search, suggested=suggested, precomparison=pd.DataFrame(comparison),
                pca=projected, variance=pca.explained_variance_ratio_, loadings=loadings)


@st.cache_data(max_entries=20, show_spinner=False)
def fit_kmeans(Z: np.ndarray, k: int, seed: int) -> dict:
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    seeds = sorted(set([seed, 0, 1, 2, 3, 4]))
    labels, rows = [], []
    for s in seeds:
        m = KMeans(n_clusters=k, n_init=10, random_state=s).fit(Z)
        labels.append(m.labels_)
        rows.append({"seed": s, "inertia": m.inertia_, "silhouette": safe_silhouette(Z, m.labels_),
                     "ARI_vs_reference": adjusted_rand_score(model.labels_, m.labels_)})
    ari = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
    return dict(labels=model.labels_, centers=model.cluster_centers_,
                stability=pd.DataFrame(rows), pairwise_ari_mean=float(np.mean(ari)))


@st.cache_data(max_entries=16, show_spinner=False)
def density_analysis(Z: np.ndarray, eps: float, min_samples: int) -> dict:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)
    rows = []
    for e in sorted(set([0.4, 0.7, 1.0, round(eps, 2)])):
        for ms in sorted(set([3, 5, 8, min_samples])):
            lab = DBSCAN(eps=e, min_samples=ms).fit_predict(Z)
            mask = lab != -1
            rows.append({"eps": e, "min_samples": ms,
                         "clusters": len(set(lab) - {-1}), "noise": int((~mask).sum()),
                         "noise_fraction": float((~mask).mean()),
                         "silhouette_without_noise": safe_silhouette(Z[mask], lab[mask])})
    return dict(labels=labels, sensitivity=pd.DataFrame(rows))


@st.cache_data(max_entries=12, show_spinner=False)
def anomaly_analysis(Z: np.ndarray, seed: int, contamination: float) -> dict:
    flags, scores, rows = {}, {}, []
    for c in sorted(set([0.02, 0.05, 0.10, contamination])):
        model = IsolationForest(n_estimators=200, contamination=c, random_state=seed, n_jobs=1).fit(Z)
        flags[c] = model.predict(Z) == -1
        scores[c] = model.decision_function(Z)
        rows.append({"contamination": c, "flagged": int(flags[c].sum()),
                     "flagged_fraction": float(flags[c].mean())})
    return dict(flags=flags, scores=scores, sensitivity=pd.DataFrame(rows))


@st.cache_data(max_entries=16, show_spinner=False)
def nonlinear_projection(Z: np.ndarray, seed: int, method: str, neighborhood: int, min_dist: float = .1) -> dict:
    if method == "t-SNE":
        embedding = TSNE(n_components=2, perplexity=neighborhood, random_state=seed,
                         init="pca", learning_rate="auto", max_iter=1000).fit_transform(Z)
    else:
        # Lazy import: other sections stay usable if UMAP installation is broken.
        from umap import UMAP
        embedding = UMAP(n_neighbors=neighborhood, min_dist=min_dist,
                         random_state=seed, n_jobs=1).fit_transform(Z)
    return dict(embedding=embedding, trustworthiness=float(trustworthiness(Z, embedding, n_neighbors=10)))


def plot(figure: go.Figure) -> None:
    figure.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=55, b=20))
    st.plotly_chart(figure, use_container_width=True)


def scatter(embedding: np.ndarray, labels: np.ndarray, title: str) -> None:
    view = pd.DataFrame({"axis_1": embedding[:, 0], "axis_2": embedding[:, 1],
                         "group": labels.astype(str)})
    plot(px.scatter(view, x="axis_1", y="axis_2", color="group", title=title, opacity=.7))


def csv_download(frame: pd.DataFrame, name: str, label: str = "Download CSV") -> None:
    st.download_button(label, frame.to_csv(index=False).encode("utf-8-sig"),
                       file_name=name, mime="text/csv", key=f"download_{name}")


def show_cm(y: pd.Series, pred: np.ndarray, title: str) -> np.ndarray:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    plot(px.imshow(cm, text_auto=True, x=["0: inactive", "1: active"],
                   y=["0: inactive", "1: active"], labels=dict(x="Predicted", y="Actual"), title=title))
    return cm


def page_data(frame: pd.DataFrame) -> None:
    overview, eda, quality = st.tabs(["Data and provenance", "EDA", "Data quality"])
    with overview:
        st.markdown(ASSUMPTIONS)
        st.dataframe(FEATURE_INFO, hide_index=True, use_container_width=True)
        a, b, c = st.columns(3)
        a.metric("Users", len(frame))
        b.metric("Features", len(FEATURES))
        c.metric("Active next week", f"{frame[TARGET].mean():.1%}")
        st.dataframe(frame.head(30), use_container_width=True)
        csv_download(frame, "fitness_SYNTHETIC.csv", "Download the original synthetic dataset")
    with eda:
        st.caption("Classification EDA uses training data only: the test set is excluded from feature and model selection.")
        train, _ = train_test_split(frame, test_size=.2, stratify=frame[TARGET],
                                   random_state=st.session_state["seed"])
        for col in FEATURES:
            plot(px.histogram(train, x=col, nbins=30, title=f"Train: {col}"))
            s = train[col].dropna()
            st.write(f"{col}: median {s.median():.1f}, 90th percentile {s.quantile(.9):.1f}, "
                     f"zeros {(s == 0).mean():.1%}; missing values are excluded from the histogram.")
        counts = train[TARGET].value_counts().rename_axis("class").reset_index(name="count")
        plot(px.bar(counts, x="class", y="count", title="Train: class balance"))
        plot(px.imshow(train[FEATURES].corr(), text_auto=".2f", zmin=-1, zmax=1,
                       title="Train: Pearson correlations (pairwise non-missing values)"))
        st.write("Different streak, workout, and notification scales justify StandardScaler for distance methods and LR. "
                 "Right tails deserve inspection. Correlation does not establish causation.")
    with quality:
        st.dataframe(frame[FEATURES].describe().T, use_container_width=True)
        st.dataframe(pd.DataFrame({"dtype": frame.dtypes.astype(str), "missing": frame.isna().sum(),
                                   "missing_fraction": frame.isna().mean()}), use_container_width=True)
        st.write(f"Duplicate user_id values: {frame.user_id.duplicated().sum()}. "
                 f"Rows with all features missing: {frame[FEATURES].isna().all(axis=1).sum()}.")
        st.info("The original table is preserved. Supervised: medians and scaling parameters are fitted within each CV fold. "
                "Unsupervised: separate preprocessing of the entire exploratory cohort; target and user_id are excluded.")


def page_supervised(result: dict) -> None:
    validation, test, explain, decision = st.tabs(["CV and model selection", "Final test evaluation", "Interpretation", "Error costs"])
    with validation:
        st.write(f"Train: {len(result['Xtr'])}; test: {len(result['Xte'])}. Split 80/20, stratify=y; the seed is fixed.")
        st.code("Pipeline(SimpleImputer(median) → StandardScaler → classifier)\nStratifiedKFold(n_splits=3, shuffle=True) → select by mean F1")
        st.dataframe(result["cv"], use_container_width=True)
        chart = result["cv"].reset_index()
        plot(px.bar(chart, x="model", y="cv_f1_mean", error_y="cv_f1_sd", range_y=[0, 1], title="Training-only CV F1 ± SD"))
        st.success(f"Selected by CV: {result['winner']}. The test set was not used for this decision.")
        st.caption("F1 is reported for class 1 (active). SD across three folds measures variability, not a confidence interval.")
    with test:
        st.dataframe(result["test"], use_container_width=True)
        st.caption("Both models are shown for descriptive comparison after selection. Test results do not change the winner. Threshold: 0.5.")
        for name, probability in result["probs"].items():
            with st.expander(name, expanded=name == result["winner"]):
                show_cm(result["yte"], (probability >= .5).astype(int), name)
                st.dataframe(result["reports"][name], use_container_width=True)
        roc = go.Figure()
        for name, p in result["probs"].items():
            fpr, tpr, _ = roc_curve(result["yte"], p)
            roc.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name}: AUC={roc_auc_score(result['yte'], p):.3f}")
        roc.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash"))
        roc.update_layout(title="Held-out ROC", xaxis_title="False positive rate", yaxis_title="True positive rate")
        plot(roc)
    with explain:
        lr = result["models"]["Logistic Regression"].named_steps["classifier"]
        rf = result["models"]["Random Forest"].named_steps["classifier"]
        coefficients = pd.DataFrame({"feature": FEATURES, "coefficient": lr.coef_[0]}).sort_values("coefficient")
        importance = pd.DataFrame({"feature": FEATURES, "importance": rf.feature_importances_}).sort_values("importance")
        plot(px.bar(coefficients, x="coefficient", y="feature", orientation="h", title="LR: standardized coefficients"))
        plot(px.bar(importance, x="importance", y="feature", orientation="h", title="RF: impurity-based importance"))
        st.write("A positive coefficient increases activity log-odds while other inputs are held fixed. "
                 "RF importance has no direction and can be biased by feature cardinality and correlations. Causality is not established.")
    with decision:
        st.markdown("**Class 1 means an active user.** FP: activity was predicted, but the user became inactive; "
                    "a retention opportunity may be missed. FN: an active user was predicted to become inactive; "
                    "an unnecessary campaign may waste budget or annoy the user.")
        fp = st.number_input("FP cost (assumed units)", min_value=1., max_value=10000., value=5.)
        fn = st.number_input("FN cost (assumed units)", min_value=1., max_value=10000., value=1.)
        threshold, costs = cost_threshold(result, fp, fn)
        st.metric("Cost-minimizing threshold on training OOF predictions", f"{threshold:.2f}")
        plot(px.line(costs, x="threshold", y="cost_per_user", title="Cost per user: training OOF"))
        st.caption("Costs are scenario assumptions. The threshold is selected without the test set. The optimized OOF minimum is optimistic; "
                   "confirmation requires a separate validation cohort. Primary metrics retain the 0.5 threshold.")
        p = result["probs"][result["winner"]]
        cm = show_cm(result["yte"], (p >= threshold).astype(int), "Test: training-selected business threshold")
        st.write(f"Test cost: {(fp * cm[0, 1] + fn * cm[1, 0]) / len(p):.3f} units/user.")


def page_clusters(frame: pd.DataFrame, u: dict, seed: int) -> None:
    choice, density, preprocessing = st.tabs(["K-Means", "DBSCAN", "Preprocessing"])
    with choice:
        st.dataframe(u["search"], hide_index=True, use_container_width=True)
        a, b = st.columns(2)
        with a:
            plot(px.line(u["search"], x="K", y="inertia", markers=True, title="Elbow: within-cluster inertia"))
        with b:
            plot(px.line(u["search"], x="K", y="silhouette", markers=True, title="Silhouette: standardized 3D features"))
        st.write(f"Candidate with the highest silhouette: K={u['suggested']}. "
                 "Compare this with the inertia elbow and profile interpretability; there is no universal true K here.")
        k = st.slider("K for cluster profile analysis", 2, 8, u["suggested"], key="cluster_k")
        km = fit_kmeans(u["Z"], k, seed)
        profiles = frame[FEATURES].assign(cluster=km["labels"]).groupby("cluster").agg(["mean", "median", "count"])
        st.dataframe(profiles, use_container_width=True)
        st.caption("Profiles use original units and observed values: counts differ when values are missing. Group IDs are arbitrary.")
        sizes = pd.Series(km["labels"]).value_counts().sort_index().rename_axis("cluster").reset_index(name="users")
        plot(px.bar(sizes, x="cluster", y="users", title="Users per segment"))
        scatter(u["pca"], km["labels"], "K-Means in PCA; fitted in standardized 3D feature space")
        st.dataframe(km["stability"], hide_index=True, use_container_width=True)
        st.metric("Mean pairwise ARI across seeds", f"{km['pairwise_ari_mean']:.3f}")
        st.caption("ARI compares partitions independently of cluster ID permutations. A value of 1 means identical partitions.")
        st.write("Manual centroid example: mean of the first 3 median-imputed observations in original units.")
        filled = SimpleImputer(strategy="median").fit_transform(frame[FEATURES])
        st.dataframe(pd.DataFrame([filled[:3].mean(axis=0)], columns=FEATURES))
        csv_download(profiles.reset_index().set_axis(
            ["cluster"] + [f"{a}_{b}" for a, b in profiles.columns], axis=1), "cluster_profiles.csv")
    with density:
        eps = st.slider("DBSCAN eps (standardized units)", .10, 2.50, .70, .05)
        ms = st.slider("DBSCAN min_samples", 3, 20, 5)
        db = density_analysis(u["Z"], eps, ms)
        st.metric("Clusters excluding noise", len(set(db["labels"]) - {-1}))
        st.metric("Noise (-1)", int((db["labels"] == -1).sum()))
        st.dataframe(db["sensitivity"], hide_index=True, use_container_width=True)
        scatter(u["pca"], db["labels"], "DBSCAN in PCA; -1 = noise")
        st.caption("Silhouette excludes noise and requires at least 2 clusters and fewer clusters than remaining points. "
                   "Otherwise it is NaN. Noise is not a verified error. Parameters apply to the original 3D space, not the 2D projection.")
        csv_download(db["sensitivity"], "dbscan_sensitivity.csv")
    with preprocessing:
        st.dataframe(u["precomparison"], hide_index=True, use_container_width=True)
        st.write("Median/mean and raw/scaled comparisons hold K and seed fixed. "
                 "StandardScaler prevents streak from dominating solely because of its measurement units. "
                 "Raw and scaled silhouettes describe different geometries; a higher value does not establish better real-world segmentation.")


def page_projections(u: dict, seed: int) -> None:
    variance = pd.DataFrame({"component": ["PC1", "PC2", "PC3"], "variance_ratio": u["variance"],
                             "cumulative": np.cumsum(u["variance"])})
    st.dataframe(variance, hide_index=True, use_container_width=True)
    st.dataframe(u["loadings"], use_container_width=True)
    st.metric("Variance retained by the first two PCs", f"{u['variance'][:2].sum():.1%}")
    labels = fit_kmeans(u["Z"], u["suggested"], seed)["labels"]
    scatter(u["pca"], labels, "PCA: linear projection")
    st.caption(f"All maps use the same K-Means labels for color, K={u['suggested']}; the target is excluded.")
    st.info("Enable t-SNE/UMAP below: the first UMAP run may take longer due to Numba compilation. "
            "Two settings are evaluated for each method. Trustworthiness(k=10) measures neighborhood preservation, not cluster quality.")
    perplexity = st.select_slider("t-SNE: selected perplexity", options=[10, 20, 30, 50], value=30)
    neighbors = st.select_slider("UMAP: selected n_neighbors", options=[5, 15, 30, 50], value=15)
    min_dist = st.slider("UMAP min_dist", 0., .9, .1, .05)
    if st.checkbox("Compute t-SNE and UMAP (including sensitivity)"):
        settings = [("t-SNE", perplexity, .1), ("t-SNE", 10 if perplexity != 10 else 30, .1),
                    ("UMAP", neighbors, min_dist), ("UMAP", 30 if neighbors != 30 else 15, min_dist)]
        rows = []
        for method, neighborhood, distance in settings:
            try:
                with st.spinner(f"{method}: neighborhood={neighborhood}..."):
                    result = nonlinear_projection(u["Z"], seed, method, neighborhood, distance)
                scatter(result["embedding"], labels, f"{method}: neighborhood={neighborhood}, min_dist={distance}")
                rows.append({"method": method, "neighborhood": neighborhood, "min_dist": distance if method == "UMAP" else None,
                             "trustworthiness_k10": result["trustworthiness"]})
            except (ImportError, ValueError, RuntimeError) as exc:
                st.error(f"{method} could not be computed: {exc}. Check requirements.txt and Python 3.11/3.12.")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("PCA reports retained variance but can miss nonlinear structure. "
                "t-SNE emphasizes neighborhoods; UMAP depends on n_neighbors/min_dist. "
                "Compare local neighborhoods and overlap; absolute map distances and island sizes are not reliable.")


def page_anomalies(frame: pd.DataFrame, u: dict, seed: int) -> None:
    contamination = st.select_slider("Isolation Forest contamination", options=[.02, .05, .10, .15], value=.05)
    eps = st.slider("DBSCAN eps for comparison", .1, 2.5, .7, .05)
    ms = st.slider("min_samples for comparison", 3, 20, 5)
    iso = anomaly_analysis(u["Z"], seed, contamination)
    db = density_analysis(u["Z"], eps, ms)
    noise = db["labels"] == -1
    flag = iso["flags"][contamination]
    st.dataframe(iso["sensitivity"], hide_index=True, use_container_width=True)
    comparisons = []
    for c, f in iso["flags"].items():
        union = int((noise | f).sum())
        comparisons.append({"contamination": c, "DBSCAN_noise": int(noise.sum()), "IF_flags": int(f.sum()),
                            "overlap": int((noise & f).sum()), "disagreement": int((noise != f).sum()),
                            "Jaccard": float((noise & f).sum() / union) if union else np.nan})
    st.dataframe(pd.DataFrame(comparisons), hide_index=True, use_container_width=True)
    categories = np.select([flag & noise, flag, noise], ["Both", "IF only", "DBSCAN only"], default="Neither")
    scatter(u["pca"], categories, "Agreement between methods in PCA")
    scores = pd.DataFrame({"decision_function": iso["scores"][contamination], "IF_flag": flag.astype(str)})
    plot(px.histogram(scores, x="decision_function", color="IF_flag", nbins=40,
                      title="IF decision_function: lower = more unusual; <0 = flag"))
    view = frame.copy()
    view["IF_flag"] = flag
    view["IF_score"] = iso["scores"][contamination]
    view["DBSCAN_noise"] = noise
    st.dataframe(view.loc[flag | noise].sort_values("IF_score"), use_container_width=True)
    st.warning("Contamination sets a threshold, not a measured true anomaly rate. "
               "Before acting, verify source logs, time zones, and the behavioral explanation. "
               "Do not automatically delete or block users.")
    csv_download(view, "anomaly_comparison.csv")


def page_predict(result: dict) -> None:
    st.write(f"Model: {result['winner']}; primary threshold: 0.5. Inputs describe this week; the prediction concerns next week.")
    with st.form("prediction"):
        values = {}
        for feature, maximum, default in [("workouts", 14, 3), ("streak", 120, 7), ("notifications", 60, 8)]:
            value = st.slider(feature, 0, maximum, default)
            missing = st.checkbox(f"{feature}: value unknown (NaN)")
            values[feature] = np.nan if missing else float(value)
        submitted = st.form_submit_button("Predict")
    if submitted:
        row = pd.DataFrame([values], columns=FEATURES)
        model = result["models"][result["winner"]]
        p = float(model.predict_proba(row)[0, 1])
        st.metric("Estimated P(active_next_week=1)", f"{p:.1%}")
        st.success("Class 1 — predicted active" if p >= .5 else "Class 0 — predicted inactive")
        st.dataframe(row, hide_index=True)
        missing_count = row.isna().sum().sum()
        if missing_count:
            st.info(f"Missing inputs: {missing_count}. Medians fitted on training data only were used. "
                    "With all inputs missing, this is only a prediction for the median profile.")
        st.caption("Sliders enforce valid ranges and numeric types. "
                   "Rare extreme inputs may be outside typical training patterns. Probability calibration and real-world validity have not been established.")


def page_export(frame: pd.DataFrame, result: dict, u: dict, seed: int, missing: float) -> None:
    st.markdown(LIMITATIONS)
    st.markdown("**Next-step materials:** download the results, then share the ZIP and projection screenshots. "
                "Reports and the README will be based on your actual run.")
    st.caption("Export uses fixed settings: K selected by silhouette, DBSCAN eps=0.7/min_samples=5, "
               "IF contamination=0.05. These settings are recorded in the manifest; other pages may use different settings.")
    if st.button("Build results ZIP"):
        km = fit_kmeans(u["Z"], u["suggested"], seed)
        db = density_analysis(u["Z"], .7, 5)
        iso = anomaly_analysis(u["Z"], seed, .05)
        summary = frame.copy()
        summary["kmeans_cluster"] = km["labels"]
        summary["dbscan_cluster"] = db["labels"]
        summary["IF_flag"] = iso["flags"][.05]
        summary["IF_score"] = iso["scores"][.05]
        summary[["PC1", "PC2"]] = u["pca"]
        profiles = frame[FEATURES].assign(cluster=km["labels"]).groupby("cluster").mean().reset_index()
        membership = frame[["user_id"]].copy()
        membership["split"] = np.where(frame.index.isin(result["Xtr"].index), "train", "test")
        tables = {"fitness_SYNTHETIC.csv": frame, "split_membership.csv": membership,
                  "supervised_cv.csv": result["cv"].reset_index(), "supervised_test.csv": result["test"].reset_index(),
                  "k_search.csv": u["search"], "k_stability.csv": km["stability"], "cluster_profiles.csv": profiles,
                  "preprocessing_comparison.csv": u["precomparison"], "dbscan_sensitivity.csv": db["sensitivity"],
                  "isolation_sensitivity.csv": iso["sensitivity"], "summary.csv": summary,
                  "pca_loadings.csv": u["loadings"].reset_index()}
        for name, p in result["probs"].items():
            tables[f"test_predictions_{name.replace(' ', '_')}.csv"] = pd.DataFrame({
                "user_id": frame.loc[result["Xte"].index, "user_id"].to_numpy(),
                "actual": result["yte"].to_numpy(), "probability_1": p, "prediction_at_0.5": (p >= .5).astype(int)})
        packages = {name: version(name) for name in ["numpy", "pandas", "scikit-learn", "streamlit", "plotly", "umap-learn"]}
        manifest = dict(synthetic=True, generator=GENERATOR_VERSION, seed=seed, n=len(frame), missing_probability=missing,
                        features=FEATURES, target=TARGET, selected_model=result["winner"], selection="training 3-fold CV F1",
                        threshold=.5, suggested_K=u["suggested"], dbscan=dict(eps=.7, min_samples=5),
                        contamination=.05, pca_variance=u["variance"].tolist(), python=platform.python_version(),
                        packages=packages, data_sha256=hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest())
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            archive.writestr("assumptions_and_limitations.md", ASSUMPTIONS + "\n" + LIMITATIONS)
            for name, table in tables.items():
                archive.writestr(name, table.to_csv(index=False))
            # Self-contained interactive HTML charts can be opened offline.
            for metric in ["inertia", "silhouette"]:
                figure = px.line(u["search"], x="K", y=metric, markers=True)
                archive.writestr(f"{metric}.html", figure.to_html(include_plotlyjs=True))
            figure = px.scatter(x=u["pca"][:, 0], y=u["pca"][:, 1], color=km["labels"].astype(str))
            archive.writestr("pca.html", figure.to_html(include_plotlyjs=True))
        st.download_button("Download ZIP with computed results", buffer.getvalue(),
                           "fitness_results_SYNTHETIC.zip", "application/zip")
    st.markdown("**Assignment sources:** the supplied Practice 8 Final Project Part 1 and Final Project P2 worksheets. "
                "[scikit-learn](https://scikit-learn.org/stable/user_guide.html), "
                "[Streamlit](https://docs.streamlit.io/), [UMAP](https://umap-learn.readthedocs.io/).")
    st.info("The worksheets require an executable Colab notebook and a short report for each part. "
            "Streamlit provides the unified demonstration interface; submission materials are prepared in the next step.")


def main() -> None:
    st.set_page_config(page_title="Fitness Retention | Variant 28", page_icon="🏋️", layout="wide")
    st.title("Fitness Retention Lab")
    st.caption("Bakhtiyar Makashev Sagandykovich · Variant 28 · AI Fundamentals")
    st.warning("Educational project: all observations and target labels are synthetic. All metrics are computed at runtime.")
    with st.sidebar:
        st.header("Experiment settings")
        n = st.slider("Users", 600, 1000, 800, 50)
        seed = int(st.number_input("Random seed", 0, 100000, 42, key="seed"))
        missing = st.slider("Missing-value probability per feature", 0., .20, .05, .01)
        page = st.radio("Navigation", ["1 · Data and EDA", "2 · Supervised", "3 · Clustering",
                                     "4 · PCA / t-SNE / UMAP", "5 · Anomaly detection", "6 · New user prediction", "7 · Export and limitations"])
        st.caption("Keep the seed, sample size, and missingness fixed for reports. Changing the seed after viewing test results is not a valid model improvement.")
    frame = generate_data(n, seed, missing)
    if page.startswith("1"):
        page_data(frame)
        return
    if page.startswith(("2", "6", "7")):
        with st.spinner("Running pipelines, 3-fold CV, and final evaluation..."):
            result = train_supervised(frame, seed)
    if page.startswith(("3", "4", "5", "7")):
        with st.spinner("Running clustering and PCA..."):
            u = unsupervised(frame, seed)
    if page.startswith("2"):
        page_supervised(result)
    elif page.startswith("3"):
        page_clusters(frame, u, seed)
    elif page.startswith("4"):
        page_projections(u, seed)
    elif page.startswith("5"):
        page_anomalies(frame, u, seed)
    elif page.startswith("6"):
        page_predict(result)
    else:
        page_export(frame, result, u, seed, missing)


if __name__ == "__main__":
    main()
