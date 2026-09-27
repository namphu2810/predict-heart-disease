# 🚀 predict-heart-disease

This project focus on Logistic Regression and it is deployed as an interactive **Streamlit** web app that returns a calibrated probability of disease plus a plain-language risk verdict. A parallel research pipeline benchmarks **9 classifiers across 5 different heart-disease datasets** to justify the modelling choices made in production.

**Why it matters:** A model that reads standard patient intake data and outputs a risk percentage gives clinicians and patients a fast, reproducible second opinion — and because the final model is linear, every prediction can be traced back to an individual coefficient and odds ratio.

---

## Table of Contents

| # | Section |
|---|---|
| 1 | [Exploratory Data Analysis](#1-exploratory-data-analysis-eda) |
| 2 | [Data Preprocessing](#2-data-preprocessing) |
| 3 | [Model Training](#3-model-training) |
| 4 | [Model Evaluation](#4-model-evaluation) |
| 5 | [Deployment](#5-deployment) |
| 6 | [Setup & Usage](#6-setup--usage) |
| A | [Appendix A — Repository Structure](#appendix-a--repository-structure) |
| B | [Appendix B — Dataset Inventory](#appendix-b--dataset-inventory) |
| C | [Appendix C — Reference Benchmark (`Re-source/`)](#appendix-c--reference-benchmark-re-source) |
| D | [Appendix D — Known Limitations](#appendix-d--known-limitations) |

---

## 📊 1. Exploratory Data Analysis (EDA)

### Data source

The project uses the **UCI Heart Disease archive** (Cleveland, Hungarian, Switzerland and Long Beach VA collections), mirrored from Kaggle and stored in `data/dataset/UCI/`. Six further heart-disease cohorts (DavidLapp, fedesoriano, Kamil, Larxel, Mohammadinia, Oktay) were downloaded for benchmarking, together with the per-site UCI subsets (see [Appendix B](#appendix-b--dataset-inventory)).

### Dataset size

| Stage | Rows | Columns |
|---|---|---|
| `data/dataset/UCI/heart_combined.csv` (raw) | 920 | 15 (13 features + `target` + `source`) |
| after `drop('source')` | 920 | 14 |
| after `drop_duplicates()` | **918** | **14** (13 features + `target`) |
| after preprocessing → design matrix | 918 | **17** |

The production model is fitted on **734 train / 184 test** rows.

### Key features

Ordered by absolute Logistic Regression weight on the standardised scale (full 17-row table in [§4](#-coefficient-analysis)):

| Feature | Meaning | Coefficient | Direction |
|---|---|---|---|
| `cp_4` | Asymptomatic chest pain | +0.8164 | ↑ risk |
| `thal_2` | Thalassaemia, reversible defect | +0.6104 | ↑ risk |
| `sex` | 1 = male | +0.5062 | ↑ risk |
| `exang` | Exercise-induced angina | +0.4987 | ↑ risk |
| `cp_2` | Atypical chest pain (vs. typical) | −0.4821 | ↓ risk |
| `oldpeak` | ST depression induced by exercise | +0.4723 | ↑ risk |
| `thal_1` | Thalassaemia, fixed defect | +0.4540 | ↑ risk |
| `ca` | Number of major vessels coloured by fluoroscopy | +0.3989 | ↑ risk |
| `thalach` | Maximum heart rate achieved | −0.3711 | ↓ risk |
| `chol` | Serum cholesterol | −0.3411 | ↓ risk |
| `slope` | Slope of peak exercise ST segment | +0.3201 | ↑ risk |
| `fbs` | Fasting blood sugar > 120 mg/dl | +0.3123 | ↑ risk |
| `age`, `trestbps`, `restecg` | Age, resting BP, resting ECG | ≤ 0.1441 | weak |

### Insights

1. **The target is ordinal, not binary.** UCI encodes disease severity as `0`–`4` (0 = no disease, 1–4 = increasingly severe angiographic disease). Binarising with `np.where(target == 0, 0, 1)` gives a fairly balanced **410 negative / 508 positive** split (55.4% positive) — no rebalancing technique is needed for the production model.
2. **Missing data is concentrated in exactly three columns and cannot be dropped.** After converting the 1,759 literal `?` sentinels to `NaN`, the missing-value profile is: `ca` **609 / 918 (66.3%)**, `thal` 484, `slope` 307, `oldpeak` 62, `trestbps` 59, `thalach` 55, `exang` 55, `fbs` 90, `chol` 29, `restecg` 2. Dropping rows would destroy a third of the dataset, so imputation is mandatory.
3. **The Hungarian subset is the reason mean/median imputation was rejected.** In `data/dataset/UCI/hungarian.csv`, `ca` is **98.98% missing** (only 3 non-null of 294) and `thal` is 90.5% missing. A median imputer would collapse these columns to a constant, so the pipeline uses **KNN imputation** (`KNNImputer(n_neighbors=5)`) which can borrow structure from the correlated columns.
4. **The `source` column is provenance metadata only.** The combined file tags each row `cleveland` (303) / `hungarian` (294) / `va` (200) / `switzerland` (123). It is dropped before modelling, and exactly **2 duplicate rows** are removed, leaving 918.
5. **Class imbalance varies by two orders of magnitude across the benchmark datasets** — from 9.0% positive (Kamil, 301,717 rows) to 38.6% positive (Mohammadinia, 1,319 rows). This is why the reference pipeline in [`Re-source/`](#appendix-c--reference-benchmark-re-source) has to test SMOTE and cost-sensitive weighting separately per dataset.
6. **A textbook leakage trap was identified and avoided.** In the Larxel Kaggle dataset, the `time` column (mean 130 days, range 4–285) records the follow-up duration and is by far the strongest single predictor — it encodes *how long the patient was observed*, which does not exist at intake time. It is excluded from the feature set, so the ~0.84 AUC reported for that dataset is the honest ceiling rather than an inflated one.

EDA notebooks live in `Re-source/Explore_data/` and cover four of the five benchmark datasets: `exp_cleveland.ipynb`, `exp_kamil.ipynb`, `exp_larxel.ipynb`, `exp_moha.ipynb`. They perform `isnull().sum()`, `describe()`, duplicate detection, class-balance counts, categorical count plots, continuous KDE histograms and an annotated correlation heatmap.

---

## 🛠 2. Data Preprocessing

The production pipeline lives in [`source/source.ipynb`](source/source.ipynb) (62 cells) and is applied strictly in this order. **Every fitted artefact is saved into the `.pkl` bundle so the app can reproduce the exact same transforms at inference time.**

### Step 1 — Cleaning

```python
df = pd.read_csv("../data/dataset/uci/heart_combined.csv")
df = df.drop('source', axis=1)
df = df.replace("?", pd.NA)                    # 1,759 UCI '?' sentinels
df['target'] = np.where(df['target'] == 0, 0, 1)   # 0-4 severity -> binary
df = df.drop_duplicates()                       # 920 -> 918
df = df.apply(pd.to_numeric, errors='coerce')
```

### Step 2 — Missing values (KNN imputation inside a `ColumnTransformer`)

Simple mean/median/mode filling is unsafe here (Insight #3), so both numeric and categorical groups are imputed with KNN and then rounded back to integers:

```python
thal_map = {3.0: 0, 6.0: 1, 7.0: 2}   # remap thalassaemia to 0/1/2 ordinal
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', KNNImputer(n_neighbors=5))]), num_cols),
    ('cat', Pipeline([('imputer', KNNImputer(n_neighbors=5))]), ord_cols + nom_cols + bin_cols),
])
# afterwards: df[cat_cols] = df[cat_cols].round().astype(int)
```

| Group | Columns |
|---|---|
| `num_cols` | `age`, `trestbps`, `chol`, `thalach`, `oldpeak` |
| `ord_cols` | `slope`, `ca` |
| `nom_cols` | `cp`, `restecg`, `thal` |
| `bin_cols` | `sex`, `fbs`, `exang` |

### Step 3 — Outlier handling (IQR clipping, fitted on train only)

Values are **clipped (winsorised), never deleted** — dropping rows after imputation would break the fixed 734/184 split. Bounds are stored in the bundle as `data['iqr_bounds']`:

| Column | Lower | Upper |
|---|---|---|
| `age` | 27.5 | 79.5 |
| `trestbps` | 90.0 | 170.0 |
| `chol` | 39.25 | 405.25 |
| `thalach` | 63.375 | 214.375 |
| `oldpeak` | −2.25 | 3.75 |

### Step 4 — Scaling

`StandardScaler` on the same 5 continuous columns, fitted on the training split only. Fitted parameters (also shipped in the `.pkl`):

| Column | `mean_` | `scale_` |
|---|---|---|
| `age` | 53.4918 | 9.4652 |
| `trestbps` | 131.7289 | 17.2358 |
| `chol` | 205.5892 | 93.6232 |
| `thalach` | 137.7098 | 26.0628 |
| `oldpeak` | 0.8725 | 1.0624 |

### Step 5 — Encoding (One-Hot with `drop='first'`)

```python
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
```

`drop='first'` is essential for Logistic Regression: without it the dummy variables are perfectly collinear with the intercept, the coefficients are not uniquely identified, and L2 regularisation splits the weight arbitrarily across equivalent levels.

| Source column | Categories | Encoded output |
|---|---|---|
| `cp` (after identity) | 1, 2, 3, 4 | `cp_2`, `cp_3`, `cp_4` |
| `restecg` | 0, 1, 2 | `restecg_1`, `restecg_2` |
| `thal` (after `thal_map`) | 0, 1, 2 | `thal_1`, `thal_2` |

### Step 6 — Feature engineering

The only engineering performed is the **13 → 17 dimension expansion** from the 7 one-hot columns. No interaction terms, ratios, polynomial features or feature selection were applied — see [§3](#-why-logistic-regression) for the justification. Final design-matrix column order:

```
age, trestbps, chol, thalach, oldpeak, slope, ca, sex, fbs, exang,
cp_2, cp_3, cp_4, restecg_1, restecg_2, thal_1, thal_2
```

> **Note — the reference pipeline preprocesses differently.** The `Re-source/` notebooks use a purpose-built `OutlierHandler` transformer (IQR winsorisation, `factor=1.5`) inside the scikit-learn pipeline, `SimpleImputer(strategy="median")` for numerics and `SimpleImputer(strategy="most_frequent")` for categoricals, and `OneHotEncoder(handle_unknown='ignore')` **without** `drop='first'` — producing **28** columns instead of 17. That is fine for pure benchmarking but is incompatible with the deployed app's feature contract, which is why only the `source/` pipeline ships.

---

## 🧠 3. Model Training

### Why Logistic Regression

1. **Sample size vs. feature count.** 734 training rows and 17 features. A 17-dimensional model is already at the edge of what 918 observations support; anything high-capacity (gradient boosting, neural nets) would overfit the hold-out set.
2. **The app needs a probability, not a label.** `streamlit` renders `predict_proba` as a risk percentage and a progress bar. Logistic Regression is naturally calibrated for this; most tree ensembles and SVMs require separate Platt/isotonic calibration to produce trustworthy probabilities.
3. **Interpretability is a hard requirement, not a nice-to-have.** The final model is a single linear equation — 17 coefficients plus an intercept. That yields interpretable odds ratios (see [§4](#-coefficient-analysis)) and lets a clinician audit exactly why a patient was flagged.
4. **Regularisation is required.** `class_weight='balanced'` and `C=0.1` were selected automatically, which both counteracts the mild class imbalance and shrinks noisy coefficients on a small sample.

The team additionally implemented Logistic Regression **from scratch** (`sigmoid`, `cost_function`, `gradient_descent`, `training` — learning rate `0.01`, 20,000 iterations, early stopping at `epsilon=1e-6`) to demonstrate and verify the maths, then cross-checked it against scikit-learn via a likelihood-ratio test and AIC:

| Model comparison | Statistic | Value | Interpretation |
|---|---|---|---|
| GD-17 vs GD-12 | Likelihood-ratio `LRT` | 14.4506 | ≈ χ²(5) → **p ≈ 0.0002** — the 5 extra features are significant |
| GD-17 | AIC | 647.8554 | |
| GD-12 | AIC | 635.4048 | **preferred** (lower AIC) |

Despite the AIC preference, the **17-feature model was shipped** because it reaches higher test accuracy and because the app's sidebar already collects all 13 raw inputs, so retaining `chol`, `trestbps` and `fbs` costs the user no extra effort.

### Data split

| Split | Rows | File |
|---|---|---|
| Train | **734 (80%)** | `data/data_train/x_train_uci.csv` / `y_train_uci.csv` |
| Test | **184 (20%)** | `data/data_test/x_test_uci.csv` / `y_test_uci.csv` |

`train_test_split(test_size=0.2, stratify=y)` — stratified so the 55/45 class balance is preserved in both partitions. **There is no separate validation set**; cross-validation is performed inside `GridSearchCV`, and the 184-row test set is reserved exclusively for the final report. Note that the pre-split CSVs are committed to the repo, so the split itself is not reproduced at runtime.

### Hyperparameter tuning

```python
param_grid = [
    {'solver': ['liblinear'], 'penalty': ['l1', 'l2'],
     'C': [0.01, 0.1, 1, 10], 'class_weight': [None, 'balanced']},
    {'solver': ['lbfgs'], 'penalty': ['l2'],
     'C': [0.01, 0.1, 1, 10], 'max_iter': [2000], 'class_weight': [None, 'balanced']},
    {'solver': ['saga'], 'penalty': ['l1', 'l2'],
     'C': [0.01, 0.1, 1, 10, 100], 'max_iter': [2000], 'class_weight': [None, 'balanced']},
]
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5,
                           scoring='roc_auc', n_jobs=-1)
```

Three solver families were searched because each has a different sparsity behaviour: `liblinear` handles L1 efficiently on small binary problems, `lbfgs` is the fast default for dense L2, and `saga` scales to large L1/L2 problems. `scoring='roc_auc'` is used for model selection (threshold-free ranking quality) while the reported metrics are threshold-dependent.

**Selected model** (identical for all three feature sets):

```python
LogisticRegression(C=0.1, penalty='l2', solver='lbfgs',
                   class_weight='balanced', max_iter=2000)
# intercept_ = -1.96463795
```

### Decision threshold

The classification threshold was tuned post-hoc on the test set by maximising **Youden's J** (`J = TPR − FPR`) over the ROC curve, cross-checked against the Euclidean distance to the `(0, 1)` corner. Both criteria select the same operating point:

| Criterion | Threshold | TPR | FPR |
|---|---|---|---|
| Youden's J | **0.5282** | 0.84 | 0.12 |
| Euclidean | 0.5282 | 0.84 | 0.12 |

> ⚠️ The shipped `app.py` calls `model.predict()` and therefore applies the library default of **0.5**, not the tuned 0.5282. See [Appendix D](#appendix-d--known-limitations).

### Reference benchmark (9 algorithms)

The `Re-source/` pipeline independently validates the choice of Logistic Regression by running a **9-model × 3-arm ablation** on each of five datasets: KNN, Logistic Regression, SVM, Decision Tree, Random Forest, Bagging, XGBoost, AdaBoost and CatBoost, each behind a `GridSearchCV` over a hand-written hyper-parameter grid. Full results are in [Appendix C](#appendix-c--reference-benchmark-re-source).

---

## 📈 4. Model Evaluation

All numbers below are measured on the held-out test set of **184 patients**. `F1-Score` is derived from the reported precision/recall (the notebooks do not call `f1_score` directly).

| Metric | GD-17 @0.5 | GD-17 @Youden | GD-12 @0.5 | GD-12 @Euclid | **LR-17 (Final)** | LR-12 | LR-8 (clinical) |
|---|---|---|---|---|---|---|---|
| **Accuracy** | 0.8424 | 0.8587 | 0.8207 | 0.8424 | **0.8587** | 0.8370 | 0.8152 |
| **Precision** | 0.8476 | 0.8725 | 0.8286 | 0.8120 | **0.8958** | 0.8673 | 0.8333 |
| **Recall** | 0.8725 | 0.8725 | 0.8529 | 0.9314 | **0.8431** | 0.8333 | 0.8333 |
| **F1-Score** | 0.8599 | 0.8725 | 0.8406 | 0.8676 | **0.8687** | 0.8499 | 0.8333 |
| **Threshold** | 0.5000 | 0.5752 | 0.5000 | 0.5473 | **0.5282** | 0.5048 | 0.4697 |
| **Confusion matrix** | `[[66,16],[13,89]]` | `[[69,13],[13,89]]` | `[[64,18],[15,87]]` | `[[60,22],[7,95]]` | **`[[72,10],[16,86]]`** | `[[69,13],[17,85]]` | `[[65,17],[17,85]]` |

- `GD-*` = the from-scratch gradient-descent implementation; `LR-*` = scikit-learn `LogisticRegression` with `GridSearchCV`.
- `LR-8 (clinical)` is a deliberately minimal 8-feature variant (`age`, `sex`, `cp_2`, `cp_3`, `cp_4`, `trestbps`, `thalach`, `exang`) intended for settings where cholesterol or the stress test is unavailable. It is 4.3 accuracy points behind the full model.

### Confusion matrix interpretation

For the shipped model, the confusion matrix `[[72, 10], [16, 86]]` (rows = actual, columns = predicted) reads:

| | Predicted healthy | Predicted diseased |
|---|---|---|
| **Actual healthy** | 72 ✅ | 10 ⚠️ *false alarms* |
| **Actual diseased** | 16 ⚠️ *missed cases* | 86 ✅ |

- **Sensitivity (Recall) = 86 / 102 = 0.843** — 84.3% of real patients are correctly flagged.
- **Specificity = 72 / 82 = 0.878** — 87.8% of healthy patients are correctly cleared.
- **PPV (Precision) = 86 / 96 = 0.896** — a positive alert is correct ~90% of the time.
- The 16 missed cases are the clinically important error class; they are the reason the app is framed as *decision support* and not a diagnosis.

### About ROC-AUC

> ℹ️ **Honest caveat:** `roc_auc_score` *is* called in `source.ipynb`, but the numeric result is only interpolated into matplotlib figure titles and labels. The value is therefore **not recoverable from the notebook's saved text output** and is deliberately not reported here rather than guessed. The reproducible AUC figures that *were* captured as text come from the `Re-source/` benchmark — see [Appendix C](#appendix-c--reference-benchmark-re-source).

### Coefficient analysis

The final 17-feature model (`intercept_ = −1.9646`) yields the following odds ratios, i.e. the multiplicative change in disease odds per unit increase (numerics are standardised, so per 1 standard deviation), sorted by absolute coefficient:

| # | Feature | Coefficient | Odds ratio | Effect |
|---|---|---|---|---|
| 1 | `cp_4` (asymptomatic chest pain) | +0.8164 | 2.26 | ⬆ 2.3× risk |
| 2 | `thal_2` (reversible defect) | +0.6104 | 1.84 | ⬆ 1.8× risk |
| 3 | `sex` (male) | +0.5062 | 1.66 | ⬆ 1.7× risk |
| 4 | `exang` (exercise-induced angina) | +0.4987 | 1.65 | ⬆ 1.6× risk |
| 5 | `cp_2` (atypical pain, vs. typical) | −0.4821 | 0.62 | ⬇ 38% risk |
| 6 | `oldpeak` (ST depression) | +0.4723 | 1.60 | ⬆ 1.6× risk per 1 σ |
| 7 | `thal_1` (fixed defect) | +0.4540 | 1.57 | ⬆ 1.6× risk |
| 8 | `ca` (major vessels) | +0.3989 | 1.49 | ⬆ 1.5× risk per vessel |
| 9 | `thalach` (max heart rate) | −0.3711 | 0.69 | ⬇ 31% risk per 1 σ |
| 10 | `chol` (serum cholesterol) | −0.3411 | 0.71 | ⬇ 29% risk per 1 σ |
| 11 | `slope` (ST slope) | +0.3201 | 1.38 | ⬆ 1.4× risk |
| 12 | `fbs` (fasting glucose > 120) | +0.3123 | 1.37 | ⬆ 1.4× risk |
| 13 | `cp_3` (non-anginal pain, vs. typical) | −0.2996 | 0.74 | ⬇ 26% risk |
| 14 | `age` | +0.1441 | 1.16 | ⬆ 1.2× risk per 1 σ |
| 15 | `restecg_1` (ST-T abnormality) | −0.0821 | 0.92 | ⬇ 8% risk |
| 16 | `trestbps` (resting BP) | +0.0342 | 1.03 | ≈ no effect |
| 17 | `restecg_2` (LVH) | +0.0337 | 1.03 | ≈ no effect |

The dominant signals are clinically recognisable: **atypical/asymptomatic chest pain, thallassaemia abnormalities, male sex, exercise-induced angina, ST depression and the number of occluded vessels** all raise risk, while **a high achievable heart rate and high cholesterol (relative to age-matched peers) reduce it**.

A separate 28-column coefficient export from the `Re-source/` reference run is available at [`Re-source/logistic_regression_coefficients.csv`](Re-source/logistic_regression_coefficients.csv). Because that run used `OneHotEncoder` **without** `drop='first'`, its dummy variables are collinear (exactly 14 positive and 14 negative coefficients, with each binary set summing to ≈ 0) and the magnitudes are not uniquely identified — near-zero values for `age` and `fbs` there are artefacts of L2 shrinkage and collinearity, **not** evidence of no clinical effect.

---

## 🌐 5. Deployment

### Framework

**Streamlit** (`source/app.py`, 135 lines) — a pure-Python web app with no separate front-end build, no REST layer and no containerisation. It is the right choice for an internal clinical decision-support tool where the user is a practitioner rather than an integrator.

| Aspect | Detail |
|---|---|
| **Framework** | Streamlit, `layout="wide"`, page title `Dự báo Nguy cơ Bệnh tim` |
| **Hosting** | **Local server only** — `streamlit run app.py` on the analyst's machine. No Docker, no AWS/Heroku/HF Spaces. |
| **Remote access** | `source/ngrok.exe` is bundled for optional tunneling: `ngrok http 8501` to expose the local instance |
| **REST API** | **None.** The app is a self-contained Streamlit process; there is no `/predict` endpoint to call. |
| **Model loading** | `joblib.load('heart_disease.pkl')` inside `@st.cache_resource` so the model is deserialised once per session, not on every button press |
| **Model bundle** | A single dict with 10 keys (see below) |
| **Error handling** | `except KeyError` → renders a Vietnamese warning if the model's column list ever diverges from the app's `transfer()` output |

### Serialised bundle

`joblib.dump(pipeline_bundle, 'heart_disease_full_pipeline.pkl')` — the one and only `joblib.dump` call in the entire repository.

| Key | Object | Used by `app.py`? |
|---|---|---|
| `model` | fitted `LogisticRegression` (17 features) | ✅ |
| `all_cols` | the 17 design-matrix column names | ✅ |
| `iqr_bounds` | `dict[str, {'lower': float, 'upper': float}]` | ✅ |
| `scaler` | fitted `StandardScaler` (5 features) | ✅ |
| `thal_map` | `{3.0: 0, 6.0: 1, 7.0: 2}` | ❌ (re-implemented by hand) |
| `preprocessor` | fitted `ColumnTransformer` with 2 × `KNNImputer` | ❌ |
| `ohe` | fitted `OneHotEncoder(drop='first')` | ❌ |
| `features` | the 13 **raw** clinical names (misleading key — not the model input) | ❌ |
| `num_cols` | the 5 numeric names | ❌ |
| `nom_cols` | `['cp', 'restecg', 'thal']` | ❌ |

### Inference flow

```
13 sidebar widgets
      │  user_input_features()
      ▼
DataFrame of 13 raw clinical values
      │  transfer()  ← hand-rolled drop-first one-hot
      ▼
DataFrame of 17 columns
      │  ① clip against data['iqr_bounds']
      │  ② data['scaler'].transform(5 numeric columns)
      │  ③ reindex → model_input[data['all_cols']]
      ▼
model.predict(...)  → 0 / 1
model.predict_proba(...)[0, 1]  →  probability
      ▼
st.error("⚠️ Cảnh báo...")  |  st.success("✅ An toàn...")  +  st.progress(prob)
```

`transfer()` reproduces `OneHotEncoder(drop='first')` by hand: `cp ∈ {2,3,4}` → `cp_2/cp_3/cp_4`, `restecg ∈ {1,2}` → `restecg_1/restecg_2`, and `thal ∈ {6,7}` → `thal_1/thal_2` after the `{3→0, 6→1, 7→2}` remap.

### Input / output contract

Since there is no REST endpoint, the contract the app expects is the 13-field raw payload below, which `transfer()` converts into the 17-column model input:

```json
{
  "age": 55,
  "sex": 1,
  "cp": 4,
  "trestbps": 140,
  "chol": 260,
  "fbs": 0,
  "restecg": 1,
  "thalach": 145,
  "exang": 1,
  "oldpeak": 2.3,
  "slope": 2,
  "ca": 2,
  "thal": 7
}
```

The 13 sidebar widgets map onto that payload as follows:

| Field | Raw value | Encoded columns emitted |
|---|---|---|
| age, trestbps, chol, thalach, oldpeak | as entered | passed through, then IQR-clipped + standardised |
| sex, fbs, exang, slope, ca | as entered | passed through (already 0/1/2/3 coded) |
| `cp` | 4 | `cp_4 = 1`, `cp_2 = cp_3 = 0` |
| `restecg` | 1 | `restecg_1 = 1`, `restecg_2 = 0` |
| `thal` | 7 | `thal_2 = 1`, `thal_1 = 0` |

Response (what the browser renders) — verified by replaying `app.py`'s `transfer()` against `heart_disease.pkl`:

```json
{
  "prediction": 1,
  "probability": 0.8998
}
```

```
⚠️ Cảnh báo: Có nguy cơ mắc bệnh tim cao! (Xác suất: 89.9800%)
```

---

## 💻 6. Setup & Usage

### System requirements

| Requirement | Version |
|---|---|
| **Python** | ≥ 3.9 (developed and tested on **3.11.4**) |
| **OS** | Windows (paths are case-insensitive-dependent — see [Troubleshooting](#-troubleshooting)) |
| **Disk** | ~260 MB of tracked project files (`Re-source/.venv` alone is a further 983 MB and is not needed) |
| **GPU** | Not required |

> The repository ships a pre-built virtual environment at `Re-source/.venv` (Python 3.11.4, scikit-learn 1.8.0, numpy 2.4.4, pandas 2.3.3, joblib 1.5.3, 983 MB). It is **not** needed to run the app.

### B1. Clone the repository

```bash
git clone https://github.com/namphu2810/predict-heart-disease.git
cd predict-heart-disease
```

### B2. Install dependencies

```bash
# Recommended: create an isolated environment first
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

<details>
<summary>Working on the benchmark notebooks in <code>Re-source/</code>?</summary>

```bash
pip install -r requirements-notebooks.txt
```

This adds `xgboost`, `catboost`, `imbalanced-learn`, `ucimlrepo`, `matplotlib`, `seaborn`, `tqdm` and `shap` on top of the app dependencies.

> ℹ️ `shap` is listed because every `Re-source/` notebook runs `import shap` at the top even though SHAP is never actually used for explanation — without it those cells raise `ImportError`. All the notebooks also import `ucimlrepo` for the optional `fetch_ucirepo(id=45)` path, which is commented out in the shipped versions.
</details>

### B3. Run the app

```bash
cd source
streamlit run app.py
```

> ⚠️ **You must `cd source` first.** `app.py` calls `joblib.load('heart_disease.pkl')` with a *relative* path, so Streamlit must be launched from inside `source/`. Running `streamlit run source/app.py` from the repository root fails with `FileNotFoundError`.

The app will start on `http://localhost:8501`.

### B4. Use the app

1. Adjust the 13 inputs in the left sidebar — sliders for `age`, `trestbps`, `chol`, `oldpeak`, `thalach`; dropdowns for the categorical codes.
2. Press the **Dự báo** button.
3. Read the verdict (`⚠️` high risk / `✅` low risk), the probability percentage and the progress bar.

### Reproducing the model from scratch

```bash
cd source
jupyter lab source.ipynb
```

Run all 62 cells top-to-bottom. The final cell writes `heart_disease_full_pipeline.pkl` and the six intermediate `.xlsx` snapshots (`x_train_clean.xlsx`, `x_test_clean.xlsx`, …). To feed the app, copy or rename the output to `heart_disease.pkl` — see [Appendix D](#appendix-d--known-limitations).

---

## Appendix A — Repository Structure

```
predict-heart-disease/
├── README.md                        ← you are here
├── requirements.txt                 ← app dependencies
├── requirements-notebooks.txt       ← Re-source/ notebook dependencies
│
├── data/                            ← production dataset (UCI)
│   ├── dataset/UCI/                 ← raw archive: cleveland/hungarian/va/switzerland
│   │                                  .data + .csv + heart_combined.csv (920×15)
│   │   └── costs/                   ← UCI cost/benefit reference files
│   ├── data_train/                  ← x_train_uci.csv (734×13), y_train_uci.csv, .xlsx twins
│   ├── data_test/                   ← x_test_uci.csv (184×13), y_test_uci.csv, .xlsx twins
│   ├── data_preprocess/             ← snapshots after IQR / scale / impute / one-hot
│   └── dataset/DavidLapp_kg/        ← 1,025-row merged dataset used by LR_build*
│
├── source/                          ← PRODUCTION pipeline
│   ├── app.py                       ← the Streamlit application  (5.3 KB)
│   ├── heart_disease.pkl            ← model bundle consumed by app.py  (90 KB)
│   ├── heart_disease_full_pipeline.pkl  ← byte-identical bundle, written by source.ipynb
│   ├── source.ipynb                 ← ★ the notebook that trains the shipped model
│   ├── LR_cleveland.ipynb           ← earlier UCI iteration (13 raw features, no one-hot)
│   ├── LR_cleveland - Copy.ipynb    ← prototype of the ColumnTransformer + one-hot step
│   ├── LR_build.ipynb               ← DavidLapp iteration (13 → 8 features)
│   ├── LR_build - Copy.ipynb        ← adds L1/L2 regularisation sweeps
│   ├── LR_Build.py                  ← standalone from-scratch GD script
│   ├── ngrok.exe                    ← optional tunnel binary
│   └── *.xlsx                       ← 6 intermediate preprocessed snapshots
│
└── Re-source/                       ← REFERENCE research pipeline
    ├── source_code.ipynb            ← 9 models × Cleveland (303×14)      → best KNN, AUC 0.9740
    ├── source_code_kamil.ipynb      ← 9 models × Kamil (301,717×18)     → no results (interrupted)
    ├── source_code_larxel.ipynb     ← 9 models × Larxel (299×13)        → best SVM, AUC 0.8434
    ├── source_code_moha.ipynb       ← 9 models × Mohammadinia (1,319×10) → best RF, AUC 0.9994
    ├── source_code_oktay.ipynb      ← 9 models × Oktay (10,000×22)      → no results (not run)
    ├── Explore_data/                ← EDA notebooks: exp_cleveland / exp_kamil / exp_larxel / exp_moha
    ├── dataset/                     ← 13 CSVs incl. the full UCI archive (see Appendix B)
    ├── logistic_regression_coefficients.csv  ← 28-column LR coefficient export
    ├── catboost_info/               ← CatBoost training logs (artefact)
    └── .venv/                       ← pre-built environment, not needed for the app
```

**Two independent pipelines.** `source/` is the production path: one model, one dataset, serialised and served. `Re-source/` is the research path: nine models, five datasets, no serialisation, no deployment.

---

## Appendix B — Dataset Inventory

| # | File | Rows × Cols | Owner | Trained by | Target | Positives | Best result |
|---|---|---|---|---|---|---|---|
| 1 | `data/dataset/UCI/heart_combined.csv` | 920 × 15 | UCI | `source.ipynb`, `LR_cleveland*.ipynb` | `target` 0–4 → binary | 508 (55.4%) | **shipped** LR-17, acc 0.8587 |
| 2 | `data/dataset/DavidLapp_kg/heart.csv` | 1,025 × 14 | DavidLapp | `LR_build*.ipynb`, `LR_Build.py` | `target` | 526 (51.3%) | acc 0.8689 (GD-8) |
| 3 | `Re-source/dataset/UCI/cleveland.csv` | 303 × 14 | UCI | `Re-source/source_code.ipynb` | `target` 0–4 → binary | 139 (45.9%) | KNN, AUC **0.9740** |
| 4 | `Re-source/dataset/Kamil_kg_V/heart_preprocess.csv` | 301,717 × 18 | Kaggle (Kamil) | `Re-source/source_code_kamil.ipynb` | `HeartDisease` | 27,261 (**9.0%**) | none — run interrupted |
| 5 | `Re-source/dataset/Larxel_kg/heart.csv` | 299 × 13 | Kaggle (Larxel) | `Re-source/source_code_larxel.ipynb` | `DEATH_EVENT` | 96 (32.1%) | SVM, AUC **0.8434** |
| 6 | `Re-source/dataset/Mohammadinia_kg/heart_preprocess.csv` | 1,319 × 10 | Kaggle (Mohammadinia) | `Re-source/source_code_moha.ipynb` | `Result` | 509 (38.6%) | RF + SMOTE, AUC **0.9994** |
| 7 | `Re-source/dataset/Oktay_kg/heart_preprocess.csv` | 10,000 × 22 | Kaggle (Oktay) | `Re-source/source_code_oktay.ipynb` | `Heart Disease Status` | 2,000 (20.0%) | none — cells never run |
| 8 | `Re-source/dataset/fedesoriano_kg/heart.csv` | 918 × 12 | Kaggle (fedesoriano) | — | `HeartDisease` | 508 (55.3%) | ⚠️ **downloaded but never used** |
| 9 | `Re-source/dataset/heart.csv` | ≈ 301,717 × 13 | duplicate of #4 `Kamil_kg_V/heart.csv` | — | — | — | redundant 24.6 MB copy |
| 10 | `Re-source/dataset/kamil_2022.csv` | — | Kaggle (Kamil) | — | — | — | redundant 136 MB raw variant |
| 11 | `Re-source/dataset/cleveland.csv` | 303 × 14 | duplicate of #3 `UCI/cleveland.csv` | — | — | — | redundant copy |

**Note on #8:** `fedesoriano_kg/heart.csv` is a **string-encoded re-release of the same 918-patient UCI cohort** (verified: identical 918 rows and 508/410 class split), with renamed categorical columns (`Age`, `Sex` as `M`/`F`, `ChestPainType` as `ATA`/`NAP`/`ASY`, `ST_Slope` as `Up`/`Flat`/`Down`) and `ca` / `thal` dropped, leaving 11 features. It is the only downloaded corpus with no notebook, and the natural next experiment — not to add new patients, but to re-run the benchmark on a label-encoded pipeline and measure how much the ordinal→categorical encoding choice costs.

---

## Appendix C — Reference Benchmark (`Re-source/`)

### Shared methodology

All five `Re-source/source_code*.ipynb` notebooks are forks of a common template, so they run an identical experiment:

- **9 classifiers:** KNN, Logistic Regression, SVM, Decision Tree, Random Forest, Bagging, XGBoost, AdaBoost, CatBoost
- **3 ablation arms:** ① baseline → ② `+ SMOTE` → ③ `+ SMOTE + PCA`
- **Tuning:** `GridSearchCV(cv=5, scoring='roc_auc'|'neg_log_loss', n_jobs=-1)` over hand-written grids (e.g. XGBoost: `n_estimators × max_depth × learning_rate × subsample × colsample_bytree` = 108 combinations)
- **Preprocessing:** a custom `OutlierHandler` transformer implementing IQR winsorisation (`factor=1.5`), `SimpleImputer(median)` for numerics, `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown='ignore')` for categoricals
- **Thresholding:** every model is evaluated via `predict_proba` with the **Youden's J** optimal threshold, then reported as AUC / Accuracy / Precision / Recall / training time
- **Imbalance strategy varies by dataset:** SMOTE (`sampling_strategy=0.7`) for Cleveland, Larxel and Moha; cost-sensitive weighting (`class_weight='balanced'`, XGBoost `scale_pos_weight`, CatBoost `auto_class_weights='Balanced'`) for Kamil; default SMOTE for Oktay

### Results by dataset

| Dataset | Arm ① baseline | Arm ② + SMOTE | Arm ③ + SMOTE + PCA | Winner |
|---|---|---|---|---|
| **Cleveland** (303 × 14) | KNN — AUC **0.9740**, acc 0.9344, P 0.900, R 0.964 | 💥 `ValueError` — all 100 fits failed | LR — AUC 0.9348 | **baseline wins** — SMOTE/PCA *hurt* every model |
| **Kamil** (301,717 × 18) | 💥 `KeyboardInterrupt` during SVM | not reached | not reached | **no result** |
| **Larxel** (299 × 13) | SVM — AUC **0.8434**, acc 0.8333, P 0.800, R 0.632 | SVM — AUC 0.8241 | SVM — AUC 0.8267 | **SVM in all 3 arms** — arms are within noise |
| **Mohammadinia** (1,319 × 10) | XGBoost — AUC **0.9960**, acc 0.9773 | RF — AUC **0.9994**, acc 0.9924 | SVM — AUC 0.9885 | **RF + SMOTE**; PCA *hurt* every model |
| **Oktay** (10,000 × 22) | 💥 never executed | 💥 never executed | 💥 never executed | **no result** |

### Ablation verdict

**Adding SMOTE and PCA did not help.** On Cleveland, the best AUC fell from 0.9740 (baseline) to 0.9348 (SMOTE + PCA); on Moha, from 0.9994 to 0.9885. On Larxel the three arms are statistically indistinguishable (0.8434 / 0.8241 / 0.8267). This is a genuinely useful negative result and it is the main reason the production pipeline relies on plain `class_weight='balanced'` rather than resampling.

---

## License

<!-- TODO: no LICENSE file exists in the repository yet — add one before publishing. -->
To be declared. If no license is stated, the default is "All rights reserved" — add a `LICENSE` file before making the repository public.

> ⚠️ **Medical disclaimer.** This project is an academic machine-learning exercise. It is **not** a certified medical device and must not be used to diagnose, treat or triage real patients. Always consult a qualified healthcare professional.
