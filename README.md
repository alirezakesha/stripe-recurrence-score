# STRIP — merchant subscription / recurrence analysis

Thesis codebase for **merchant-level subscription behaviour** from Stripe-style payment panels: raw payments are cleaned and merged with merchant metadata, **pseudo-labels** are derived from payment-type mix, **hand-crafted recurrence features** and **aligned time series** feed tabular models, **MiniROCKET** (local `rocket/`), optional **SAX** bag-of-symbols features, and an optional **periodic positional Transformer**. Outputs include confusion matrices, precision–recall curves, cosine-similarity candidate lists, and business-facing conversion estimates.

---

## 1. Problem

We study **which merchants behave like subscription businesses** when we only observe aggregated payment activity (daily volumes, checkout vs. payment-link vs. subscription-tagged volume, industry, size).

Goals in this repository:

- **Binary classification**: separate merchants with a **high** share of subscription-tagged volume from those with a **low** share, using features derived from payment timing and amounts.
- **Targeting / discovery**: rank **low–subscription-ratio** merchants that look **similar** (in feature space) to known high-subscription merchants, as candidates for product or go-to-market outreach.

The core difficulty is that **subscription-ness is not a direct class label**: it must be inferred from **noisy, aggregated proxies** tied to how volume is routed through Stripe products (checkout, Payment Links, subscription billing).

---

## 2. Preprocessing

End-to-end steps (see `Stipe_final.ipynb`):

1. **Load** merchant and payment tables from `dataset/dstakehome_merchants.xlsx` and `dataset/dstakehome_payments.xlsx`.
2. **Normalize keys**: cast `merchant` identifiers to string on both sides.
3. **Date unification** (`unify_test_dates`): map heterogeneous date strings (ISO with `Z`, `YYYY-MM-DD`, `dd-Mon-yy`, etc.) to parsed datetimes; invalid values become `NaT`. The same helper is applied to payment `date` and merchant `first_charge_date`.
4. **Merge**: `payments_df_clean` left-join **merchants** on `merchant`; drop `first_charge_date` and `country` from the wide table; **remove** sentinel merchant `'0'`.
5. **Merchant-level aggregates**: compute **subscription ratio**  
   \(\text{subscription\_ratio} = \sum \text{subscription\_volume} / \sum \text{total\_volume}\)  
   and basic **coefficient-of-variation** stats on total volume; merge these back onto the panel for filtering and analysis.
6. **Feature extraction loop**: group by `merchant` and pass each group to `merchant_recurring_features` (daily resampling, gaps, autocorrelations, etc.; see below).
7. **Missing values**: numeric feature matrix is filled (e.g. `fillna(0)`) before similarity and sklearn models.

For **MiniROCKET** and the **Transformer**, per-merchant **daily total volume** is additionally materialized on a **common calendar grid** between global `min_date` and `max_date` so every merchant has the same sequence length for convolutional / rocket transforms.

---

## 3. Proxy mechanism (problem definition)

We treat this as a **novel weak-supervision scenario**:

- There is **no** curated human label for “runs a subscription program.”
- Instead, **subscription_ratio** (share of volume attributed to subscription flows) acts as a **continuous proxy** for underlying business model.
- **Pseudo-labels** for supervised models are created by **thresholding** that proxy, for example:
  - **Positive**: merchants with `subscription_ratio` above a high cutoff (e.g. **> 0.7** in training splits).
  - **Negative**: merchants with `subscription_ratio` below a low cutoff (e.g. **< 0.2**).

This is explicitly **proxy-based supervision**: the model learns patterns correlated with **payment-rail semantics**, not a definitive ontology of “subscription business.” Analysis and deployment should treat errors as **proxy noise** (mis-tagged volume, hybrid models, seasonality) rather than only classifier noise.

Cosine similarity to the **mean feature vector of high-proxy merchants** (`sim_to_sub`, `similarity_to_good`) turns the same representation into a **ranking score** for low-proxy merchants who **look like** high-proxy peers.

---

## 4. Hand-crafted features

`merchant_recurring_features` builds **one row per merchant** from the daily series of `total_volume` (zeros on inactive days after `asfreq('D')`). Examples:

| Category | Features (illustrative) |
|----------|-------------------------|
| **Activity** | `active_ratio` (share of days with positive volume) |
| **Recurrence / seasonality** | `weekly_autocorr`, `biweekly_autocorr`, `monthly_autocorr` |
| **Payment cadence** | `avg_gap_between_payments`, `gaps_std`, `gap_regularity_share_7d`, `gap_regularity_share_30d` |
| **Variability** | `cv_volume`, `cv_weekly`, `cv_monthly`, `burstiness` |
| **Calendar shape** | `weekday_entropy` |
| **Product mix (raw proxy channels)** | `subscription_ratio`, `checkout_ratio`, `payment_link_ratio` |
| **Metadata priors** | `industry_boost` (rule-based score from industry string), `volume_scale`, `total_volume` |
| **Targeting auxiliary** | `similarity_to_good` (cosine distance to centroid of very high `subscription_ratio` merchants) |

Tree models in the notebook often emphasize **monthly autocorrelation**, **industry boost**, and **monthly CV**, which matches the intuition that **stable, periodic** volume and **sector** carry signal beyond the raw proxy.

---

## 5. Periodic positional encoding, Transformer, and variable-length series

### Periodic positional encoding

Standard learned positions do not encode **billing-period structure**. The notebook’s `PeriodicPositionalEncoding` adds a **fixed** multi-period sinusoidal map over time indices, with periods aligned to **7, 14, 30, 90** days (week, biweek, month, quarter), plus a **normalized linear index**, then a **linear projection** to `d_model`. That gives the self-attention stack explicit **harmonic cues** for weekly and monthly structure.

### Transformer stack (conceptual)

- **Front end**: `MultiScaleTemporalConv` applies 1D convolutions with kernels **7 / 14 / 30** on the raw scalar series so local **multi-scale periodic** structure is available before attention.
- **Attention**: custom encoder-style layers inject **relative position bias** (bucketed by lag) so the model can learn **distance-dependent** coupling between days (e.g. month-apart peaks), with **padding masked** so padded positions do not contribute.

### How variable length is handled

| Approach | Mechanism |
|----------|-----------|
| **Hand-crafted features** | Variable-length history is **summarized** into scalars (autocorr, gaps, CV, etc.); no sequence model. |
| **MiniROCKET path** | All merchants share the **same** `min_date`–`max_date` daily vector length; no per-sample length issue at transform time. |
| **Transformer (`MerchantDataset`)** | Each series is **right-padded with zeros** to `max_len`; a **boolean padding mask** marks padded positions for the model. |
| **SAX pipeline (optional in notebook)** | **PAA + symbolization + bag-of-symbols** yields a **fixed-size** vector for any input length (with interpolation when the series is shorter than the number of PAA segments). |

---

## 6. Methods compared (tabular run, saved notebook outputs)

The numbers below are **test-set accuracy** (and **PR-AUC** where computed) from executed cells in `Stipe_final.ipynb` for **hand-crafted numeric features** only (`subscription_ratio`, `business_size`, and `industry` excluded from the design matrix for classification). Splits and class ratios follow the notebook (high-proxy vs. low-proxy merchants with controlled sampling). **Re-running the notebook will change** exact figures slightly due to random splits.

| Method | Test accuracy | Train accuracy | Notes |
|--------|----------------:|----------------:|--------|
| **Ridge classifier** | **0.792** | 0.794 | Linear baseline on engineered features. |
| **Decision tree** | **0.820** | 0.826 | Stronger nonlinear splits; importances highlight monthly autocorr. |
| **Random forest** | **0.790** | 0.813 | Test **precision–recall AUC ≈ 0.678** (positive = high-proxy class). |
| **MiniROCKET** | — | — | Fitted on **calendar-aligned** daily series; in the current main training path, rocket outputs are **commented out** of the sklearn feature matrix—enable concatenation to evaluate end-to-end. |
| **Periodic Transformer** | — | — | Implemented in-notebook (`PeriodicMerchantTransformer` + masked training); **no accuracy stream is checked into the JSON**—run the transformer cells on CPU/GPU to populate metrics. |
| **SAX + tabular (optional)** | — | — | Notebook cells build **SAX** vectors and optional **concatenation** with hand-crafted (and transformer embeddings if you export them); metrics depend on your chosen classifier and run. |

At a glance, **decision trees** achieve the **highest test accuracy** among the reported sklearn baselines on this split, while **random forest** exposes **ranking quality** via PR-AUC for the minority positive class.

---

## Repository layout

| Path | Role |
|------|------|
| `Stipe_final.ipynb` | End-to-end notebook (main entry point) |
| `analysis.ipynb`, `analysis copy.ipynb` | Additional exploratory analyses |
| `subscription_candidates.ipynb` | Candidate scoring heuristics (gap / price / industry components) |
| `rocket/` | MiniROCKET / related transforms (requires **Numba**) |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU **PyTorch**, follow the install command from [pytorch.org](https://pytorch.org) for your platform, then install the rest from `requirements.txt` if needed.

---

## Pipeline overview

```mermaid
flowchart TD
    A[Raw payments + merchants] --> B[Date cleaning + merge]
    B --> C[Merchant panel + subscription_ratio proxy]
    C --> D[Hand-crafted features + similarity]
    C --> E[Aligned daily series]
    E --> F[MiniROCKET optional]
    E --> G[Transformer + periodic PE + padding mask]
    D --> H[Tabular models RF / DT / Ridge]
    F --> H
    G --> I[Sequence classifier]
    H --> J[Metrics CM / PR]
    D --> K[Cosine targeting candidates]
```

Data paths and credentials are not committed; configure paths inside your notebooks or environment as you do locally.
