# 🛒 Customer Behavior Analytics: Segmentation, Retention, and Campaign Effectiveness in Retail

*A data science case study spanning customer segmentation, retention, and marketing measurement — from a raw, unverified relational dataset to a validated PostgreSQL database, an evidence-based customer segmentation, and a campaign-effectiveness analysis that catches its own selection-bias trap.*

## Overview

This project asks four related questions about retail customer behavior: which households are currently most valuable and how often do they buy (RFM), do natural customer segments exist beyond simple RFM quartiles (clustering), how well does the business retain households over time (cohorts), and do marketing campaigns actually change spending — or would spending look the same without them (campaign effectiveness).

The core of the project runs entirely on one dataset ( dunnhumby's *The Complete Journey* ) loaded into a properly normalized PostgreSQL database and queried with SQL wherever SQL is the natural tool, with pandas and scikit-learn taking over for scoring, scaling, and modeling. Two additional questions the core dataset can't answer cleanly — funnel conversion (no session-level events) and true A/B testing (campaign targeting isn't randomized) — are answered instead by two small, self-contained microcases on their own purpose-built datasets, rather than forcing dunnhumby to simulate data it doesn't have.

Every methodological choice — which outlier method, which statistical test, which correction, whether to reuse a technique from an earlier project — is made and justified against what the specific data and question require, not applied by default. Where a comparison design would have been misleading on its own (Notebook 05), the project runs a second design specifically to catch that trap, rather than reporting the first result at face value.

### Why Three Datasets

| Question | Needs | dunnhumby have it? | Dataset used |
|---|---|---|---|
| RFM, clustering, cohorts, campaign SQL | Household-level transactions, multi-table relational structure | Yes | dunnhumby — *The Complete Journey* |
| Funnel conversion (view → cart → purchase) | Session-level event logs | No — transaction-line grain only, no browsing events | eCommerce Events History in Cosmetics Shop |
| A/B testing on a genuinely randomized experiment | Randomized treatment/control assignment | No — campaign targeting is retailer-chosen, not randomized (confirmed in Notebook 05) | Kevin Hillstrom — MineThatData E-Mail Analytics |

This last point is a finding in its own right, not just a scoping note: Notebook 05 shows that dunnhumby's own campaign targeting is confounded by selection (targeted households were already high spenders before their campaigns started), which is exactly why a dataset with *real* randomized assignment (Hillstrom) is needed to demonstrate what a clean A/B test looks like by contrast.

## Repository Structure

```
├── data/
│   ├── README.md                                         # Where to download each dataset
│   └── raw/
│       ├── dunnhumby/                                    # 8 CSVs — The Complete Journey
│       ├── eCommerce Events History in Cosmetics Shop/   # 2019-Nov.csv only
│       └── Kevin Hillstrom MineThatData E-MailAnalytics/ # hillstrom.csv only
├── notebooks/
│   ├── 01_data_setup_and_loading.ipynb
│   ├── 02_rfm.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_retention_cohorts.ipynb
│   ├── 05_campaigns_and_coupons.ipynb
│   ├── 06_funnel_microcase.ipynb
│   └── 07_ab_testing_microcase.ipynb
├── outputs/
│   └── figures/                                          # PNGs saved by notebooks 03, 04, 05, and 06
├── .env                     # PostgreSQL credentials (excluded from git)
├── .env.example             # Template for .env — safe to commit
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Notebooks `01`–`05` share a single PostgreSQL database (`dunnhumby`), built in `01` and extended with two intermediate tables (`rfm_household` in `02`, `rfm_segmented` in `03`) that later notebooks load independently rather than assuming an earlier notebook already ran in the same session. `06` and `07` are fully independent microcases — no PostgreSQL, no shared tables — each reading its own raw CSV directly.

## The Data

| Dataset | Granularity | Scale | Key columns | Used in |
|---|---|---|---|---|
| dunnhumby — *The Complete Journey* | 8 relational tables, transaction-line grain | 2,500 households · 2.6M transactions · 36.8M `causal_data` rows | `household_key`, `basket_id`, `product_id`, `sales_value`, `day`, `campaign`, `coupon_upc` | 01–05 |
| eCommerce Events History in Cosmetics Shop | One row per event (`view`/`cart`/`remove_from_cart`/`purchase`) | 942,022 sessions · 4,635,837 events (Nov 2019 only) | `event_time`, `event_type`, `user_session`, `product_id` | 06 |
| Kevin Hillstrom — MineThatData E-Mail Analytics | One row per customer | 64,000 customers, 3 randomized groups (~21,300 each) | `segment`, `conversion`, `spend` | 07 |

dunnhumby is a third-party public dataset, not one generated for this project, so Notebook 01 treats it accordingly: nothing about its structure, cleanliness, or coverage is assumed before it's checked.

**Two data-quality issues drive Notebook 01's methodology**, found only because the full files were checked rather than a sample:
- **`product.csv`** — 33.1% of `CURR_SIZE_OF_PRODUCT` values looked blank in a row-level check but were neither `NaN` nor empty strings; they were whitespace-only strings, normalized to real `NULL` before loading.
- **`coupon.csv`** — 4.1% exact duplicate rows, distinguished from `coupon_redempt.csv`'s superficially similar duplicates (which turned out to be genuine repeated business events, not noise) by re-checking against every column before deciding to deduplicate.

**Additional limitations are documented rather than hidden**: only 32% of purchasing households have a matching `hh_demographic` record (so demographic segmentation is out of scope for the core analysis); `causal_data` (promotional context) covers weeks 9–101, while `transaction_data` covers weeks 1–102, leaving weeks 1–8 *and* week 102 without display/mailer data — 9 weeks total, not just the first ~8; and `causal_data` has 15,245 rows (0.041%) that repeat on (`product_id`, `store_id`, `week_no`), differing only in `display`/`mailer` — preserved rather than deduplicated, since their business meaning (separate promotional events vs. conflicting records) isn't established.

## The Pipeline

| # | Notebook | What it does | Key techniques |
|---|---|---|---|
| 01 | **Data Setup & Loading** | Audits all 8 raw dunnhumby CSVs (dtypes, nulls, duplicates, anomalous values) before assuming a schema; corrects the whitespace-as-null and exact-duplicate issues above; designs and loads an 8-table PostgreSQL schema with primary/foreign keys grounded in what the audit actually found. | Data auditing, PostgreSQL DDL design, chunked & `COPY` bulk loading, referential-integrity verification |
| 02 | **RFM** | Builds a household-level RFM table (2,500 households) directly in SQL; scores Recency/Frequency/Monetary by quintile, validates strict monotonicity across all three scores, and applies a rule-based segmentation (Champions, At Risk, Lost…) as a descriptive baseline — while surfacing its own limitation (a combined-condition segment whose raw averages don't match its name) as the concrete reason a clustering notebook follows. | SQL CTEs, `pd.qcut` quantile scoring, IQR outlier detection |
| 03 | **Clustering** | Answers the question `02` left open: do natural segments exist beyond RFM quartiles? K-means on scaled RFM features, k chosen via elbow + silhouette (LOOCV deliberately not used — it doesn't apply without a target variable). Runs the full pipeline twice, with and without outliers, to diagnose K-means' sensitivity to extreme spenders, then combines both results into a final 4-tier segmentation that preserves outliers as an explicit "Elite/Outlier" tier instead of letting them distort the other clusters. | `StandardScaler`, K-means, elbow method, silhouette score, outlier-sensitivity comparison |
| 04 | **Retention Cohorts** | Builds cohorts on 28-day periods rather than calendar months (dunnhumby has no real dates) and a retention matrix/heatmap; identifies an apparent late-period retention drop as a window-edge artifact — households near the end of the observation period simply haven't had time to repurchase — rather than a real decline. | Cohort analysis, retention-matrix construction, heatmap diagnostics |
| 05 | **Campaigns & Coupons** | Tests whether coupon campaigns change spending using two complementary comparison designs — within-household pre/post and between-household treatment/control — specifically because either design alone is vulnerable to a different confound. Cross-referencing the two reveals the treatment/control result was driven by selection bias, not a campaign effect. Ranks all 30 campaigns individually via a SQL window function and tests the single best- and worst-performing campaign on their own: neither reaches significance, reinforcing the portfolio-wide null under the most favorable individual scrutiny available. | Paired & independent (Welch's) t-tests, SQL window functions (`RANK() OVER`), selection-bias diagnosis |
| 06 | **Funnel Microcase** | Independent microcase on the Cosmetics Shop event log. Measures the view → cart → purchase funnel at session level, then explicitly checks whether ratios between stage totals equal true session-level conditional rates — they don't, since not every purchase session has a recorded cart event — and reports the corrected conditional rates instead. | Session-level funnel construction, conditional-rate validation against raw stage ratios |
| 07 | **A/B Testing Microcase** | Independent microcase on the Hillstrom dataset. Tests two email campaigns against a control group with a two-proportion z-test (conversion) and Welch's t-test (spend), then applies Bonferroni correction jointly across all 4 resulting tests — chosen over Benjamini-Hochberg specifically because a family of only 4 tests doesn't benefit from BH's power advantage. | Two-proportion z-test, Welch's t-test, Bonferroni correction |

## Key Methods Used

- **SQL-first architecture**: PostgreSQL is the primary engine for extraction and feature construction (RFM, cohorts, campaign pre/post windows), with pandas and scikit-learn as the secondary layer for scoring, scaling, and modeling — not the other way around.
- **Unsupervised segmentation, evaluated on its own terms**: K-means validated via elbow method and silhouette score; LOOCV — appropriate for a prior supervised project — is explicitly rejected here as methodologically unjustified without a target variable, rather than reused by default.
- **Outlier-sensitivity diagnostics**: the clustering pipeline is run twice, with and without outliers, specifically to tell a stable segment (found both ways) from one that's an artifact of extreme values (found only one way).
- **Two-design causal validation**: within-household and between-household comparisons run side by side in Notebook 05, specifically to expose a selection-bias trap that either design alone would have missed.
- **Multiple-comparison correction applied only where genuinely warranted**: Bonferroni is used once, for the 4 real simultaneous tests in Notebook 07, and justified there against the alternative (Benjamini-Hochberg) — not applied as a blanket default elsewhere in the project.
- **Window functions**: per-campaign effect ranking via `RANK() OVER` directly in SQL, avoiding a pandas round-trip for something the database already does natively.

## Conclusions

Across the seven notebooks, three threads tie the project together:

1. **Customer value is real but not uniform, and simple RFM quartiles undersell that.** K-means finds a stable, low-engagement "Inactive" segment that holds regardless of how outliers are handled (named to avoid colliding with Notebook 02's differently-defined, much larger rule-based "Lost" segment), and a much less stable "High Value" tier that a handful of extreme spenders can distort substantially. The final segmentation resolves this by keeping those extreme spenders as an explicit "Elite/Outlier" category rather than letting them warp the boundary for everyone near them.

2. **Retention is strong and stable, not decaying — but the raw numbers alone would have said otherwise.** 74.6–84.7% retention holds across periods 1–20 of the 25-period window (Notebook 04, Section 2.1); the one visible dip, at each cohort's final columns, is a window-edge artifact rather than a genuine late-stage drop, identified only because the matrix was inspected for *why* it dipped, not just *that* it dipped.

3. **Campaigns, on average, don't move spend — and the more naive comparison would have said the opposite.** The between-household design alone suggested a large campaign effect; cross-referencing with the within-household design shows that gap is pre-existing selection, not a causal lift. Ranking all 30 campaigns individually and testing the single best- and worst-performing one on its own confirms the null holds even under the most favorable individual scrutiny: neither reaches significance.

**Scope and limitations.** The core segmentation is based on transactional behavior only — demographic data covers just 32% of purchasing households and isn't used to build it. The campaign-effectiveness findings describe this retailer's specific, non-randomized targeting; they don't generalize to a properly randomized campaign, which is precisely why the A/B testing microcase uses a different dataset rather than trying to force a randomized-style conclusion out of dunnhumby's observational data. The funnel and A/B testing microcases are deliberately self-contained: this project treats them as two related but methodologically distinct questions answered on their own appropriate data, not as one dataset stretched to cover everything.

## Tech Stack

Python · pandas · numpy · scikit-learn · scipy.stats · statsmodels · SQLAlchemy · psycopg2 · PostgreSQL · matplotlib

## Running the Pipeline

Notebooks `01`–`05` are numbered and designed to be run in order against a shared PostgreSQL database; `06` and `07` are independent and can be run in any order, without `01`–`05` having been run first.

**Prerequisite:** a `dunnhumby` database already created in PostgreSQL (via pgAdmin) — Notebook 01 drops and recreates its own 8 tables, so any other tables already present in that database are left untouched — and a `.env` file in the project root with the connection credentials (excluded from git via `.gitignore`; copy `.env.example` and fill in your own values).

```bash
git clone https://github.com/juan-llanos/retail-consumer-behavior.git
cd retail-consumer-behavior
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your own PostgreSQL credentials
jupyter notebook notebooks/01_data_setup_and_loading.ipynb
```

## Author

**Juan Felipe Llanos Pérez** — data analysis.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
