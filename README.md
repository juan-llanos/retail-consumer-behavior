# 🛒 Customer Behavior Analytics: Segmentation, Retention, and Campaign Effectiveness in Retail

*A data science case study spanning customer segmentation, retention, and marketing measurement — from a raw, unverified relational dataset to a validated PostgreSQL database, an evidence-based customer segmentation, and a campaign-effectiveness analysis that catches its own selection-bias trap.*

## Overview

This project asks four related questions about retail customer behavior: which households are currently most valuable and how often do they buy (RFM), does clustering find segments beyond simple RFM quartiles (clustering), how consistently households stay active over time (cohorts), and do marketing campaigns actually change spending — or would spending look the same without them (campaign effectiveness).

The core of the project runs entirely on one dataset ( dunnhumby's *The Complete Journey* ) loaded into a properly normalized PostgreSQL database and queried with SQL wherever SQL is the natural tool, with pandas and scikit-learn taking over for scoring, scaling, and modeling. Two additional questions the core dataset can't answer cleanly — funnel conversion (no session-level events) and true A/B testing (campaign targeting isn't randomized) — are answered instead by two small, self-contained microcases on their own purpose-built datasets, rather than forcing dunnhumby to simulate data it doesn't have.

Every methodological choice — which outlier method, which statistical test, which correction — is made and justified against what the specific data and question require, not applied by default. Where a comparison design would have been misleading on its own (Notebook 05), the project adds designs that cover its weakness rather than reporting the first result at face value.

### Why Three Datasets

| Question | Needs | dunnhumby have it? | Dataset used |
|---|---|---|---|
| RFM, clustering, cohorts, campaign SQL | Household-level transactions, multi-table relational structure | Yes | dunnhumby — *The Complete Journey* |
| Funnel conversion (view → cart → purchase) | Session-level event logs | No — transaction-line grain only, no browsing events | eCommerce Events History in Cosmetics Shop |
| A/B testing on a genuinely randomized experiment | Randomized treatment/control assignment | No — campaign targeting is retailer-chosen, not randomized | Kevin Hillstrom — MineThatData E-Mail Analytics |

Notebook 05 shows why this matters: targeted households spend about twice as much as non-targeted ones during campaign windows, but the gap was already there before their campaigns started, so a plain targeted-vs-untargeted comparison mostly reflects who gets targeted. A dataset with *real* randomized assignment (Hillstrom) is needed to show what a clean A/B test looks like by contrast.

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

Notebooks `01`–`05` share a single PostgreSQL database (`dunnhumby`), built in `01` and extended with two intermediate tables: `rfm_household` (built in `02`, loaded by `03` from the database rather than assuming `02` ran in the same session) and `rfm_segmented` (the final segmentation saved by `03`; `04` and `05` work from `transaction_data` directly and don't read it). `06` and `07` are fully independent microcases — no PostgreSQL, no shared tables — each reading its own raw CSV directly.

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
| 02 | **RFM** | Builds a household-level RFM table (2,500 households) directly in SQL; scores Recency/Frequency/Monetary by quintile, validates strict monotonicity across all three scores, and applies a rule-based segmentation (Champions, At Risk, Lost…) as a descriptive baseline — while surfacing its own limitation (quintile scores rank each household against everyone else, not against its own buying rhythm) as one concrete reason a clustering notebook follows. | SQL CTEs, `pd.qcut` quantile scoring, IQR outlier detection |
| 03 | **Clustering** | Tests whether clustering finds segments beyond RFM quartiles, and how clearly separated they are. K-means on scaled RFM features, k chosen via elbow + silhouette (LOOCV deliberately not used — it doesn't apply without a target variable). Runs the full pipeline twice, with and without outliers, to diagnose K-means' sensitivity to extreme spenders, then combines both results into a final 4-tier segmentation that preserves outliers as an explicit "Elite/Outlier" tier instead of letting them distort the other clusters. | `StandardScaler`, K-means, elbow method, silhouette score, outlier-sensitivity comparison |
| 04 | **Retention Cohorts** | Builds cohorts on 28-day periods rather than calendar months (dunnhumby has no real dates) and a matrix/heatmap of the share of each cohort active per period. Because dunnhumby is a fixed two-year household panel, this measures how steadily panel households keep buying, not the retailer's customer churn. Identifies an apparent late-period drop as a window-edge artifact: the last period has only 11 days. | Cohort analysis, retention-matrix construction, heatmap diagnostics |
| 05 | **Campaigns & Coupons** | Tests whether coupon campaigns change spending with a within-household pre/during comparison, a between-household targeted/non-targeted comparison, and a difference-in-differences that combines both, since each of the first two is vulnerable to a different confound. Ranks all 30 campaigns via a SQL window function, then tests every campaign with Holm correction and 95% confidence intervals. | Paired & Welch's t-tests, difference-in-differences, Holm correction, SQL window functions (`RANK() OVER`) |
| 06 | **Funnel Microcase** | Independent microcase on the Cosmetics Shop event log. Measures the view → cart → purchase funnel at session level (per-user conversion is not measured), then explicitly checks whether ratios between stage totals equal true session-level conditional rates — they don't, since not every purchase session has a recorded cart event — and reports the corrected conditional rates instead. | Session-level funnel construction, conditional-rate validation against raw stage ratios |
| 07 | **A/B Testing Microcase** | Independent microcase on the Hillstrom dataset. Tests two email campaigns against a control group with a two-proportion z-test (conversion) and Welch's t-test (spend), then applies Bonferroni correction jointly across all 4 resulting tests — chosen over Benjamini-Hochberg because a family of only 4 tests doesn't benefit from BH's power advantage. Checks randomization balance on pre-treatment covariates and reports 95% confidence intervals for every difference. | Two-proportion z-test, Welch's t-test, Bonferroni correction, confidence intervals |

## Key Methods Used

- **SQL-first architecture**: PostgreSQL is the primary engine for extraction and feature construction (RFM, cohorts, campaign pre/post windows), with pandas and scikit-learn as the secondary layer for scoring, scaling, and modeling — not the other way around.
- **Unsupervised segmentation, evaluated on its own terms**: K-means validated via elbow method and silhouette score; LOOCV is not used, since there is no target variable to validate against.
- **Outlier-sensitivity diagnostics**: the clustering pipeline is run twice, with and without outliers, specifically to tell a stable segment (found both ways) from one that's an artifact of extreme values (found only one way).
- **Several comparison designs for observational data**: within-household, between-household and difference-in-differences comparisons in Notebook 05, because campaign targeting isn't randomized and each design alone is exposed to a different confound.
- **Multiple-comparison correction where several tests answer one question**: Holm across the 30 per-campaign tests in Notebook 05, and Bonferroni for the 4 tests in Notebook 07, justified there against Benjamini-Hochberg.
- **Window functions**: per-campaign effect ranking via `RANK() OVER` directly in SQL, avoiding a pandas round-trip for something the database already does natively.

## Conclusions

Across the seven notebooks, four findings tie the project together:

1. **Customer value is mostly a gradient, with one clearly separate group.** K-means finds a stable, low-engagement "Inactive" segment that holds regardless of how outliers are handled (named to avoid colliding with Notebook 02's differently-defined, much larger rule-based "Lost" segment), and a much less stable "High Value" tier that a handful of extreme spenders can distort substantially. The final segmentation resolves this by keeping those outliers as an explicit "Elite/Outlier" category rather than letting them warp the boundary for everyone near them. Beyond the Inactive group, the segments are best read as data-driven cut points along a value gradient: `frequency` and `monetary` are strongly correlated (r = 0.70) and silhouette scores are moderate (0.50–0.58).

2. **Panel households keep buying at a steady rate, with no decay.** Between 74.6% and 84.7% of each reliable cohort is active in every period from 1 to 20 (Notebook 04, Section 3), with a slight upward drift (+0.08 points per period). The one visible dip, in each cohort's final column, is a window-edge artifact: the 26th period has only 11 days. Since dunnhumby is a fixed two-year panel, this describes the behavior of panel households, not the retailer's customer churn.

3. **Campaigns don't show a clear effect on spending, and the naive comparison would have said the opposite.** Targeted households spend about twice as much as non-targeted ones during campaign windows, but they already spent $6.14/day more before their campaigns started (significant in 29 of 30 campaigns), so that gap reflects who gets targeted. Within households, spend barely changes from the pre-window to the campaign window (+$0.01/day, p = 0.84). Tested one by one with Holm correction, only campaign 18 (1,133 households) shows a significant pre/during change (+$0.62/day), and most of it was shared by non-targeted households: its difference-in-differences (+$0.27/day) doesn't survive the correction, and neither does any other campaign's.

4. **Randomized email campaigns do lift conversion and spend.** In the Hillstrom experiment (Notebook 07), the Mens email raises conversion from 0.57% to 1.25% (+0.68 points, 95% CI +0.50 to +0.86) and the Womens email to 0.88% (+0.31 points); both lifts in conversion and spend survive Bonferroni correction, and the spend lift comes mainly from more customers buying, not from larger baskets.

**Scope and limitations.** The core segmentation is based on transactional behavior only — demographic data covers just 32% of purchasing households and isn't used to build it. The campaign-effectiveness findings describe this retailer's specific, non-randomized targeting; they don't generalize to a properly randomized campaign, which is precisely why the A/B testing microcase uses a different dataset rather than trying to force a randomized-style conclusion out of dunnhumby's observational data. The funnel and A/B testing microcases are deliberately self-contained: this project treats them as two related but methodologically distinct questions answered on their own appropriate data, not as one dataset stretched to cover everything.

## Tech Stack

Python · pandas · numpy · scikit-learn · scipy.stats · statsmodels · SQLAlchemy · psycopg2 · PostgreSQL · matplotlib

## Running the Pipeline

**Requirements:** Python 3.12, [Anaconda](https://www.anaconda.com/) or Miniconda, and PostgreSQL.

Notebooks `01`–`05` are numbered and designed to be run in order against a shared PostgreSQL database; `06` and `07` are independent and can be run in any order, without `01`–`05` having been run first.

**Prerequisite:** a `dunnhumby` database already created in PostgreSQL (via pgAdmin) — Notebook 01 drops and recreates its own 8 tables, so any other tables already present in that database are left untouched — and a `.env` file in the project root with the connection credentials (excluded from git via `.gitignore`; copy `.env.example` and fill in your own values).

**Data:** the raw datasets are not included in the repository — download them and place them under `data/raw/` as described in [`data/README.md`](data/README.md).

```bash
git clone https://github.com/juan-llanos/retail-consumer-behavior.git
cd retail-consumer-behavior
conda create -n retail_env python=3.12 -y
conda activate retail_env
pip install -r requirements.txt
cp .env.example .env   # Windows: copy .env.example .env — then edit .env with your own PostgreSQL credentials
jupyter notebook notebooks/01_data_setup_and_loading.ipynb
```

## Author

**Juan Felipe Llanos Pérez** — data analysis.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
