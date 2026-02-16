# Mobile.bg Car Price Prediction 

## Overview

This project is a complete pipeline for automated data collection, processing and supervised machine-learning modeling of car listings from mobile.bg.
It scrapes listings, preprocesses and engineers features, and trains regression models to predict market price. The main objective is to maximize explained variance (R²) and minimize mean absolute error (MAE).

---

## Key components (modular architecture)

* **Scraper** — automated web scraper that downloads HTML pages, parses listings and saves raw data as CSV. Configurable parameters include brand filters, page depth, price range, etc. Image URLs are saved when available.
* **Preprocessing** — data cleaning and normalization: fixes/removes missing or malformed values, expands boolean feature lists into binary columns, strips HTML artifacts and merges multiple scraping sessions into a single dataset ready for modeling.
* **Feature engineering** — creates aggregated and derived features (brand/model statistics, ratio metrics, keyword counts from free-text descriptions).
* **Modeling modules** — independent training pipelines for:

  * Linear models (linear regression, polynomial expansions, spline transforms)
  * Tree-based models and boosting (DecisionTree, RandomForest, XGBoost)
  * Neural networks (PyTorch MLPs with optional embeddings for high-cardinality categorical variables)

---

## Data schema

**Numeric columns**

* Year, Month
* Horsepower
* CubicCapacity
* KmDriven
* WLTP_Range_km
* Battery_Capacity_kWh

**Categorical columns**

* Status, City, Area
* Brand, Model
* EngineType, EuroStandard
* TransmissionType, Category, Color, Condition

**Boolean columns**

* ~100 binary columns representing presence/absence of extras (GPS tracker, spoiler, tow hook, comprehensive insurance "Casco", on-board computer, etc.)

**Text**

* `Description` — free text from the ad (used for lightweight sentiment/keyword features).

---

## Feature engineering highlights

* Aggregated statistics computed per brand/model on the train set:

  * `MeanPriceByModel`, `StdPriceByModel`, `VariancePriceByModel`
  * `MeanPriceByBrand`, `StdPriceByBrand`, `VariancePriceByBrand`
* Class mapping by brand/model (luxury / mid / budget) based on project lookup tables.
* Derived numeric ratios: `KmPerYear`, `Horsepower_per_age`, etc.
* Simple keyword counting from `Description` using positive/negative word lists (adds modest predictive signal).

---

## Modeling notes & important preprocessing tricks

* **Target transformation:** models are trained to predict `log(price)`; final predictions are obtained by exponentiating model outputs. This stabilizes variance and reduces skew caused by the heavy-tailed price distribution.
* **Categorical handling:** for neural networks an embedding layer for `Brand`/`Model` was implemented and concatenated with numeric features. For tree models the engineered aggregate statistics were particularly effective.
* **Boolean expansion:** many extras are supplied as nested lists in raw scrape results — these were expanded to separate binary columns.

---

## Libraries & tools used

* Python (codebase)
* `requests` — HTTP requests & page retrieval
* `pandas`, `numpy` — data processing and numeric operations
* `scikit-learn` — preprocessing, splines, PCA, baseline models and hyperparameter search
* `xgboost` — gradient boosting models (best performing)
* `torch` / `PyTorch` — neural network implementation

**Related work / references:** datasets and experiments on platforms such as Kaggle were reviewed during the domain study, but this project focuses on automatically scraped data from mobile.bg.

> *Note: each external entity above is referenced once (see headings).*

---

## Selected results (test set)

### Linear models (R² / Adj R²)

| Model                                         |         R² | Adj R² |
| --------------------------------------------- | ---------: | -----: |
| Baseline Linear Regression                    |     0.6606 | 0.6587 |
| Polynomial Regression with PCA                |     0.6070 | 0.6056 |
| Spline Regression                             |     0.7366 | 0.7260 |
| Spline Regression (numeric columns)           |     0.7368 | 0.7351 |
| Polynomial Regression with log transform      |     0.7844 | 0.7832 |
| Polynomial + log transform + model info       |     0.8073 | 0.8058 |
| Spline (numeric) + log transform + model info | **0.8322** | 0.8309 |

### Tree-based models (test R², test MAE)

| Model                                    |    test_r2 |    test_mae |
| ---------------------------------------- | ---------: | ----------: |
| Decision Tree Regressor                  |     0.8328 |     7023.57 |
| Decision Tree (optimized w/ model stats) |     0.8651 |     6172.12 |
| Random Forest (optimized w/ model stats) |     0.9012 |     4899.24 |
| XGBoost (w/ model stats)                 |     0.9186 |     4850.84 |
| XGBoost (all brand/model features)       |     0.9441 |     4693.78 |
| XGBoost (final, keywords included)       | **0.9531** | **4428.79** |

### Neural networks (test R², test MAE)

| Model                                        | test_r2 | test_mae |
| -------------------------------------------- | ------: | -------: |
| MLP (baseline)                               |  0.8925 |  4923.14 |
| MLP + log transform                          |  0.9233 |  4802.13 |
| MLP + log + batchnorm + dropout              |  0.9402 |  4723.24 |
| MLP + log + regularization + brand embedding |  0.9405 |  4712.84 |

**Observations**

* XGBoost with full feature engineering achieved the best result: **R² = 0.9531**, MAE ≈ **4428.79**.
* Linear models show limited capacity for capturing complex nonlinear relationships; substantial gains come from engineered brand/model aggregates and log-transforming the target.
* Neural nets performed strongly but embeddings provided only marginal improvement over well-engineered tabular features.

---

## Experiments & implementation details

* Scraping uses `requests` + regular expressions to parse HTML; results are saved as CSV files under a `raw/` directory.
* Preprocessing merges multiple scraping sessions, expands booleans, imputes/corrects malformed values and applies feature transforms.
* Hyperparameter searches (grid / randomized) were run for linear, tree and neural models. For tree models a systematic optimization of model-level and brand/model aggregate features produced the largest performance gains.
* Target is modeled as `log(price)` for most experiments; exponentiation is used at inference.

---

## Conclusions

* The combination of automated scraping from mobile.bg, careful feature engineering (especially brand/model aggregates) and modern tree-based boosting yields a highly accurate price predictor (XGBoost, R² ≈ 0.95).
* Feature engineering—particularly brand/model aggregated statistics—provides the most significant uplift.
* Neural network approaches are competitive but did not surpass the best boosted trees in this project.

---

## Future work

* **Feature selection / dimensionality reduction:** reduce from ~270 columns to a compact set that preserves predictive performance.
* **Image data:** integrate car images (downloaded by scraper) using convolutional neural networks to capture visual cues affecting price.
* **Advanced text modeling:** replace keyword counts with transformer-based text features or fine-tuned language models to extract richer signals from `Description`.
* **Deployment:** package the final model behind a lightweight API and add continuous data collection to monitor model drift.

