# How it works (and how to explain it in a viva)

## The problem in ML terms
Supervised **regression on a time series**: for every (date, meal) predict a count ≥ 0. Rows are days × meals
(3 meals × ~500 days ≈ 1,500 rows) — small data, so *tree ensembles + good features* beat deep learning here.

## Pipeline, step by step
1. **Features** (`features/build.py`). Calendar (weekday, month…), timetable load, weather, event flags, and history
   (`lag_1`, `lag_7`, rolling means, same-weekday 4-week average). *Every history feature uses only earlier days.*
2. **Hold-out.** Last 60 days never touched during training/tuning → honest score.
3. **Time-aware CV.** Expanding windows: train Jan–Mar / validate Apr · train Jan–Apr / validate May · … Random K-fold would
   leak the future into training and flatter the score.
4. **Tuning.** `RandomizedSearchCV` samples hyper-parameters (depth, learning rate, trees…) and keeps the best by MAE.
   XGBoost/LightGBM may also pick a **Poisson objective** — footfall is a count.
5. **Ensemble.** Weighted average, weights ∝ 1 / CV-error. Averaging different model families reduces variance.
6. **Quantile models.** Instead of predicting the mean, LightGBM predicts the 10th/80th/90th percentile directly
   (pinball loss). A **calibration factor** learned from out-of-fold data makes the nominal level match reality.
7. **Recursive multi-day forecasts.** Predict tomorrow → treat it as "yesterday" → predict the day after…
8. **Planning layer.** `portions = cook-for × portions_per_head`; `ingredient = portions × recipe qty × (1 + buffer)`.

## Why not just linear regression?
Footfall interacts: rain matters more on snack time, exams hit day-scholars not hostelers, a holiday changes how much last
week's number means. Trees learn such interactions automatically; linear models need them hand-crafted. The Ridge
reference in the report lets you *show* the gap instead of claiming it.

## Metrics
* **MAE** — "on average we're off by N students" · **RMSE** — punishes big misses · **WAPE** — total error as % of total demand
* **Within ±10 %** — share of meals predicted close enough to be useful
* **Interval coverage** — does the "80 % range" really contain 80 % of outcomes?
* **Business replay** — food wasted % / meals that ran out % under each cooking strategy.

## Likely viva questions
* *Why time-based splits?* Because the model will be used on the future; random splits leak it.
* *What is data leakage and where could it creep in here?* Using today's footfall to build today's features; the unit test
  `test_no_leakage_in_history_features` guards against it.
* *Why quantile regression?* The cost of too little food ≠ cost of too much; a single average hides that. Choose the quantile to match the cost.
* *How do you handle students who eat outside?* Implicitly — we model the observed canteen count, which already reflects them.
* *Is the demo data real?* No. It's a simulator; real data swaps in without code changes. (Say this first — examiners respect it.)
* *What would you do with more time?* Dish-level data, POS integration, drift monitoring, SHAP, deployment.
