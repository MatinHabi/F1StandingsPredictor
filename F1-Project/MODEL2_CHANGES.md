# model vs model2, and changes to model2

## How it was measured

- **Metric:** mean Kendall's tau per race between predicted and actual finishing order. This is what `run_races` prints. Huber loss on lap times is reported as a secondary metric.
- **Splits:** train on 2018–2023, eval on 2024, test on 2025 (unchanged).
- **Seeds:** every number below is the mean of 5 training seeds (0, 1, 2, 3, 42). On a single seed, tau moves by about ±0.02–0.04, so one run can't separate the models.
- **Selection:** changes were chosen by their 2024 eval score. That makes the eval gain slightly optimistic, so the 2025 test number is the fair one.
- **Baseline to beat:** starting grid order. It scores tau 0.642 (2024) and 0.591 (2025).

## Which one is better: main.py or main2.py

| | 2024 eval tau | 2025 test tau | 2024 eval loss | 2025 test loss |
|---|---|---|---|---|
| `main.py` / model (autoregressive) | 0.273 ± 0.025 | **0.256** ± 0.026 | **5.22** | **4.14** |
| `main2.py` / model2 (no lag, team strategies) | **0.394** ± 0.019 | 0.245 ± 0.023 | 5.73 | 5.21 |

**model2 is the better standings predictor.** It's clearly ahead on 2024 and level with model on 2025, within noise.

model is better at single lap times, because it gets the previous lap, track temperature and compound as inputs. In the simulation, though, the race order mostly comes from the previous lap it predicted itself. The driver and car embeddings carry little, so the order it produces is weak.

The seed-42 run on its own (model 0.231, model2 0.375 on eval) made the gap look bigger than it is.

## Changes to model2.py

All changes are in `model2.py`. `main2.py` is unchanged. Each step below is cumulative.

| Step | 2024 eval tau | 2025 test tau | eval loss | test loss |
|---|---|---|---|---|
| model2 as it was | 0.394 | 0.245 | 5.73 | 5.21 |
| 1. lap range fix | 0.394 | 0.242 | 5.73 | 5.21 |
| 2. + team/track aliases for the embeddings | 0.460 | 0.216 | 5.21 | 4.44 |
| 3. + train the default embedding (`NEW_P`) | 0.454 | 0.214 | 5.15 | 4.29 |
| 4. + recency weighting (`YEAR_DECAY = 0.5`) | **0.475** ± 0.013 | **0.274** ± 0.017 | 5.29 | 4.56 |

### 1. Simulate laps 1..TotalLaps (bug fix)

- **Change:** `simulate_race` used `range(1 + total_laps + 1)`, which is `range(0, total_laps + 2)`. It is now `range(1, total_laps + 1)`.
- **Why:** the old loop simulated a lap 0 and a lap after the flag with negative `FuelLoad`, and both were added to every driver's race time.
- **Effect:** no measurable change, because every driver got the same two extra laps. It's a correctness fix.

### 2. Use `TEAM_ALIASES` / `TRACK_ALIASES` for the embeddings, not only the strategies

- **Change:** the two alias dicts moved above the data load, and `Car`/`Track` are renamed once on the full dataframe.
- **Why:** in 2024, `RB` and `Kick Sauber` never appear in the 2018–2023 training data under those names. Before this change they fell back to the "unknown team" embedding. 2025 adds `Racing Bulls` and the `Miami Gardens` track key. Aliasing gives them the history of Toro Rosso/AlphaTauri, Sauber/Alfa Romeo, and Miami.
- **Effect:**
  - 2024 eval tau +0.066.
  - Taking it out of the final version drops eval from 0.475 to 0.351 and leaves test unchanged.
  - Test tau dipped slightly at this step alone. Those two teams' 2025 form differs from their 2018–2023 average, which step 4 deals with.
  - Lap-time loss also improves (eval 5.73 → 5.21, test 5.21 → 4.44).

### 3. Actually use `NEW_P` in `training()`

- **Change:** `NEW_P = 0.05` was declared but never used. The training loop now swaps in the default embedding (index 0) for 5% of rows, as `model.py` already does.
- **Why:** without it, the default embedding is never trained and stays random. Every rookie gets that random vector: 3 in 2024, 6 in 2025 (ANT, BEA, BOR, COL, DOO, HAD).
- **Effect:** neutral on its own. In the final version, taking it out drops test tau from 0.274 to 0.236, so it matters once recency weighting is on.

### 4. Recency-weighted training loss

- **Change:** `W_train = YEAR_DECAY ** (2023 - Year)`. The training loss is now a weighted Huber loss, so 2023 counts 1, 2022 counts 0.5, 2021 counts 0.25, and so on.
- **Why:** a car embedding is one vector per team for all six training years. Without weighting it reflects 2018–2023 on average, so Mercedes-era pace dominates. Weighting recent years makes it closer to current form, which is what 2024/2025 look like.
- **Effect:** test tau 0.214 → 0.274 at this step, and eval 0.454 → 0.475. Lap-time loss rises a little (eval 5.15 → 5.29, test 4.29 → 4.56), since older years count less, but it's still well below where model2 started.
- **Tuning:** `YEAR_DECAY = 0.7` gave about the same score (0.473 / 0.271). 0.5 was kept because it varied less across seeds.

**Note:** the `training_loss` that `training()` prints is now the year-weighted loss, with the `NEW_P` swaps included. It's higher than before (about 3.46 vs 3.29 at the last epoch), but the two numbers aren't comparable.

### Overall improvement (mean of 5 seeds)

| | before | after | change |
|---|---|---|---|
| 2024 eval tau | 0.394 | 0.475 | **+0.081 (+21%)** |
| 2025 test tau | 0.245 | 0.274 | **+0.029 (+12%)** |
| 2024 eval loss | 5.73 | 5.29 | −0.44 |
| 2025 test loss | 5.21 | 4.56 | −0.65 |

- **Eval:** the gain is consistent. Every seed after the change (0.457–0.488) scores higher than every seed before it (0.374–0.423).
- **Test:** the gain is smaller. 4 of 5 seeds improve, and 12 of 24 races have a higher seed-averaged tau.
- **Single run:** `python main2.py` (seed 42) now gives eval tau 0.488, up from 0.375, and test tau 0.292, up from 0.240.

## Step 5: qualifying gap as a feature

- **Change:** a new input, `QualiGap`. It's each driver's best qualifying lap (fastest of Q1/Q2/Q3) as a % gap to the fastest qualifying lap of that session, capped at 7% (F1's 107% rule).
- **Why:** the models only knew pace from 2018–2023 history. Qualifying is the best pre-race measure of this weekend's pace. It happens before the race, so nothing leaks from the result.

### Code (`model2.py`)

- **`get_quali_gaps()`:**
  - Loads each race's qualifying session from fastf1 and works out the gaps.
  - Saves them to `quali_gaps.csv` in `F1-Project/` (git-ignored like the other CSVs), so later runs don't download again.
  - On the fastf1 rate limit (`500 calls/h`) it sleeps an hour and retries, the same way `build_all_data.py` does.
- **Merged into `d`** on Year/Race/Driver. A driver with no qualifying time gets the slowest gap in that session. Then the 7% cap is applied.
- **Normalised** with the training mean/std, added to `to_tensor`, and passed per driver in `simulate_race`. If a whole session has no times, the gap is filled with 0.0 (the average gap).

### Data

- **Download:** 172 qualifying sessions, about 870 API calls. That hits fastf1's limit once, so the first download takes about 1h10m.
- **Missing session:** 2025 Miami has the qualifying order but no lap times in fastf1's source, so every driver there gets the average gap.
- **Missing drivers:** 42 other drivers have no time in their session.
- **Fresh clone:** the first run of `main2.py` will do this download itself.
- **Building the CSV here:** reloading cached sessions still makes 2–6 API calls each, from live-timing lookups that fail and aren't cached. So `quali_gaps.csv` was built once with `fastf1.Cache.offline_mode(True)`, using the same `get_quali_gaps` code, with zero API calls.

### Results (mean of 5 seeds)

| | 2024 eval tau | 2025 test tau | races beating grid (eval / test) | eval loss | test loss |
|---|---|---|---|---|---|
| before (steps 1–4) | 0.475 ± 0.013 | 0.274 ± 0.017 | 3.0 / 0.4 of 24 | 5.29 | 4.56 |
| + qualifying gap, no cap | 0.545 ± 0.025 | 0.355 ± 0.037 | 6.2 / 1.2 | 5.32 | 4.55 |
| **+ qualifying gap, capped at 7%** | **0.554** ± 0.013 | **0.359** ± 0.017 | **6.8 / 0.8** | 5.32 | 4.55 |

- **Gain:** +0.079 tau on 2024 and +0.085 on 2025. Since the original model2, 2024 is up from 0.394 to 0.554 and 2025 from 0.245 to 0.359.
- **Cap:** the 7% cap gave about the same mean as no cap, with half the spread across seeds.
- **Lap-time loss:** unchanged. The feature helps the order, not single laps.
- **Single run:** `python main2.py` (seed 42) gives eval 0.577 with 8/24 races beating the grid, and test 0.368 with 2/24.

### The model is diluting the qualifying signal

Ranking drivers by `QualiGap` alone, with no model, scores **0.675** on 2024, which beats the grid (0.642). On 2025 it scores 0.573, against the grid's 0.586 on the same 23 races.

The model with the feature scores 0.554 / 0.359, which is lower. Its car/driver embeddings (2018–2023 form) and the team strategy's stop-count penalty are still pulling the order away from what qualifying says.

## Races beating the grid baseline (new output line)

`run_races` now prints `<name> races beating the grid baseline: X/24` after the mean taus, the same way `model.py` does. It counts a race as a win only when that race's model tau is higher than the grid tau for the same race. A tie doesn't count.

## Tried and not kept

- **Dropping slow laps from training** (over 107% of the driver's median lap in that race, which removes about 10% of laps: lap 1, in/out laps, safety car, red flags). It lowered lap-time loss but cut test tau from 0.271 to 0.251.
- **30 epochs instead of 20.** No gain (0.474 / 0.263).

## Still open

- model2 is still below the grid baseline on average (0.554 vs 0.642 in 2024, 0.359 vs 0.591 in 2025), even though qualifying order alone beats or matches the grid. The next thing to try is making the model rely on `QualiGap` more than on the old embeddings. For example, raise `NEW_P` so the embeddings are dropped more often, or shrink the car embedding.
- Test trains on data that ends in 2023 and predicts 2025, a year further out than eval. Retraining on 2018–2024 before scoring 2025 would be the realistic setup, and would probably help most.
- Recency weighting (step 4) would probably help `model.py` too. It wasn't applied there.
