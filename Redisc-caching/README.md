# Redisc-caching — Quick start

This document shows how to run Redis for the `first-semantic_cache` examples, prepare a Conda environment, install Python dependencies, register a Jupyter kernel, and run the demo notebooks/scripts.

Paths referenced in this README assume you are in the repository root:
`/Users/itz_time_to_work/Study/GenAI-Practice`

---

## 1) Start Redis (recommended): docker-compose

From the `Redisc-caching` folder run:

```bash
cd Redisc-caching
docker-compose up -d
```

Check running containers and view logs:

```bash
docker-compose ps
docker-compose logs -f redis
```

The provided `docker-compose.yml` starts a plain Redis 8 image and maps port `6379` on the host. The notebook code defaults to `REDIS_URL=redis://localhost:6379` so this compose file will work out-of-the-box.

If you prefer a single container without docker-compose:

```bash
docker run --name redis-local -p 6379:6379 -d redis:8.0.3
```

Verify Redis is reachable:

```bash
# from host (if redis-cli installed)
redis-cli -h 127.0.0.1 -p 6379 ping

# or using docker exec
docker exec -it redis redis-cli ping
# or
docker exec -it redis-local redis-cli ping

# expected output: PONG
```

You can also verify from Python:

```bash
conda activate redis
python - <<'PY'
import os, redis
url = os.getenv('REDIS_URL', 'redis://localhost:6379')
print('Using REDIS_URL=', url)
r = redis.Redis.from_url(url)
print('PING ->', r.ping())
PY
```

---

## 2) Optional: Use Redis Stack (RedisJSON/RediSearch)

If your notebooks need Redis modules (RedisJSON, RediSearch), use the `redis/redis-stack-server` image.

Edit `docker-compose.yml` and replace the image line:

```yaml
image: redis/redis-stack-server:latest
```

Then run:

```bash
docker-compose down
docker-compose pull
docker-compose up -d
```

---

## 3) Create Conda environment and install requirements

Create the `redis` environment (Python 3.10):

```bash
conda create -n redis python=3.10 -y
conda activate redis
```

Install top-level requirements (recommended) or the `first-semantic_cache` subset:

```bash
pip install -r Redisc-caching/requirements.txt
# then for the notebook folder specifically
pip install -r Redisc-caching/first-semantic_cache/requirements.txt
```

Notes:
- Prefer `conda install -c conda-forge <package>` for large binary packages (e.g. `torch`) when available.
- If a package fails to build on macOS, try installing a conda-forge build first.

---

## 4) Register the `redis` Conda env as a Jupyter kernel

Install `ipykernel` and register:

```bash
conda activate redis
pip install ipykernel
python -m ipykernel install --user --name redis --display-name "Python (redis)"

# Verify kernel is registered
jupyter kernelspec list
```

In JupyterLab or VS Code choose the `Python (redis)` kernel when opening the notebooks.

---

## 5) Run the demo script and agent modules

Run the demo from the `first-semantic_cache` folder so `agent` is importable as a package:

```bash
conda activate redis
cd Redisc-caching/first-semantic_cache
export REDIS_URL=redis://127.0.0.1:6379
python agent/demo.py
```

If scripts expect other environment variables (API keys, endpoints), export them before running.

If a module uses `redis://` client features (Redis Stack modules), ensure you used the `redis-stack` image.

---

## 6) Run the notebooks (interactive or headless)

Interactive (recommended):

```bash
conda activate redis
cd Redisc-caching/first-semantic_cache
jupyter lab
```

Headless (execute notebooks non-interactively):

```bash
conda activate redis
cd Redisc-caching/first-semantic_cache
jupyter nbconvert --to notebook --execute agent_with_cache.ipynb --ExecutePreprocessor.timeout=600 --output executed-agent_with_cache.ipynb
jupyter nbconvert --to notebook --execute semantic_caching.ipynb --ExecutePreprocessor.timeout=600 --output executed-semantic_caching.ipynb
```

Use `papermill` if you need parameterization:

```bash
pip install papermill
papermill semantic_caching.ipynb out.ipynb -p REDIS_URL redis://127.0.0.1:6379
```

---

## Troubleshooting

- If `redis` connection fails, check `docker-compose logs redis` and confirm `PONG` from `redis-cli`.
- Ensure `REDIS_URL` in the environment matches the host/port mapping. On macOS `localhost:6379` is usually correct.
- If imports fail inside notebooks, confirm you selected the `Python (redis)` kernel.
- If a notebook requires Redis modules and you used plain `redis:8.0.3`, switch to `redis-stack` as shown above.

---

If you'd like, I can:
- Modify `docker-compose.yml` to use `redis-stack` and commit the change.
- Add a small `run_demo.sh` script to automate the steps above.
- Execute one notebook headless here to verify the end-to-end flow (will run in the `redis` env and may take time).

Created by the repository helper script — feel free to edit or expand with project-specific notes.

---

## Evaluation Metrics (Precision / Recall / F1 / WCL)

This project includes helper classes to evaluate cache effectiveness (`CacheEvaluator`) and performance (`PerfEval`) in `Redisc-caching/first-semantic_cache/cache/evals.py`.

Below are the common metrics used and how you can compute or read them from the code.

- **Cache Hit Rate**: fraction of queries that returned a cached result (hit) vs total queries.
	- Formula: cache_hit_rate = (TP + FP) / (TP + FP + FN + TN)
	- In code: `CacheEvaluator.get_metrics()` returns `cache_hit_rate`.

- **Precision**: fraction of returned cache matches that are correct (true positives among returned positives).
	- Formula: precision = TP / (TP + FP)  (defaults to 1 if denominator is 0 in the code)
	- In code: `CacheEvaluator.get_metrics()` returns `precision`.

- **Recall**: fraction of actual positives that were returned by the cache (true positives among actual positives).
	- Formula: recall = TP / (TP + FN)  (defaults to 1 if denominator is 0 in the code)
	- In code: `CacheEvaluator.get_metrics()` returns `recall`.

- **F1 Score**: harmonic mean of precision and recall, giving a single balanced metric.
	- Formula used in code: f1_score = 2 * TP / (2 * TP + FP + FN)
	- In code: `CacheEvaluator.get_metrics()` returns `f1_score`.

- **Utility**: a combined metric provided in the code that is the harmonic mean of precision and cache_hit_rate.
	- In code: `utility` = harmonic_mean(precision, cache_hit_rate)

Wording note: In this codebase `TP, FP, TN, FN` are computed using the cache results and (optionally) a ground-truth label set. The helper `CacheEvaluator.report_metrics()` will render a small panel showing these values and a confusion matrix.

WCL (Weighted Cost / Loss) — practical recipe
- The repository doesn't implement a metric literally named `WCL`, but you can compute a sensible weighted cost metric from data the repo already provides (costs from `PerfEval` and hit/miss rates from `CacheEvaluator`).

One practical WCL definition (Weighted Cost per Query):

WCL = cache_miss_rate * avg_llm_cost_per_query + cache_hit_rate * avg_cache_cost_per_query

Where:
- `cache_miss_rate = 1 - cache_hit_rate`
- `avg_llm_cost_per_query` can be obtained from `PerfEval.get_costs()` (e.g. `avg_cost_per_query` or `avg_cost_per_call`)
- `avg_cache_cost_per_query` is typically very small (cost of cache lookup) — set it to a small constant or estimate if you instrument cache costs.

Example code snippet (use inside a notebook or script after running an experiment):

```python
from cache.evals import CacheEvaluator, PerfEval

# Assume you have `true_labels` (list of booleans) and `cache_results` from your wrapper
ce = CacheEvaluator(true_labels, cache_results)
metrics = ce.get_metrics()
cache_hit_rate = metrics['cache_hit_rate']

# If you measured LLM calls with PerfEval
perf = PerfEval()
# ... run experiment and call perf.record_llm_call(...) during LLM calls ...
costs = perf.get_costs()
avg_llm_cost = costs.get('avg_cost_per_query', costs.get('avg_cost_per_call', 0.0))

# Set a small cache cost per query (estimate)
avg_cache_cost = 0.00001  # USD, example value

cache_miss_rate = 1.0 - cache_hit_rate
WCL = cache_miss_rate * avg_llm_cost + cache_hit_rate * avg_cache_cost
print(f"WCL (est.): ${WCL:.6f} per query")

print('Precision:', metrics['precision'])
print('Recall:', metrics['recall'])
print('F1:', metrics['f1_score'])
```

Notes and pointers
- File: `Redisc-caching/first-semantic_cache/cache/evals.py` — see `CacheEvaluator` and `PerfEval` implementations for exact formulas and plotting utilities.
- The notebooks in `first-semantic_cache` already use these helpers (`CacheEvaluator`, `PerfEval`, and the `LLMEvaluator` wrapper) — see `cache_effectiveness.ipynb` and `Enhancing_effectiveness.ipynb` for examples.

If you prefer a different definition of WCL (for example weighting latency vs cost), tell me the formula and I will add it and a short example calculation to this README.

