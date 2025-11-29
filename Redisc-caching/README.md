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
