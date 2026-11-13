# Hardware tests: what one training window costs, and the fastest way to train many

**Machine.** NVIDIA RTX 5080 (16 GB, of which the desktop and other programs hold 2–3 GB), Intel i7-12700KF (12 cores, 20 threads), 32 GB of RAM, Windows 11, PyTorch 2.11 with CUDA 12.8. Measured on 2026-11-12.

**Method.** `python tests/hardware/run_tests.py` starts the training processes of each scenario and samples the GPU and the system four times a second. Every scenario uses the paper's settings: an 8-year window, 30 epochs of 16 batches of 126 days, lookback 30. All numbers below are in [RESULTS.md](RESULTS.md); the raw measurements are in `results/`.

## Conclusions

1. **One window is a small job.** One model on one 8-year window trains in about 5 seconds, plus 1.5 seconds to start Python and CUDA.
2. **Nothing is saturated while it trains.** The GPU is busy 13–38% of the time, depending on K, and the process uses 0.4 of one CPU core. The step time is the fixed cost of issuing a few hundred tiny GPU operations, not the work in them.
3. **Parallel processes help little.** Two processes train 24 windows less than twice as fast as one, and three or four add almost nothing. Each process also costs about 0.9 GB of GPU memory and 4 GB of committed memory, so six would not have fit in the 18 GB of commit that was free.
4. **Training many models in one GPU pass is the better way.** 24 models train in 24 s and 36 in 34 s: 3.5 to 4 times faster than one at a time, and twice as fast as four processes. This is what the code does.
5. **So there is nothing to schedule.** `python -m afsa.run` trains up to 36 models per pass. One row of Table 2 (24 test years × 3 seeds = 72 models) took 70 s for K = 8 and 93 s for K = 30, measured end to end; the K = 30 run peaked at 9.5 GB of GPU memory with the GPU 97% busy. All seven rows should take about 11 minutes.

## What one window needs

One model, K = 8, trained alone (other K in RESULTS.md):

| Resource | Use |
| --- | --- |
| Time | 5.4 s for 30 epochs (0.14 s per epoch after the first), 0.01 s to evaluate the test year, 1.5 s of start-up |
| GPU utilization | 15% on average, 21% at most (38% and 56% for K = 100) |
| GPU memory | 0.13 GB for training, 0.19 GB for nine years of data; 0.77 GB for the whole process once the CUDA context and PyTorch's cache are counted |
| RAM | 1.7 GB resident, 3.8 GB committed |
| CPU | 0.4 of one core |

Without the GPU the same window takes about 21 s on 5.7 cores (estimated from 2 epochs), so a CPU-only machine could still run a K = 8 row of Table 2 (72 models) in about 25 minutes.

## Three ways to train the 24 windows of one seed (K = 8)

| Way | Time | Models per minute | GPU busy | GPU memory |
| --- | --- | --- | --- | --- |
| One process, one model at a time | 84–95 s | 15–17 | 22–27% | 1.2 GB |
| 2 processes | 51 s | 28 | 84% | 2.0 GB |
| 3 processes | 47 s | 31 | 87% | 2.7 GB |
| 4 processes | 47 s | 31 | 87% | 3.5 GB |
| **One process, 24 models in the same pass** | **24 s** | **59** | 87% | 5.2 GB |

- Six or more processes were not run: each commits 4.3 GB of virtual memory, and 18 GB was free.
- Splitting a batch across processes is worse than keeping it whole: 48 models took 43 s as one batch and 60 s as two processes of 24.

## Why batching wins

A training step is a few hundred GPU operations. With one model each operation carries almost no work, so the 8 ms a step takes is mostly the cost of dispatching them, and the GPU sits idle in between. Parallel processes fill some of those gaps, but the GPU serves one process at a time, so they saturate near two processes' worth of work.

Stacking M models along a leading dimension keeps the number of operations the same and puts M times the work in each. The models share no parameters, and the unit tests check that a batch of models ends with the same parameters as each model trained alone (`tests/test_train.py`).

## Batch size

| K | GPU memory per model | Largest batch run | Time per model-step | Default in `afsa.run` | 72 models take |
| --- | --- | --- | --- | --- | --- |
| 8 | 0.17 GB | 72 (12.4 GB) | 1.9 ms at 36 | 36 | 70 s (measured) |
| 30 | 0.20 GB | 48 (10.0 GB) | 2.5 ms at 48 | 36 | 93 s (measured) |
| 100 | 0.36 GB | 24 (9.3 GB) | 5.7 ms at 24 | about 27 | about 200 s |

- The gain flattens as the GPU fills: for K = 8, 24 models run at 59 a minute, 36 at 64, 48 at 67 and 72 at 69. Larger batches buy little and leave less memory free, so the default stops at 36.
- For K = 100 the K × K solves and the larger attention tensors dominate. One model alone already keeps the GPU 38% busy, so batching gains only 1.6 times.
- `afsa.run` sizes the batch from the free GPU memory (`--max-models` overrides it), so it also fits Ander's 8 GB laptop with smaller batches.

## Cautions

- **Running out of GPU memory may not fail on Windows: it slows down.** The NVIDIA driver can spill into system RAM instead of raising an error. That is the likely reason an early version, which evaluated 54 models in one pass on top of 9 GB of training cache, took 11 s for a pass that should take a fraction of a second. Evaluation is now done a few models at a time, and the default batch leaves 20% of the free memory unused.
- The GPU memory held by other programs moved between 2.0 and 3.3 GB during the tests. The "process total" figures carry that noise; PyTorch's own figures do not.
- "GPU busy" is nvidia-smi's utilization: the share of time the GPU had work, not how much it did.
- Timings vary by about 10% between runs: the same one-at-a-time scenario took 84 s once and 95 s another time.
- A model trained in a batch and the same model trained alone agree in exact arithmetic. In float32 they differ in the last digits, as two runs on different hardware would.

## Files

| File | What it is |
| --- | --- |
| `run_tests.py` | Runs the scenarios, samples the hardware, writes `results/*.json` and RESULTS.md |
| `worker.py` | One training process: trains its models and reports time and memory |
| `RESULTS.md` | Generated tables of every scenario |
| `results/` | Raw measurements and worker logs (git-ignored) |
