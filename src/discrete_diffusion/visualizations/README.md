# Visualization Scripts

This folder contains small terminal-oriented tools for inspecting the SIK forward process.

The scripts are focused on three different questions:

1. What does the SIK kernel think is semantically close to a token?
2. How does a sequence evolve under the SIK noising process over time?
3. What does the uniform teleport schedule `lambda(t)` look like?

## Files

### `analyze_kernel.py`

Inspects the local SIK kernel around one or more tokens.

What it shows:
- nearest neighbors for a token
- raw embedding distance
- kernel score (`logR`)
- transition probabilities at a few times

Use this when:
- a sampled jump looks surprising
- you want to check whether two tokens are actually neighbors
- you want to debug the effect of `epsilon`, `gamma`, or variable bandwidth

Notes:
- defaults are aligned with the animation cache setup
- tokenization matters a lot: `dom`, ` master`, `master`, and ` dom` are different GPT-2 tokens

Example:

```bash
python src/discrete_diffusion/visualizations/analyze_kernel.py --word dom --word master --top_k 32
```

### `visualize_sik_knn.py`

Runs one SIK trajectory for a sentence with the k-NN kernel and reconstructs the sequence at several fixed time snapshots.

What it shows:
- one sampled trajectory from `t=0` to `t=1`
- the decoded sequence at times like `0.0`, `0.1`, ..., `1.0`

Use this when:
- you want a compact, static view of how SIK corruption evolves
- you want to inspect the overall behavior without live animation

Example:

```bash
python src/discrete_diffusion/visualizations/visualize_sik_knn.py --sentence "The cat sat on the mat"
```

### `cache_sik_demo.py`

Builds and saves a cached GPT-2 SIK k-NN kernel to `gpt2_sik_cache.pt` at the repo root.

What it does:
- loads GPT-2 embeddings
- builds the k-NN graph
- stores neighbor indices and kernel scores

Use this when:
- you want `animate_sik.py` to start instantly
- you want the animation to use a fixed cached kernel

Current cache kernel defaults:
- `epsilon = 0.01`
- `gamma = 0.0`
- `variable_bandwidth = True`
- `k_neighbors = 7`
- `top_k = 64`

Example:

```bash
python src/discrete_diffusion/visualizations/cache_sik_demo.py
```

### `animate_sik_knn.py`

Animates one sampled SIK noising trajectory in the terminal using the cached kernel.

What it shows:
- live sequence evolution over time
- jump history for each position
- color-coded jump type
- current teleport probability
- ASCII plot of `lambda(t)`

Color meaning:
- green: kernel / semantic jump
- yellow: uniform / teleport jump

Important:
- this script expects `gpt2_sik_cache.pt`
- run `cache_sik_demo.py` first if the cache file does not exist

Example:

```bash
python src/discrete_diffusion/visualizations/animate_sik_knn.py --text "The cat sat on the mat"
```

### `plot_lambda.py`

Plots the teleport schedule `lambda(t)` in the terminal.

What it shows:
- the shape of the uniform-mixture probability over time
- parameters loaded from `configs/forward_process/sik_knn.yaml`

Use this when:
- you want to understand the schedule independently of token trajectories
- you are tuning `lambda_min`, `lambda_sigmoid_s`, or `lambda_t0`

Example:

```bash
python src/discrete_diffusion/visualizations/plot_lambda.py
```

## Suggested Order

If you are trying to understand the current SIK setup, a good order is:

1. `plot_lambda.py`
2. `analyze_kernel.py`
3. `visualize_sik_knn.py`
4. `cache_sik_demo.py`
5. `animate_sik_knn.py`

## Important Caveats

### Offline GPT-2 cache

These scripts expect GPT-2 assets to already be present in the local Hugging Face cache. They are written to run in offline environments such as cluster jobs.

### Token-level, not word-level

Most surprising behavior comes from GPT-2 tokenization. The visualizations operate on token ids, not words. A displayed decoded string can hide the distinction between tokens with and without leading whitespace.

### Analyzer vs animation consistency

`analyze_kernel.py` and `animate_sik.py` have been aligned so that their default kernel settings match the cached animation kernel:

- `epsilon = 0.01`
- `gamma = 0.0`
- `variable_bandwidth = True`
- `k_neighbors = 7`
- `temperature_beta = 0.0`

That makes it easier to compare a jump seen in the animation with the neighborhood reported by the analyzer.
