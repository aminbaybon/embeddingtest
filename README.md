# Persian embedding evaluation

FaMTEB / MTEB(fas) evaluation of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) and [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) on Persian embedding tasks ([Farahani et al., 2025](https://arxiv.org/abs/2502.11571)).

**Models:** [BAAI](https://huggingface.co/BAAI) (BGE-M3) and [Google](https://huggingface.co/google) (EmbeddingGemma-2).

**Datasets** are listed by Hugging Face owner. Most are from [MCINext](https://huggingface.co/MCINext). Four pair-classification sets are not:

| Owner | Role | Tasks |
|---|---|---|
| [MCINext](https://huggingface.co/MCINext) | STS, classification, clustering, retrieval | 45 listed (41 scored; FaIntent failed; 3 large corpora omitted) |
| [MTEB](https://huggingface.co/mteb) | Pair classification | [FarsTail](https://huggingface.co/datasets/mteb/FarsTail), [ParsinluEntail](https://huggingface.co/datasets/mteb/ParsinluEntail), [ParsinluQueryParaphPC](https://huggingface.co/datasets/mteb/ParsinluQueryParaphPC) |
| [alighasemi](https://huggingface.co/alighasemi) | Pair classification | [farsi_paraphrase_detection](https://huggingface.co/datasets/alighasemi/farsi_paraphrase_detection) |

Full per-task Hub IDs: **[COMPARISON_REPORT.md § Datasets](COMPARISON_REPORT.md#datasets)**.

## Results

On **45 scored tasks**, EmbeddingGemma-2 wins **20**, BGE-M3 wins **17**, and **8** are ties. Ranking below is **quality only**; wall-clock is not part of the score.

| Category | n | BGE-M3 | EmbeddingGemma-2 |
|---|---:|---:|---:|
| STS (Spearman) | 3 | **0.678** | 0.668 |
| Pair classification (AP) | 7 | **0.835** | 0.831 |
| Classification (accuracy) | 19 | 0.668 | **0.670** |
| Clustering (v-measure) | 2 | 0.474 | **0.475** |
| Retrieval (nDCG@10) | 14 | 0.351 | **0.372** |

| | BGE-M3 | EmbeddingGemma-2 |
|---|---:|---:|
| Task wins | 17 | **20** |
| Ties | 8 | 8 |
| Dense dim (native) | 1024 | 768 |

Full write-up with per-task scores: **[COMPARISON_REPORT.md](COMPARISON_REPORT.md)**

<details>
<summary>Largest classification gaps</summary>

| Task | BGE-M3 | Gemma-2 | Δ |
|---|---:|---:|---|
| PerSHOPClassification | 0.681 | 0.536 | BGE-M3 +0.145 |
| SynPerChatbotConvSASatisfaction | 0.697 | 0.797 | Gemma +0.100 |
| SynPerChatbotSatisfaction | 0.383 | 0.481 | Gemma +0.098 |
| SynPerChatbotConvSAAnger | 0.773 | 0.837 | Gemma +0.063 |
| StyleClassification | 0.695 | 0.742 | Gemma +0.047 |
| SynPerTextToneClassification | 0.738 | 0.693 | BGE-M3 +0.045 |

</details>

Protocol: 1,200 examples for STS, pair, classification, and clustering; 100 queries and 2,000 corpus documents for retrieval. PersianWeb, MS MARCO-Fa, and NQ-Fa omitted. Hardware: GTX 1650 4 GB, CUDA 12.6.

Runtime is a separate GPU wall-clock measurement, not a ranking input. Method, task list, and why a ~5× encode gap is expected from the architectures: **[COMPARISON_REPORT.md § Runtime](COMPARISON_REPORT.md#runtime)**.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Use the PyTorch index that matches your CUDA driver: https://pytorch.org/get-started/locally/

Optional: `huggingface-cli login` if a model or dataset is gated.

Place checkpoints in `../bge-m3` and `../embeddinggemma-2`, or in `models/`, or let the runner download them from Hugging Face.

```bash
python download_models.py
```

## Run

`--batch-size 0` picks a batch size from available GPU memory. EmbeddingGemma-2 uses bfloat16 when the GPU supports it, otherwise float32. BGE-M3 uses float16 on CUDA.

```bash
python eval_persian_embeddings.py \
  --models bge-m3 embeddinggemma-2 \
  --batch-size 0 \
  --max-sts 1200 --max-pair 1200 --max-clf 1200 --max-cluster 1200 \
  --max-queries 100 --max-corpus 2000 \
  --skip-mteb
```

Resume is on by default: finished tasks in `results/<model>.json` are skipped.

Print the comparison table from saved scores:

```bash
python eval_persian_embeddings.py --summarize
```

Official MTEB(fas) (optional):

```bash
python eval_persian_embeddings.py --models bge-m3 embeddinggemma-2 --mteb-only
```

## Output

| File | Contents |
|---|---|
| `results/bge-m3.json` | Per-task scores and timings |
| `results/embeddinggemma-2.json` | Per-task scores and timings |
| `results/summary.json` | Head-to-head summary |
| `COMPARISON_REPORT.md` | Written comparison |

Dataset row cache lives under `results/data_cache/` and is not committed. Model weights (`*.bin`, `*.safetensors`, `models/`) are gitignored.
