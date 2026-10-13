# Persian embedding evaluation

FaMTEB / MTEB(fas) evaluation of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) and [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) on Persian datasets from [MCINext](https://huggingface.co/MCINext) ([paper](https://arxiv.org/abs/2502.11571)).

**Results:** [COMPARISON_REPORT.md](COMPARISON_REPORT.md)

On 45 scored tasks, EmbeddingGemma-2 wins 20, BGE-M3 wins 17, and 8 are ties. Quality is close; EmbeddingGemma-2 is about 5× faster to encode on a 4 GB GPU.

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
