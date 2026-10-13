# BGE-M3 vs EmbeddingGemma-2 on Persian embedding tasks

Comparison of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) and [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) on FaMTEB / MTEB(fas) Persian datasets ([Farahani et al., 2025](https://arxiv.org/abs/2502.11571)).

| | |
|---|---|
| Hardware | NVIDIA GeForce GTX 1650, 4 GB, CUDA 12.6 |
| Software | PyTorch 2.14.1, sentence-transformers 6.1, transformers 5.19 |
| Protocol | 1,200 examples for STS, pair, classification, and clustering; 100 queries and 2,000 corpus documents for retrieval |
| Large corpora | PersianWeb, MS MARCO-Fa, and NQ-Fa omitted |
| Date | October 2026 |

## Summary

The two models are close on quality and far apart on throughput.

On 45 scored tasks, EmbeddingGemma-2 wins 20, BGE-M3 wins 17, and 8 are ties. Category means differ by at most two points except retrieval, where Gemma leads (nDCG@10 0.372 vs 0.351). BGE-M3 is stronger on STS, several pair-classification sets, and PerSHOP. Gemma is stronger on retrieval and several chatbot classification tasks.

On this GPU, EmbeddingGemma-2 encodes about five times faster than BGE-M3.

## Models

| | BGE-M3 | EmbeddingGemma-2 |
|---|---|---|
| Architecture | XLM-RoBERTa, 1024-d | Gemma embedding, 768-d, text encoder |
| Parameters | ~568M | ~270M (text) |
| Dtype | float16 | bfloat16 |
| Batch size | 8 | 4 |
| Sequence length | 512 | 512 |

EmbeddingGemma-2 is evaluated in text-only mode. It requires bfloat16 or float32; float16 is not used.

## Quality

Primary metrics: STS Spearman, pair average precision, classification accuracy (5-fold logistic regression), clustering v-measure, retrieval nDCG@10.

| Category | n | BGE-M3 | EmbeddingGemma-2 |
|---|---:|---:|---:|
| STS | 3 | 0.678 | 0.668 |
| Pair classification | 7 | 0.835 | 0.831 |
| Classification | 19 | 0.668 | 0.670 |
| Clustering | 2 | 0.474 | 0.475 |
| Retrieval | 14 | 0.351 | 0.372 |

### STS

| Task | BGE-M3 | Gemma-2 |
|---|---:|---:|
| FarSick | 0.671 | 0.666 |
| SynPerSTS | 0.885 | 0.857 |
| Query2Query | 0.479 | 0.483 |

### Pair classification

| Task | BGE-M3 | Gemma-2 |
|---|---:|---:|
| SynPerQAPC | 0.940 | 0.936 |
| SynPerTextKeywordsPC | 0.974 | 0.952 |
| SynPerChatbotRAGFAQPC | 0.647 | 0.678 |
| FarsTail | 0.731 | 0.755 |
| FarsiParaphraseDetection | 0.955 | 0.959 |
| ParsinluQueryParaphPC | 0.912 | 0.886 |
| ParsinluEntail | 0.687 | 0.654 |

### Classification

Largest gaps:

| Task | BGE-M3 | Gemma-2 | Δ |
|---|---:|---:|---:|
| PerSHOPClassification | 0.681 | 0.536 | BGE-M3 +0.145 |
| SynPerChatbotConvSASatisfaction | 0.697 | 0.797 | Gemma +0.100 |
| SynPerChatbotSatisfaction | 0.383 | 0.481 | Gemma +0.098 |
| SynPerChatbotConvSAAnger | 0.773 | 0.837 | Gemma +0.063 |
| StyleClassification | 0.695 | 0.742 | Gemma +0.047 |
| SynPerTextToneClassification | 0.738 | 0.693 | BGE-M3 +0.045 |

Remaining classification tasks differ by less than 0.04, including several exact ties on small ConvSA label sets. FaIntent failed for both models because a rare class has too few examples for stratified cross-validation; it is excluded from scoring.

### Clustering

| Task | BGE-M3 | Gemma-2 |
|---|---:|---:|
| BeytooteClustering | 0.568 | 0.540 |
| SIDClustering | 0.381 | 0.410 |

### Retrieval

Corpus is capped at 2,000 documents plus gold hits. FiQA-2018-Fa, Quora-Fa, and SynPerQARetrieval are near floor for both models under this cap and are not used to rank the embedders.

| Task | BGE-M3 | Gemma-2 |
|---|---:|---:|
| SAMSumFa | 1.000 | 0.991 |
| SynPerChatbotSumSRetrieval | 0.585 | 0.594 |
| SynPerChatbotRAGSumSRetrieval | 0.798 | 0.806 |
| SciFact-Fa | 0.307 | 0.352 |
| ArguAna-Fa | 0.358 | 0.349 |
| FiQA2018-Fa | 0.039 | 0.040 |
| Quora-Fa | 0.020 | 0.020 |
| SCIDOCS-Fa | 0.082 | 0.087 |
| TRECCOVID-Fa | 0.566 | 0.552 |
| NFCorpus-Fa | 0.566 | 0.576 |
| SynPerQARetrieval | 0.010 | 0.010 |
| SynPerChatbotTopicsRetrieval | 0.250 | 0.392 |
| SynPerChatbotRAGTopicsRetrieval | 0.275 | 0.355 |
| SynPerChatbotRAGFAQRetrieval | 0.064 | 0.076 |

Gemma’s retrieval advantage is largest on synthetic chatbot topic retrieval and SciFact-Fa. BGE-M3 leads on SAMSumFa, ArguAna-Fa, and TRECCOVID-Fa.

## Runtime

Task times include encoding and metric computation.

| | BGE-M3 | EmbeddingGemma-2 | Ratio |
|---|---:|---:|---:|
| 14 retrieval tasks | 7,942 s | 1,387 s | 5.7× |
| DigiMagClassification | 569 s | 114 s | 5.0× |
| BeytooteClustering | 152 s | 32 s | 4.8× |

Encode-dominated retrieval (SciFact, ArguAna, SCIDOCS, NFCorpus, summary retrieval) is consistently about 5× faster with EmbeddingGemma-2. That matches the smaller text encoder and 768-d output, even with a smaller batch than BGE-M3.

Classification times also include logistic-regression cross-validation.

## Recommendation

- **Throughput and retrieval:** EmbeddingGemma-2.
- **STS, keyword/QA pair matching, and PerSHOP-style classification:** BGE-M3.
- Overall on this protocol: EmbeddingGemma-2, given more task wins, a higher retrieval mean, and substantially lower encode time.

## Reproducibility

```bash
python eval_persian_embeddings.py \
  --models bge-m3 embeddinggemma-2 \
  --batch-size 0 \
  --max-sts 1200 --max-pair 1200 --max-clf 1200 --max-cluster 1200 \
  --max-queries 100 --max-corpus 2000 \
  --skip-mteb
```

Scores: `results/bge-m3.json`, `results/embeddinggemma-2.json`, `results/summary.json`.
