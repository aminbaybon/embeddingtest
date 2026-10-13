# BGE-M3 vs EmbeddingGemma-2 on Persian embedding tasks

Comparison of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) and [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) on FaMTEB / MTEB(fas) Persian tasks ([Farahani et al., 2025](https://arxiv.org/abs/2502.11571)). Dataset Hub owners are listed in [Datasets](#datasets).

| | |
|---|---|
| Hardware | NVIDIA GeForce GTX 1650, 4 GB, CUDA 12.6 |
| Software | PyTorch 2.14.1, sentence-transformers 6.1, transformers 5.19 |
| Protocol | 1,200 examples for STS, pair, classification, and clustering; 100 queries and 2,000 corpus documents for retrieval |
| Large corpora | PersianWeb, MS MARCO-Fa, and NQ-Fa omitted |
| Date | October 2026 |

## Summary

The two models are close on quality. **Wins and category means use quality metrics only.** Runtime is reported later as a separate measurement and is not used to rank the models.

On 45 scored tasks, EmbeddingGemma-2 wins 20, BGE-M3 wins 17, and 8 are ties. Category means differ by at most two points except retrieval, where Gemma leads (nDCG@10 0.372 vs 0.351).

## Models

| | BGE-M3 | EmbeddingGemma-2 |
|---|---|---|
| Hub owner | [BAAI](https://huggingface.co/BAAI) | [Google](https://huggingface.co/google) |
| Encoder | XLM-RoBERTa | Gemma embedding text encoder |
| Parameters | ~568M | ~270M (text) |
| Layers | 24 | 24 |
| Hidden / FFN | 1024 / 4096 | 512 / 2048 |
| Attention | Full attention, 16 heads | 20 sliding-window + 4 full; GQA (4 heads, 2 KV) |
| Output dim | 1024 | 768 |
| Dtype | float16 | bfloat16 |
| Batch size | 8 | 4 |
| Sequence length | 512 | 512 |

EmbeddingGemma-2 is evaluated in text-only mode. It requires bfloat16 or float32; float16 is not used.

## Datasets

Hub IDs as loaded. The benchmark is FaMTEB / MTEB(fas); **the Hugging Face owner is not always MCINext.**

| Owner | n listed | n scored |
|---|---:|---:|
| [MCINext](https://huggingface.co/MCINext) | 45 | 41 |
| [MTEB](https://huggingface.co/mteb) | 3 | 3 |
| [alighasemi](https://huggingface.co/alighasemi) | 1 | 1 |

FaIntent (MCINext) failed CV and is excluded from scores. PersianWeb, MS MARCO-Fa, and NQ-Fa (MCINext) were listed and omitted as large corpora.

### STS — MCINext

| Task | Dataset |
|---|---|
| FarSick | [MCINext/farsick-sts](https://huggingface.co/datasets/MCINext/farsick-sts) |
| SynPerSTS | [MCINext/synthetic-persian-sts](https://huggingface.co/datasets/MCINext/synthetic-persian-sts) |
| Query2Query | [MCINext/query-to-query-sts](https://huggingface.co/datasets/MCINext/query-to-query-sts) |

### Pair classification

| Task | Dataset | Owner |
|---|---|---|
| SynPerQAPC | [MCINext/synthetic-persian-qa-pair-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-pair-classification) | MCINext |
| SynPerTextKeywordsPC | [MCINext/synthetic-persian-text-keyword-pair-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-text-keyword-pair-classification) | MCINext |
| SynPerChatbotRAGFAQPC | [MCINext/synthetic-persian-chatbot-rag-faq-pair-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-pair-classification) | MCINext |
| FarsTail | [mteb/FarsTail](https://huggingface.co/datasets/mteb/FarsTail) | MTEB |
| FarsiParaphraseDetection | [alighasemi/farsi_paraphrase_detection](https://huggingface.co/datasets/alighasemi/farsi_paraphrase_detection) | alighasemi |
| ParsinluQueryParaphPC | [mteb/ParsinluQueryParaphPC](https://huggingface.co/datasets/mteb/ParsinluQueryParaphPC) | MTEB |
| ParsinluEntail | [mteb/ParsinluEntail](https://huggingface.co/datasets/mteb/ParsinluEntail) | MTEB |

### Classification — MCINext

| Task | Dataset |
|---|---|
| SIDClassification | [MCINext/sid-classification](https://huggingface.co/datasets/MCINext/sid-classification) |
| DigiMagClassification | [MCINext/DigiMag](https://huggingface.co/datasets/MCINext/DigiMag) |
| FaIntent | [MCINext/FaIntent](https://huggingface.co/datasets/MCINext/FaIntent) (failed; not scored) |
| PerSHOPClassification | [MCINext/pershop-classification](https://huggingface.co/datasets/MCINext/pershop-classification) |
| StyleClassification | [MCINext/style-classification](https://huggingface.co/datasets/MCINext/style-classification) |
| SynPerTextToneClassification | [MCINext/synthetic-persian-text-tone-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-text-tone-classification) |
| SynPerChatbotSatisfaction | [MCINext/synthetic-persian-chatbot-satisfaction-level-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-satisfaction-level-classification) |
| SynPerChatbotToneUser | [MCINext/synthetic-persian-chatbot-tone-user-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-tone-user-classification) |
| SynPerChatbotToneChatbot | [MCINext/synthetic-persian-chatbot-tone-chatbot-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-tone-chatbot-classification) |
| SynPerChatbotRAGToneUser | [MCINext/synthetic-persian-chatbot-rag-tone-user-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-tone-user-classification) |
| SynPerChatbotRAGToneChatbot | [MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-tone-chatbot-classification) |
| SynPerChatbotConvSAHappiness | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-happiness) |
| SynPerChatbotConvSASadness | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-sadness) |
| SynPerChatbotConvSAAnger | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-anger) |
| SynPerChatbotConvSAFear | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-fear) |
| SynPerChatbotConvSALove | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-love) |
| SynPerChatbotConvSASurprise | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-surprise) |
| SynPerChatbotConvSASatisfaction | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-satisfaction) |
| SynPerChatbotConvSAFriendship | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-friendship) |
| SynPerChatbotConvSAJealousy | [MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-conversational-sentiment-analysis-jealousy) |

### Clustering — MCINext

| Task | Dataset |
|---|---|
| BeytooteClustering | [MCINext/beytoote-clustering](https://huggingface.co/datasets/MCINext/beytoote-clustering) |
| SIDClustering | [MCINext/sid-clustering](https://huggingface.co/datasets/MCINext/sid-clustering) |

### Retrieval — MCINext

| Task | Dataset | Status |
|---|---|---|
| SAMSumFa | [MCINext/samsum-fa](https://huggingface.co/datasets/MCINext/samsum-fa) | scored |
| SynPerChatbotSumSRetrieval | [MCINext/synthetic-persian-chatbot-summary-retrieval](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-summary-retrieval) | scored |
| SynPerChatbotRAGSumSRetrieval | [MCINext/synthetic-persian-chatbot-rag-summary-retrieval](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-summary-retrieval) | scored |
| SciFact-Fa | [MCINext/scifact-fa-v2](https://huggingface.co/datasets/MCINext/scifact-fa-v2) | scored |
| ArguAna-Fa | [MCINext/arguana-fa-v2](https://huggingface.co/datasets/MCINext/arguana-fa-v2) | scored |
| FiQA2018-Fa | [MCINext/fiqa-fa-v2](https://huggingface.co/datasets/MCINext/fiqa-fa-v2) | scored |
| Quora-Fa | [MCINext/quora-fa-v2](https://huggingface.co/datasets/MCINext/quora-fa-v2) | scored |
| SCIDOCS-Fa | [MCINext/scidocs-fa-v2](https://huggingface.co/datasets/MCINext/scidocs-fa-v2) | scored |
| TRECCOVID-Fa | [MCINext/trec-covid-fa-v2](https://huggingface.co/datasets/MCINext/trec-covid-fa-v2) | scored |
| NFCorpus-Fa | [MCINext/nfcorpus-fa](https://huggingface.co/datasets/MCINext/nfcorpus-fa) | scored |
| SynPerQARetrieval | [MCINext/synthetic-persian-qa-retrieval](https://huggingface.co/datasets/MCINext/synthetic-persian-qa-retrieval) | scored |
| SynPerChatbotTopicsRetrieval | [MCINext/synthetic-persian-chatbot-topics-retrieval](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-topics-retrieval) | scored |
| SynPerChatbotRAGTopicsRetrieval | [MCINext/synthetic-persian-chatbot-rag-topics-retrieval](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-topics-retrieval) | scored |
| SynPerChatbotRAGFAQRetrieval | [MCINext/synthetic-persian-chatbot-rag-faq-retrieval](https://huggingface.co/datasets/MCINext/synthetic-persian-chatbot-rag-faq-retrieval) | scored |
| PersianWebDocumentRetrieval | [MCINext/persian-web-document-retrieval](https://huggingface.co/datasets/MCINext/persian-web-document-retrieval) | omitted (large) |
| MSMARCO-Fa | [MCINext/msmarco-fa](https://huggingface.co/datasets/MCINext/msmarco-fa) | omitted (large) |
| NQ-Fa | [MCINext/nq-fa](https://huggingface.co/datasets/MCINext/nq-fa) | omitted (large) |

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

Runtime is **not** part of the quality ranking. It is a wall-clock observation on one GPU, under the protocol above.

### What was measured

Each task records `seconds` as `time.time()` around the whole task: dataset load, `SentenceTransformer.encode`, and the metric (cosine / logistic regression / v-measure / nDCG). It is **not** isolated tokens/s and **not** a matched microbenchmark with equal batch size and dtype.

Both models in the table below ran on the same **NVIDIA GeForce GTX 1650 4 GB**, CUDA 12.6, `max_seq_length=512`. BGE-M3 used float16 with batch 8; EmbeddingGemma-2 used bfloat16 with batch 4 (Gemma forbids float16). Gemma is still faster with the smaller batch.

STS and pair-classification wall-clocks are omitted. Those traces are not a matched GPU encode comparison, so they are not used to estimate throughput.

### Encode-dominated GPU tasks

These retrieval sets encode on the order of 1,000–2,300 passages at 512 tokens. Encode dominates the timer.

| Task | n (queries + corpus) | BGE-M3 | Gemma-2 | BGE / Gemma |
|---|---:|---:|---:|---:|
| SciFact-Fa | 100 + 1,937 | 1,342.8 s | 252.4 s | 5.3× |
| ArguAna-Fa | 100 + 1,971 | 958.3 s | 189.7 s | 5.1× |
| SCIDOCS-Fa | 100 + 1,606 | 801.9 s | 153.8 s | 5.2× |
| NFCorpus-Fa | 100 + 2,299 | 1,812.7 s | 302.4 s | 6.0× |
| SAMSumFa | 100 + 1,561 | 159.7 s | 35.8 s | 4.5× |
| SynPerChatbotSumSRetrieval | 100 + 1,537 | 214.7 s | 40.1 s | 5.4× |
| SynPerChatbotRAGSumSRetrieval | 100 + 1,087 | 142.4 s | 28.3 s | 5.0× |
| **Mean of ratios** | | | | **5.2×** |

Supporting checks on the same GPU (encode plus CPU metric): DigiMagClassification 568.8 s vs 114.2 s (5.0×); BeytooteClustering 151.9 s vs 31.8 s (4.8×). Classification includes 5-fold logistic regression, so those two are not pure encode.

### What the ~5× is not

The sum over all 14 retrieval tasks is 7,942 s vs 1,387 s (**5.7×**). That aggregate is **not** the encode estimate. TRECCOVID-Fa is 767.8 s vs 35.4 s (22×) on only 50 queries and 376 documents; encode volume is too small for a 22× gap to be an encode result, so it is treated as an I/O / document-length outlier and excluded from the 5.2× mean.

Short-text retrieval (Quora-Fa, topic sets) is 2.6–4.0×, as expected when encode is a smaller share of wall-clock.

### Why ~5× is expected on this GPU

The gap is larger than the parameter ratio (~568M / ~270M ≈ **2.1×**) because compute scales with width and attention, not only parameter count:

- FFN matmuls scale as hidden × intermediate: 1024×4096 vs 512×2048 ≈ **4×**.
- Both have 24 layers, but BGE uses full self-attention in every layer (16 heads). Gemma uses sliding-window attention in 20 layers and full attention in 4, with grouped-query attention. At the 512-token cap the sliding window is also 512, so attention is not much cheaper here; width (1024 vs 512) still costs about **2×**.
- On 4 GB VRAM, BGE sits closer to the memory ceiling, which lowers occupancy relative to the smaller Gemma encoder.

A 5× wall-clock ratio on this card is therefore in the same range as a 3–4× compute gap plus memory-bound overhead. It is **not** a claim that EmbeddingGemma-2 is 5× faster on large-batch servers; there the gap should move closer to the 2–3× parameter/width ratio.

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
