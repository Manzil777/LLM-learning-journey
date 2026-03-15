# LLM Learning Journey

A structured, hands-on journey from ML student to applied LLM engineer.
Built in public. Every session produces a working artifact.

**Current Level:** 4/10 → targeting 6+ before graduation
**Stack:** Python · HuggingFace Transformers · PyTorch · Google Colab
**Timeline:** 18 months · 6–8 hours/week

---

## What I've Built

| Day | Notebook | What it does |
|-----|----------|--------------|
| 1 | `day1_hf_inference.ipynb` | Load distilgpt2, tokenize input, run generation, experiment with temperature / top_k / top_p / repetition_penalty |
| 2 | `day2_tokenizer_and_generation.ipynb` | Tokenization deep dive, forward pass, manual next-token prediction via argmax, verified against model.generate(), chat template demo |
| 3 | `src/generate.py` | Reusable CLI generation script with argparse — load model, accept prompt, generate, save outputs |
| 4 | `day4_generation_algorithms.ipynb` | Side-by-side comparison of Greedy, Beam Search, Top-K, Top-P, and Combined sampling |
| 5 | `day5_parameter_controls_and_testing.ipynb` | Systematic parameter sweeps with reusable test function — temperature, top_k, top_p, repetition_penalty |
| 6 | `day6_sentiment_analysis_pipeline.ipynb` | Sentiment pipeline, confidence score analysis, sarcasm tests, edge cases, failure analysis |

---

## Repository Structure
```
LLM-learning-journey/
├── notebooks/          # Daily experiment notebooks
├── src/                # Reusable Python scripts
│   └── generate.py     # CLI text generation tool
├── outputs/            # Generated text examples
├── logs/               # Daily session logs
├── requirements.txt
└── README.md
```

---

## Key Concepts Covered

- **Tokenization** — BPE, subword units, input_ids, vocab size
- **Generation internals** — logits, argmax, the generate() loop
- **Generation algorithms** — Greedy, Beam Search, Sampling
- **Parameter controls** — temperature, top_k, top_p, repetition_penalty
- **Chat templates** — instruct vs base models, special tokens
- **HuggingFace pipelines** — sentiment analysis, confidence scores
- **Failure analysis** — sarcasm, mixed sentiment, neutral class problem

---

## Background

3rd year BE student in AI & Data Science at CMRIT, Bangalore.
Previously built a transformer-based protein classification system
using ESM embeddings across ~320 protein classes, compared against
a CNN baseline.

This repo tracks my transition from ML model training to applied
LLM engineering.

---

## Roadmap

- [x] Phase 1 — HuggingFace & Generation Fundamentals
- [ ] Phase 2 — RAG Systems (embeddings, FAISS, retrieval pipelines)
- [ ] Phase 3 — Fine-tuning with LoRA/PEFT
- [ ] Phase 4 — Deployment (FastAPI, Docker)

---

## Learning Philosophy

> 70% building · 30% learning
> Every session produces a concrete artifact.
> Consistency over intensity.
