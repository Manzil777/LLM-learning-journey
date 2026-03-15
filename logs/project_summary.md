# Project Summary Cards

---

## Day 1 — HuggingFace Inference
**File:** `notebooks/day1_hf_inference.ipynb`
**What I built:** Loaded distilgpt2, tokenized input text, ran
model.generate(), ran 4 generation experiments changing one
parameter at a time.
**Key learning:** The model never sees raw text — only integers.
Temperature controls randomness. Low temperature collapses to
high-probability tokens and loops.
**Surprising finding:** temperature=0.3 produced 35 blank lines —
the model kept picking newline as the highest probability token.

---

## Day 2 — Tokenizer & Generation Internals
**File:** `notebooks/day2_tokenizer_and_generation.ipynb`
**What I built:** Tokenized 3 texts, ran forward pass, inspected
logits shape (batch, seq_len, vocab_size), manually predicted next
token using argmax, verified Match: True against model.generate().
Added chat template demo with SmolLM2-Instruct.
**Key learning:** model.generate() is just a loop around argmax.
logits[0, -1, :] = scores for every possible next token.
**Surprising finding:** "Artificial Intelligence" tokenizes into 3
tokens — BPE splits it into Art / ificial / intelligence.

---

## Day 3 — Reusable Generation Script
**File:** `src/generate.py`
**What I built:** A CLI script with load_model(), generate(),
save_output() functions. Accepts --prompt, --temperature, --top_k,
--top_p, --max_new_tokens, --model via argparse. Saves all
generations with timestamps to outputs/generation_examples.txt.
**Key learning:** Functions > notebook cells. if __name__ == main
makes a file both a script and an importable module.
**Surprising finding:** Colab doesn't persist files between
sessions — must write files to disk programmatically.

---

## Day 4 — Generation Algorithms
**File:** `notebooks/day4_generation_algorithms.ipynb`
**What I built:** Side-by-side comparison of 6 generation methods
on the same prompt using torch.manual_seed for reproducibility.
**Key learning:** Greedy and Beam Search both collapsed to newlines
on open-ended generation. Sampling fixed this. Real LLM APIs use
sampling, not beam search.
**Surprising finding:** Greedy and Beam Search gave the exact same
first sentence — "...change the way we think about the world."

---

## Day 5 — Parameter Controls & Testing
**File:** `notebooks/day5_parameter_controls_and_testing.ipynb`
**What I built:** Reusable test_generation() function, ran 5
systematic parameter sweeps across temperature, top_k, top_p,
repetition_penalty, and multiple prompts.
**Key learning:** Parameters interact — temperature reshapes the
distribution, top_k cuts the tail, top_p refines further.
Industry standard: temp=0.7, top_k=50, top_p=0.9, rep_penalty=1.2
**Surprising finding:** Low temperature AND low top_p both caused
the same repetition loop through different mechanisms.

---

## Day 6 — Sentiment Analysis Pipeline
**File:** `notebooks/day6_sentiment_analysis_pipeline.ipynb`
**What I built:** Full sentiment pipeline evaluation — basic tests,
confidence analysis, sarcasm tests (5/5 failed), edge cases,
batch processing, failure analysis with accuracy count.
Model: distilbert-base-uncased-finetuned-sst-2-english (2 labels).
**Key learning:** Confidence score ≠ accuracy. Model was 99.89%
confident "Oh great, another Monday" was POSITIVE. No neutral
class forces all inputs into binary buckets.
**Surprising finding:** "The food was fine" → POSITIVE with 0.9998
confidence. The model learned "fine" from positive movie reviews.
```

---

