#  LLM Learning — Concepts & Experiments

A personal learning repo documenting how Large Language Models work. From tokenization to generation. Built while studying AI & DS at CMRIT.

---

##  Table of Contents

- [Overview](#overview)
- [Concepts](#concepts)
  - [Tokenization](#tokenization)
  - [Logits & Next Token Prediction](#logits--next-token-prediction)
  - [Generation Parameters](#generation-parameters)
- [Examples](#examples)
- [References](#references)

---

## Overview

This repo contains notes, code experiments, and Jupyter notebooks exploring how LLMs work under the hood. Focusing on the inference pipeline: tokenization → model forward pass → logits → sampling → output.

---

## Concepts

### Tokenization

Tokenization is the process of converting raw text into a sequence of integers (tokens) that the model can process.

Modern LLMs don't tokenize by word or character. They use **Byte Pair Encoding (BPE)** or similar subword algorithms. This lets the model handle rare and unknown words efficiently by breaking them into smaller, known subword units.

**Key ideas:**

- Each token maps to an ID in the model's **vocabulary** (typically 32k–128k tokens)
- Common words are often a single token; rare words get split into multiple tokens
- Special tokens like `<BOS>` (beginning of sequence) and `<EOS>` (end of sequence) are added automatically

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)
# [15496, 11, 703, 389, 345, 30]

print(tokenizer.decode(tokens))
# "Hello, how are you?"

# See individual token strings
print([tokenizer.decode([t]) for t in tokens])
# ['Hello', ',', ' how', ' are', ' you', '?']
```

**Special tokens:**

| Token | Meaning |
|---|---|
| `<BOS>` / `<s>` | Beginning of sequence |
| `<EOS>` / `</s>` | End of sequence |
| `<PAD>` | Padding (for batching) |
| `<UNK>` | Unknown token (rare in BPE) |

> **Note:** The same sentence can tokenize differently across models. Always use the tokenizer that matches your model.

---

### Logits & Next Token Prediction

At its core, an LLM is a **next-token predictor**. Given a sequence of tokens, it outputs a probability distribution over the entire vocabulary  and the next token is sampled from that distribution.

**The pipeline:**

```
Input Text → Tokenizer → Token IDs → Model → Logits → Softmax → Probabilities → Sample → Next Token
```

**What are logits?**

Logits are the raw, unnormalized scores the model assigns to every token in the vocabulary. They're the final layer's output before any normalization. A higher logit = the model thinks that token is more likely to come next.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

input_text = "The capital of France is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(input_ids)

# Logits shape: (batch_size, sequence_length, vocab_size)
logits = outputs.logits

# Get logits for the LAST token position (next token prediction)
next_token_logits = logits[:, -1, :]  # shape: (1, vocab_size)

# Convert to probabilities
probs = torch.softmax(next_token_logits, dim=-1)

# Greedy decoding — pick the highest probability token
next_token_id = torch.argmax(probs, dim=-1)
print(tokenizer.decode(next_token_id))
# "Paris"
```

**Greedy vs Sampling:**

| Strategy | How it works | Trade-off |
|---|---|---|
| Greedy | Always pick the highest prob token | Deterministic but repetitive |
| Temperature sampling | Sample from distribution | More varied, can be incoherent |
| Top-k sampling | Sample from top K tokens only | Balanced |
| Top-p (nucleus) | Sample from tokens summing to prob p | Adaptive, widely used |

---

### Generation Parameters

When calling `model.generate()`, these parameters control how tokens are sampled at each step.

```python
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.2,
    do_sample=True,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

**Parameter reference:**

| Parameter | Type | Effect |
|---|---|---|
| `max_new_tokens` | `int` | Maximum number of tokens to generate |
| `temperature` | `float` | Scales logits before softmax. Lower = more confident/deterministic. Higher = more random. `1.0` = unchanged |
| `top_k` | `int` | At each step, only sample from the top K most likely tokens |
| `top_p` | `float` | Nucleus sampling — only sample from the smallest set of tokens whose cumulative probability ≥ p |
| `do_sample` | `bool` | If `False`, uses greedy decoding. Must be `True` for temperature/top_k/top_p to apply |
| `repetition_penalty` | `float` | Penalizes tokens that have already appeared. `1.0` = no penalty, `>1.0` = discourages repetition |
| `num_beams` | `int` | Beam search — explore multiple sequences in parallel. `1` = greedy/sampling |

**Temperature intuition:**

```
temp = 0.1  → Very confident, near-deterministic, "safe" outputs
temp = 0.7  → Balanced creativity and coherence  ← sweet spot
temp = 1.0  → Raw model distribution
temp = 1.5  → Very random, may be incoherent
```

---

## Examples

### Temperature Effect

**Prompt:** `"Once upon a time, in a land far away,"`

```
temperature=0.1
→ "...there lived a king who ruled his kingdom with wisdom and justice.
   His people loved him dearly and the land prospered under his reign."

temperature=0.7
→ "...a young inventor stumbled upon a clockwork forest that ticked
   and hummed with secrets older than memory."

temperature=1.5
→ "...dolphins elected a parliament and proceeded to outlaw Tuesdays
   for reasons that made perfect sense to everyone involved."
```

---

### top_p Effect

**Prompt:** `"Artificial intelligence will"`

```
top_p=0.5  (conservative — samples from very likely tokens only)
→ "...transform industries and change the way we work and live."

top_p=0.9  (wider pool — more variety)
→ "...rewrite the social contract in ways we haven't even begun to imagine."

top_p=1.0  (full vocabulary — anything goes)
→ "...probably need a good therapist, honestly, given everything."
```

---

### Repetition Penalty Effect

**Prompt:** `"The sun is"`

```
repetition_penalty=1.0  (none)
→ "...bright. The sun is warm. The sun is the center of our solar system.
   The sun is..."

repetition_penalty=1.3
→ "...bright and warm, sitting at the center of our solar system,
   roughly 93 million miles from Earth."
```

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer paper
- [HuggingFace Transformers Docs](https://huggingface.co/docs/transformers)
- [Andrej Karpathy — nanoGPT](https://github.com/karpathy/nanoGPT)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/)
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's tokenizer
