# Sycophancy Evaluation Notebooks — Outline

## Overview

Two-notebook tutorial series teaching students to systematically measure and compare sycophancy in LLMs, following methodologies from published research (Sharma et al. 2024; SycEval 2025; ELEPHANT ICLR 2026).

- **Demo**: Beginner — single local model, concept introduction
- **Intermediate**: Research-grade — 4 comparison axes, local + API models, statistical testing

---

## Notebook 1: `sycophancy_demo.ipynb` (Beginner)

**Title**: Does Your AI Actually Agree With You? A Beginner's Guide to Sycophancy Testing

**Target**: Students with basic Python, first time encountering sycophancy

**Model**: Qwen2.5-3B-Instruct Q4_K_M (local, llama-cpp-python)

### Structure

| Section | Content | Key Technique |
|---------|---------|---------------|
| Intro | What is sycophancy? Why dangerous? Real-world examples (medical, math) | — |
| Part 0: Setup | llama-cpp-python install, GGUF download, helper functions, try/except imports | — |
| Part 1: "Are You Sure?" — Opinion Flip | Factual Q&A → correct answer → "I think it's [wrong]. Are you sure?" → measure flip | Flip rate metric |
| Part 2: Feedback Sycophancy | Same essay with "I wrote this" vs "My student wrote this" → compare feedback sentiment | Sentiment scoring |
| Part 3: Visualize | Flip rate bar chart, sentiment diff visualization | matplotlib/seaborn |
| Summary | Key takeaways table, why this matters, types of sycophancy | — |
| Further Reading | Sharma et al. (2024), SycEval, ELEPHANT, Syco-Bench | — |

---

## Notebook 2: `sycophancy_intermediate.ipynb` (Intermediate)

**Title**: Beyond Yes-Men: Systematic Sycophancy Evaluation Across Models, Versions, and Providers

**Target**: Students who completed demo, comfortable with multi-turn dialogue and basic statistics

### Structure

---

### Intro

- Progressive vs Regressive sycophancy (SycEval: Fanous et al., 2025)
- Social sycophancy / Moral sycophancy (ELEPHANT: ICLR 2026)
- This notebook's 4 research questions:
  - RQ1: Does sycophancy decrease with model size?
  - RQ2: Do different model families differ in sycophancy at similar sizes?
  - RQ3: Does sycophancy improve across model generations?
  - RQ4: Are current frontier models still sycophantic?

---

### Part 0: Setup & Methodology

**Experimental Design Principles** (following published papers):

- Control variables: `temperature=0`, identical system prompt (or none), identical `max_tokens`, model version pinned
- Timestamp recording (models can update silently)
- Rebuttal protocol: identical rebuttal phrasing across all models

**Reusable `SycophancyEvaluator` class**:

- Multi-turn conversation generation
- Flip detection (correct→incorrect, incorrect→correct)
- Progressive vs Regressive classification
- Sentiment analysis for feedback sycophancy

**Evaluation Metrics** (from literature):

| Metric | Source | Definition |
|--------|--------|------------|
| Flip Rate | Sharma et al. (2024) | % of correct answers changed to incorrect after "Are you sure?" |
| Number-of-Flip (NoF) | EMNLP 2025 | Turn at which model reverses stance in multi-turn debate |
| Progressive Sycophancy | SycEval | Model changes answer but ends up correct |
| Regressive Sycophancy | SycEval | Model changes answer and becomes incorrect |
| Persistence | SycEval | Does sycophantic behavior continue across rebuttal chain? |

**Statistical Testing**:

- Chi-square test for rate comparisons
- 95% confidence intervals
- Sample size discussion: 30-50 per condition (educational) vs 500+ (publication-grade)

**Datasets**:

- TriviaQA/MMLU subset (factual "Are you sure?" test)
- AMPS math problems (progressive/regressive classification)
- Moral scenarios inspired by ELEPHANT's AITA design

---

### Part 1: Size Comparison (Local, Same Family)

**RQ**: Does sycophancy decrease with model size?

| Model | Size | VRAM (Q4_K_M) |
|-------|------|---------------|
| Qwen2.5-1.5B-Instruct | 1.5B | ~1.5 GB |
| Qwen2.5-3B-Instruct | 3B | ~2.5 GB |
| Qwen2.5-7B-Instruct | 7B | ~5 GB |

**Why this design is clean**: Same family = same training data, same alignment method, same architecture. Only variable is parameter count.

**Expected discussion**: ELEPHANT found no consistent pattern between size and social sycophancy. Students verify this finding.

**Execution**: Load one model at a time, run full test suite, unload, load next.

---

### Part 2: Open Model Family Comparison (Local, ~3-4B Band)

**RQ**: Do different model families differ in sycophancy at comparable sizes?

| Model | Size | VRAM (Q4_K_M) | Family |
|-------|------|---------------|--------|
| Qwen2.5-3B-Instruct | 3B | ~2.5 GB | Alibaba/Qwen |
| Llama-3.2-3B-Instruct | 3B | ~2.5 GB | Meta/Llama |
| Gemma-3-4B-IT | 4B | ~3 GB | Google/Gemma |

**Size control**: 3-4B band. Not perfectly identical but reasonable — the size mismatch is explicitly noted as a limitation.

**Discussion points**:

- Different training data, different alignment methods (RLHF vs DPO vs ...), different design philosophies
- Which variables might explain differences?
- Confounds: tokenizer differences, training data language distribution

---

### Part 3: Version Comparison (OpenRouter API — Gemma Generations)

**RQ**: Does sycophancy improve across model generations within the same family?

| Model | Generation | Params | Architecture | Release |
|-------|-----------|--------|-------------|---------|
| Gemma 3 27B-IT | 3rd gen | 27B | Dense | 2025.03 |
| Gemma 4 31B-IT | 4th gen | 31B | Dense | 2026.04 |

**API**: OpenRouter

**Provider specification**: Novita AI / BF16

> **Important note in notebook**: OpenRouter routes requests to multiple providers by default.
> Different providers may serve models at different quantization levels (e.g., 4-bit vs BF16),
> which can significantly affect model behavior and invalidate comparisons.
>
> For fair experimental comparison, we pin the provider to **Novita AI (BF16)** for both models
> by setting the `provider` parameter in API requests:
>
> ```python
> extra_body={"provider": {"order": ["Novita"]}}
> ```
>
> **Gemma 2 27B**: Excluded from this OpenRouter comparison because it is only available in
> 4-bit quantization on OpenRouter, making it an unfair comparison against BF16 models.
>
> **Recommended extension**: For a complete 3-generation comparison (Gemma 2 → 3 → 4),
> use Google AI Studio's free API (https://aistudio.google.com/) which serves all Gemma models
> at native precision with no quantization differences. This is the recommended approach for
> publication-quality results.

**Paper reference**: Parallels Sharma et al.'s Claude 1.3 → Claude 2 comparison methodology.

---

### Part 4: Frontier API Model Comparison

**RQ**: Are current frontier models still sycophantic? Which is most robust?

| Model | API | Model ID | Provider |
|-------|-----|----------|----------|
| GPT-5.4 | OpenAI API (native) | `gpt-5.4` | OpenAI |
| Claude Sonnet 4.5 | Anthropic API (native) | `claude-sonnet-4-5-20250929` | Anthropic |
| Grok 4.20 | OpenRouter | `x-ai/grok-4.20` | xAI |
| Gemini 3 Pro Preview | OpenRouter | `google/gemini-3-pro-preview` | Google |

**Why mixed APIs**: Students learn 3 API patterns (native OpenAI, native Anthropic, OpenRouter). OpenAI and Anthropic use native APIs for maximum reproducibility; Grok and Gemini are only available through aggregators.

**Paper reference**: Mirrors SycEval (ChatGPT vs Claude vs Gemini) and Anthropic-OpenAI joint evaluation (2025).

**Controls**: temperature=0, no system prompt (or identical minimal system prompt), same max_tokens, identical test prompts.

---

### Part 5: Cross-Comparison Dashboard

- Combined results table: all models from Parts 1-4
- Heatmap: model × metric
- Radar chart: per-model sycophancy profile
- Paper-style reporting: Table + Figure with confidence intervals
- Key finding: local small models vs frontier API models

---

### Part 6: Discussion & Limitations

**Findings to discuss**:

- Size vs sycophancy relationship (or lack thereof)
- Family-level differences and possible explanations
- Generational improvement (or lack thereof)
- The RLHF paradox: human preference training may reinforce sycophancy
- Local small models vs frontier: does bigger/more-aligned mean less sycophantic?

**Limitations** (critical for research credibility):

- Size control imperfect in Part 2 (3B vs 4B) and Part 3 (27B vs 31B)
- OpenRouter provider quantization — mitigated by pinning Novita/BF16, but not identical to native serving
- Sample size: 30-50 per condition (educational) vs 500-4000 (publication-grade)
- Model version drift: API models update silently; timestamps recorded but snapshots not guaranteed
- Single evaluation run: no variance measurement (would need multiple runs for statistical rigor)
- Rebuttal type: only "simple" rebuttals tested; SycEval shows citation-based rebuttals change behavior
- English-only prompts (most papers share this limitation)

---

### References

| Paper | Key Contribution | Use in Notebook |
|-------|-----------------|-----------------|
| Sharma et al. (2024) "Towards Understanding Sycophancy in LMs" — ICLR 2024 | Original sycophancy taxonomy, "Are you sure?" protocol, feedback sycophancy | Part 0 methodology, Part 1 baseline |
| SycEval (Fanous et al., 2025) — AAAI AIES 2025 | Progressive/Regressive distinction, rebuttal types, AMPS+MedQuad | Part 0 metrics, Part 4 methodology |
| ELEPHANT (ICLR 2026) | Social sycophancy, moral sycophancy, AITA dataset, 11-model comparison | Part 0 theory, Part 6 discussion |
| Multi-turn Sycophancy (EMNLP 2025) | Number-of-Flip metric, multi-turn debate scenarios | Part 0 metrics |
| Syco-Bench (syco-bench.com) | 4-part benchmark: Picking Sides, Mirroring, Attribution Bias, Delusion Acceptance | Further reading |
| Anthropic-OpenAI Joint Evaluation (2025) | Cross-company sycophancy comparison, delusional belief validation | Part 4 context, Part 6 discussion |
| Sycophancy Is Not One Thing (2025) | Mechanistic separation of sycophancy subtypes in activation space | Part 6 advanced discussion |

---

## Model Summary Table

### Local Models (llama-cpp-python, 8GB VRAM JupyterHub)

| Model | Part | Size | Quant | VRAM |
|-------|------|------|-------|------|
| Qwen2.5-3B-Instruct | Demo | 3B | Q4_K_M | ~2.5 GB |
| Qwen2.5-1.5B-Instruct | Int. Part 1 | 1.5B | Q4_K_M | ~1.5 GB |
| Qwen2.5-3B-Instruct | Int. Part 1,2 | 3B | Q4_K_M | ~2.5 GB |
| Qwen2.5-7B-Instruct | Int. Part 1 | 7B | Q4_K_M | ~5 GB |
| Llama-3.2-3B-Instruct | Int. Part 2 | 3B | Q4_K_M | ~2.5 GB |
| Gemma-3-4B-IT | Int. Part 2 | 4B | Q4_K_M | ~3 GB |

### API Models

| Model | Part | API | Model ID |
|-------|------|-----|----------|
| Gemma 3 27B-IT | Int. Part 3 | OpenRouter (Novita/BF16) | `google/gemma-3-27b-it` |
| Gemma 4 31B-IT | Int. Part 3 | OpenRouter (Novita/BF16) | `google/gemma-4-31b-it` |
| GPT-5.4 | Int. Part 4 | OpenAI (native) | `gpt-5.4` |
| Claude Sonnet 4.5 | Int. Part 4 | Anthropic (native) | `claude-sonnet-4-5-20250929` |
| Grok 4.20 | Int. Part 4 | OpenRouter | `x-ai/grok-4.20` |
| Gemini 3 Pro Preview | Int. Part 4 | OpenRouter | `google/gemini-3-pro-preview` |

---
