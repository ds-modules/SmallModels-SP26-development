# Sycophancy Notebooks — Outline

## Overview

Three-notebook tutorial series for a freshman-level course on small language models.
The sequence moves from concept introduction, to benchmark design, to training-pipeline analysis.

- **Demo(CPU)**: Beginner — concept introduction with one local model and two hosted models
- **Intermediate(CPU)**: Benchmarking — research datasets, judge model, cross-model comparison
- **Advanced(GPU 8GB VRAM)**: Mechanisms — open checkpoints, post-training stages, and prompt sweeps

All three notebooks now keep the benchmark families separate:

- **Explicit sycophancy**: factual answer changes under user pressure
- **Conversational conformity**: longer free-form dialogue pressure
- **Social sycophancy**: validation, indirectness, and preference-preserving framing

These notebooks mainly measure the **first** category.

---

## Notebook 1: `sycophancy_demo.ipynb` (Beginner)

**Title**: Does Your AI Actually Agree With You? A Beginner's Guide to Sycophancy Testing

**Target**: Students seeing the idea of sycophancy for the first time

**Teaching goal**: Show that a model can sound confident, helpful, and still change its answer too easily when the user pushes back.

**Note**: This notebook is a **classroom simplification** of Sharma et al. (2024).

**Why Llama in the demo?** Meta's official Instruct models are post-trained assistant models, so they give students a simple, recognizable example for asking whether a helpful assistant can also become too willing to yield to user pressure.

### Model setup

| Role | Model | Serving |
|------|-------|---------|
| Local baseline | Llama 3.2 1B Instruct Q4_K_M | `llama-cpp-python` |
| Hosted comparison | Llama 3.1 8B Instruct | OpenRouter -> DeepInfra BF16 |
| Hosted comparison | Llama 3.1 70B Instruct | OpenRouter -> DeepInfra BF16 |

### Structure

| Section | Content | Key idea |
|---------|---------|----------|
| Intro | What sycophancy is and why it matters | Agreement is not the same as correctness |
| Part 0: Setup | Install packages, load local GGUF, configure OpenRouter | Local vs hosted model workflow |
| Part 1: "Are You Sure?" flip test | Ask factual questions, challenge the answer, measure flips | Simple explicit-sycophancy test |
| Part 2: Feedback sycophancy | Compare feedback when the user claims ownership | Agreement can appear in evaluation language too |
| Part 3: Size comparison | Compare 1B, 8B, and 70B | Bigger does not automatically mean robust |
| Summary | Key takeaways, limitations, next steps | Demo is a starting point, not a full benchmark |

### Limitations

- Small hand-written question set
- Simplified rebuttal wording
- No research dataset loading
- No judge model
- Not designed for publication-grade claims

---

## Notebook 2: `sycophancy_intermediate.ipynb` (Intermediate)

**Title**: Benchmarking Sycophancy: Cross-Model Evaluation with Research Datasets and LLM-as-Judge

**Target**: Students who already understand the basic flip test and are ready to read a benchmark notebook

**Teaching goal**: Show how to turn a classroom demo into a more systematic evaluation pipeline.

### Core questions

- How do we load and use official research datasets?
- What is the difference between **progressive** and **regressive** explicit sycophancy?
- How much do rebuttal type, model family, and model size matter?
- When does keyword matching fail, and when do we need an LLM judge?

### Benchmark scope

The main benchmark in this notebook stays inside **explicit factual sycophancy**. SYCON Bench and ELEPHANT appear as research context for the later classroom extensions, not as direct benchmark reproductions.

### Structure

| Section | Content | Key technique |
|---------|---------|---------------|
| Background | Separate explicit sycophancy, conversational conformity, social sycophancy | Benchmark hygiene |
| Part 0: Setup and methodology | Experimental controls, API setup, judge setup | Reproducible evaluation framing |
| Dataset loading | `meg-tong/sycophancy-eval` + a harder math subset | Research dataset handling |
| Part 1: Official dataset evaluation | Progressive vs regressive classification | Core benchmark loop |
| Part 2: Rebuttal type comparison | Simple vs authority vs citation prompts | SycEval-inspired prompt variant comparison |
| Part 3: Multi-turn pressure test | Repeated scripted rebuttals on one factual question | SYCON-inspired classroom extension |
| Part 4: Social sycophancy mini probe | Short user-validation scenarios | ELEPHANT-inspired classroom extension |
| Part 5: Judge vs keyword | GPT-4o judge vs string matching | Label quality comparison |
| Part 6: Open model family comparison | ~27-32B and ~70B open models | Size-banded model comparison |
| Part 7: Frontier model comparison | OpenAI, Anthropic, OpenRouter models | Multi-provider evaluation |
| Part 8: Mitigation prompt sweep | Baseline vs short anti-sycophancy prompts | Simple solution-oriented study |
| Part 9: Dashboard | Combined table and plots | Paper-style reporting |
| Part 10: Discussion | RLHF paradox, limitations, benchmark boundaries | Research interpretation |

### Model groups

| Group | Models | Serving |
|------|--------|---------|
| Demo model | Llama 3.3 70B | OpenRouter |
| Open ~27-32B | Qwen3 32B, OlMo 3.2 32B, Gemma 3 27B | OpenRouter / DeepInfra FP8 |
| Open ~70B | Qwen2.5 72B, Llama 3.3 70B | OpenRouter (mixed providers) |
| Frontier | GPT-5.4, Claude Sonnet 4.6, Gemini 3.1 Pro, Grok 4.20 | Native APIs + OpenRouter |

### Datasets

| Dataset | Use |
|---------|-----|
| `meg-tong/sycophancy-eval` | Official factual multi-turn prompts |
| Harder math subset from `hendrycks/competition_math` | Progressive/regressive cases |
| Hand-written social scenarios | Small classroom social-sycophancy probe |

The intermediate notebook adds two short classroom extensions after the main explicit-factual benchmark. The **multi-turn pressure test** moves from a single rebuttal to repeated pressure, and the **social mini probe** adds a few short reassurance-seeking scenarios. Both are kept small so students can read them easily, and both should be treated as **benchmark-inspired teaching extensions**, not as full reproductions of SYCON Bench or ELEPHANT. The later **mitigation prompt sweep** then shifts from diagnosis to a practical question: which simple prompt strategies reduce flip behavior most reliably?

### Limitations

- Small sample sizes for teaching
- Mixed providers in some comparisons
- Judge cost and possible judge bias
- OpenRouter routing differences
- The multi-turn and social sections are classroom extensions, not full benchmark reproductions
- The mitigation section tests prompt-level fixes, not training-level solutions
- Educational benchmark, not publication-grade statistics

---

## Notebook 3: `sycophancy_advanced.ipynb` (Advanced)

**Title**: Inside the Training Pipeline: How SFT, DPO, and RLVR Shape Sycophancy

**Target**: Students who finished the intermediate notebook and want to connect observed behavior to post-training stages

**Teaching goal**: Move from “which model is more sycophantic?” to “which training stage may have changed the behavior?”

### Why this notebook is different

Most sycophancy notebooks compare finished products. This notebook compares **open checkpoints** inside one family so students can reason about what changed during post-training. To keep that comparison readable, it stays with **explicit factual sycophancy** plus a small **feedback-sycophancy** probe. SYCON Bench and ELEPHANT remain nearby research context, but they are not run directly here.

### Structure

| Section | Content | Key question |
|---------|---------|--------------|
| Background | Clarify benchmark family and explain the OlMo pipeline | What exactly are we measuring? |
| Part 0: Setup | GGUF paths, VRAM management, local loading | How do we run multiple 7B checkpoints safely? |
| Part 1: Training stage comparison | SFT vs DPO vs RLVR | How does preference optimization change agreement-seeking? |
| Part 2: Version comparison | OlMo 2 vs OlMo 3 | How do changes in data, curriculum, and post-training recipe affect explicit sycophancy? |
| Part 3: Instruct vs Think | Final instruct vs think checkpoint | Does explicit reasoning help resist pressure? |
| Part 4: System prompt sweep | Transfer-style mitigation prompts | Can prompting offset a post-training tendency? |
| Part 5: Discussion | Interpret stage effects and open-model value | Why checkpoint transparency matters |

### Checkpoints

| Model | Stage | Purpose |
|-------|-------|---------|
| OlMo3-SFT | SFT only | Baseline before preference optimization |
| OlMo3-DPO | SFT + DPO | Preference-optimized checkpoint |
| OlMo3-Instruct | SFT + DPO + RLVR | Final instruct checkpoint |
| OlMo2-Instruct | Previous generation final checkpoint | Generation comparison |
| OlMo3-Think | Thinking variant | Instruct vs think comparison |

### Why OlMo 2 vs OlMo 3 is interesting

This is not just a simple version-number comparison. OlMo 3 changes the **pretraining data**, the **mid-training curriculum**, the **post-training datasets**, and the **target capability mix**. So if OlMo 3 behaves differently from OlMo 2 on the sycophancy benchmark, the more useful interpretation is not just "newer is better," but that **different training pipelines inside the same open family** can produce different agreement behavior.

---

## Notebook Sequence

| Notebook | Main question | Best use |
|---------|---------------|----------|
| Demo | What is sycophancy? | First exposure |
| Intermediate | How do we benchmark it systematically? | Methods and comparison |
| Advanced | How might training create or reduce it? | Mechanism and interpretation |

---

## References

| Paper | Main contribution | Where it appears |
|-------|-------------------|------------------|
| Sharma et al. (2024) | Original explicit-sycophancy framing and “Are you sure?” protocol | Demo background, Intermediate methodology |
| SycEval (2025) | Progressive vs regressive distinction, rebuttal effects, judge design | Intermediate core framing |
| SYCON Bench (2025) | Conversational conformity benchmark | Boundary-setting discussion |
| ELEPHANT (2026) | Social sycophancy benchmark | Boundary-setting discussion |
| OlMo technical releases | Open post-training checkpoints | Advanced notebook |

---

## Practical Summary

- Start with **Demo** if students are new to the topic.
- Use **Intermediate** when teaching benchmark design, controls, and judge models.
- Use **Advanced** when teaching how alignment stages may shape behavior.
- Treat the three notebooks as a ladder: **concept -> benchmark -> mechanism**.
