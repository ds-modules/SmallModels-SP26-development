# Planned Notebooks Outline

## Overview

This document outlines three separate teaching notebooks for a freshman-level course on small language models.
Each notebook stands on its own and focuses on a different classroom question.

These notebooks cover three different topics:

- **Output stability**: Does the same model give the same answer every time?
- **Judge reliability**: Can a model judge be pushed by answer order or answer length?
- **Safety behavior**: How does one local model behave across bias, harmful content, misuse, and truthfulness tasks?

---

## Notebook 1: `llm_output_drift_tutorial.ipynb`

**Title**: Output Drift in a Small Local Model

**Target**: Students who are seeing reproducibility and output drift for the first time

**Teaching goal**: Help students learn the main idea of the paper through a smaller classroom notebook. One local model can produce different outputs for the same prompt, and simple repeated-run metrics can reveal that drift without claiming a full paper replication.

**Note**: This notebook is a **classroom learning version** of Khatchadourian and Franco's *LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows* (arXiv:2511.07585), not a full replication study.

### Model setup

| Role | Model | Serving |
|------|-------|---------|
| Local model | Qwen 2.5 3B Instruct Q4_K_M GGUF | `llama-cpp-python` |

### Structure

| Section | Content | Key idea |
|---------|---------|----------|
| Intro | Define output drift, temperature, and seed | Repeated runs can differ even when the prompt stays the same |
| Step 1: Scope | Explain what the notebook can and cannot claim | Classroom demo is weaker than a research benchmark |
| Step 2: Prompt design | Build three simplified finance-style tasks | Different task shapes can drift in different ways |
| Step 3: Repeated runs | Run the same prompt many times at two temperatures | Small randomness can change text output |
| Step 4: Raw inspection | Read real responses before computing metrics | Tables do not replace close reading |
| Step 5: Metrics | Compute exact match, edit distance, and simplified factual drift | Stability can be measured in several ways |
| Step 6: Task comparison | Compare citation, JSON-style, and SQL-style outputs | Structured tasks often drift less |
| Step 7: Practical controls | Test fixed seed and prompt structure | Simple controls may reduce drift |
| Step 8: Prompt format comparison | Compare open and structured prompts | Output format can change stability |
| Summary and exercise | Reflect on what the run supports | Keep conclusions modest |

### References

| Reference | Why it matters here |
|-----------|---------------------|
| Khatchadourian and Franco (2025), *LLM Output Drift: Cross-Provider Validation & Mitigation for Financial Workflows* | This is the main paper behind the notebook. It provides the classroom framing for repeated runs, finance-shaped task types, invariant checks, and careful limits on the claim. |

### Limitations

- One local model only
- Small classroom prompt set
- Simplified finance-style tasks instead of a full benchmark
- Reduced metrics compared with a research paper
- Hardware, software version, and local setup can change the results

---

## Notebook 2: `llm_judge_bias_experiment.ipynb`

**Title**: Can We Trust AI Judges? Exploring Bias in LLM-as-a-Judge

**Target**: Students who already understand basic prompt-and-response evaluation and want to test evaluator bias

**Teaching goal**: Show how answer order and answer length can distort a model judge, and explain why self-enhancement is harder to study cleanly than it first appears.

**Note**: This notebook is a **classroom demo** built from a small hand-written dataset, not a full benchmark study.

### Model setup

| Role | Model | Serving |
|------|-------|---------|
| Judge | Claude Haiku 4.5 | Anthropic API |
| Judge | GPT-5 mini | OpenAI API |

### Structure

| Section | Content | Key idea |
|---------|---------|----------|
| Intro | Define LLM-as-a-judge and notebook roadmap | Judges are models too, so they can have biases |
| Step 1: Setup | Load API clients and experiment settings | Careful setup matters in evaluation work |
| Step 2: Classroom dataset | Build twelve hand-written question and answer pairs | Small readable datasets help students inspect the design |
| Step 3: Judge prompt | Ask both judges for a small JSON verdict | Structured outputs are easier to compare |
| Part 1: Position bias | Swap answer order and compare decisions | A judge may change its choice because of presentation order |
| Part 2: Verbosity bias | Test long wrong answers while controlling for position | Length can look like quality if the design is weak |
| Part 3: Self-enhancement | Run an exploratory family-preference check | Self-bias claims need extra caution |
| Exercise and wrap-up | Extend the dataset or change the prompt | Evaluation design choices shape the result |

### References

| Reference | Why it matters here |
|-----------|---------------------|
| Zheng et al. (2023), *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | Motivates the position-bias section by showing that answer order can change judge decisions. |
| Ye et al. (2025), *Justice or Prejudice? Quantifying Biases in LLM-as-a-Judge* | Frames the broader question of judge bias and supports the idea that these effects can appear across models. |
| Gu et al. (2025), *A Survey on LLM-as-a-Judge* | Provides the background for how LLM judges are used and what evaluation risks students should watch for. |
| Wataoka et al. (2024), *Self-Preference Bias in LLM-as-a-Judge* | Explains why self-enhancement is difficult to measure cleanly and why this notebook treats that section as exploratory. |

### Limitations

- Small hand-written dataset
- Two judges only
- Classroom questions are easier than stress-test benchmarks
- Self-enhancement section is exploratory, not causal
- API model versions may change over time

---

## Notebook 3: `gemma_safety_audit.ipynb`

**Title**: Gemma 3 Safety Audit with `llama-cpp-python`

**Target**: Students who want to build a larger local evaluation notebook and inspect a safety audit step by step

**Teaching goal**: Show how to run one local Gemma GGUF model across several safety-related task families, organize the outputs, and prepare tables for manual review without claiming a full benchmark reproduction.

**Note**: This notebook **borrows ideas** from BBQ, WinoBias, StereoSet, ToxiGen, TruthfulQA, HarmBench, and the Gemma and ShieldGemma model cards, but it keeps all audit sets small and classroom-readable.

### Model setup

| Role | Model | Serving |
|------|-------|---------|
| Local audit model | Gemma 3 4B IT QAT Q4_0 GGUF | `llama-cpp-python` |

### Audit settings

| Setting | What changes | Purpose |
|---------|--------------|---------|
| `plain_chat` | No extra safety instruction | Baseline local behavior |
| `safer_prompt` | Add one safety-focused system prompt | Prompt-level mitigation |
| `simple_filter` | Add the safety prompt plus a small keyword filter | Teaching baseline for basic moderation logic |

### Structure

| Section | Content | Key question |
|---------|---------|--------------|
| Intro | Explain the audit goals and research background | What does a classroom safety audit look like? |
| Setup | Check the GGUF path, load the model, and run one smoke test | Is the local model ready to audit? |
| Conditions | Define three chat settings | How much can prompt-level controls change behavior? |
| Bias audit | Run a small ambiguous-versus-clear bias set | Does the model guess in ambiguous cases? |
| Harmful-content audit | Test refusal and over-refusal on toxic and benign prompts | Can the model block harmful content without blocking too much? |
| Misuse audit | Test harmful instructions and safe requests | Does the model refuse dangerous help? |
| Truthfulness audit | Ask misconception-rich questions and save outputs | Fluent answers are not always true answers |
| Annotation workflow | Export CSV sheets and build paper-style summary tables | Human review is still necessary |
| Reporting | Write short result paragraphs and make a quick plot | Results need readable summaries |
| Wrap-up | Optional cleanup and final reflection | Local evaluation is a process, not one score |

### References

| Reference | Why it matters here |
|-----------|---------------------|
| Gemma 3 model card | Provides the model background and the safety-evaluation context for the local model being audited. |
| BBQ | Inspires the ambiguous-versus-disambiguated bias design, where the model should avoid stereotype-based guessing when evidence is missing. |
| WinoBias | Motivates using small wording changes and matched examples to reveal bias patterns. |
| StereoSet | Adds the idea that bias can appear as a preference for stereotypical continuations, not only as lower accuracy. |
| ToxiGen | Supports the harmful-content section's focus on refusal and over-refusal, especially around identity-related language. |
| TruthfulQA | Motivates the misconception-based truthfulness probe instead of relying only on ordinary factual questions. |
| HarmBench | Shows that misuse evaluation should separate dangerous requests from safe requests and reminds students that a full attack pipeline is much larger than this classroom demo. |
| ShieldGemma model card | Clarifies that the notebook's simple keyword filter is only a teaching baseline, not a real moderation model. |

### Limitations

- One model family only
- Small hand-built audit sets
- Simple keyword filtering is not equivalent to ShieldGemma
- Automatic proxy scores are weaker than manual labels
- Classroom audit borrows benchmark ideas without fully reproducing benchmark difficulty

---

### In-Progress
- 3-4-Google_AI_Studio.ipynb

### Planning 
- Sycophancy Evaluation Notebooks
- LLM Output stability
- LLM Judge reliability
- LLM Safety behavior
- LLM reasoning behavior
  
### Pending (Paused Notebooks -> Personal Summer Projects)
Honestly, I designed these projects to address a gap I experienced myself. 
- Debugging Tutor -> Coding Agent Systems for CS with a focus on teachback, understanding debt, and the real skills students still need in the age of AI (Shift from code writing → code reading & review)
- DATA88E Research Agent (sequential vs parallel, single vs multi agent trade-off) -> Agentic Analysis with a focus on verification, trace observability, and process validation
- KV Cache Reuse, Sharing, and Quantization (Long Context, Agent Workload)
