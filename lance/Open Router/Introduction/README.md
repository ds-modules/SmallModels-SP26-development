# Notebook 1 — Introduction to OpenRouter: Calling AI Models with OLMo 3

## What this notebook is

This is the first notebook in a series on working with language models through OpenRouter.
It introduces the core tools and concepts students will use in every subsequent notebook:
connecting to OpenRouter, calling a model, using system messages, and understanding model IDs.

The anchor model is **OLMo 3** from AllenAI — a fully open-weight model whose training code,
data, and weights are all public.

---

## Prerequisites

### Python packages

| Package | Purpose |
|---|---|
| `openai` | API client (works with OpenRouter via `base_url`) |
| `python-dotenv` | Loads the API key from a `.env` file |

Both packages are installed automatically if missing when you run the imports cell.

### API key

You need a free **OpenRouter account** and an API key.

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Add a small amount of credit (a few dollars covers a full course)
3. Generate an API key in your account settings
4. Create a `.env` file with one line:

```
OPENROUTER_API_KEY="sk-or-v1-..."
```

The notebook loads this file with `load_dotenv()`.
The default path is `/home/jovyan/shared/.env` (shared course environment).
Change the path in the `load_dotenv(...)` call if you are running locally.

**Never paste your API key directly into the notebook.**

---

## What the notebook covers

| Section | Concept |
|---|---|
| What is OpenRouter? | Single API gateway to hundreds of models |
| OLMo 3 and AllenAI | Open-weight models and why openness matters |
| Imports and setup | `os`, `dotenv`, `OpenAI` client with `base_url` |
| API key loading | Secure `.env` pattern |
| Model IDs | `provider/model-name` format; switching models in one line |
| System messages | How to set a model's role before a conversation |
| Experiment 1 | Same question, three different system message roles |
| Trying other models | Swapping model IDs with no other code changes |
| Benchmarks | MMLU, ARC, GSM8K, HellaSwag, TruthfulQA explained |
| Benchmark tests | Commonsense, math, knowledge, truthfulness — 7B vs. 32B |
| Multi-turn conversations | Building and sending a growing message history |

---

## How to run it

1. Open `OpenRouter_Introduction.ipynb` in JupyterLab or Jupyter Notebook
2. Confirm your `.env` file path is correct in the `load_dotenv(...)` cell
3. Run cells top to bottom — each cell depends on the ones before it
4. The API key cell will print `Ready` if the key loaded correctly

**Estimated API cost:** approximately $0.05–$0.15 for a full run, depending on model responses.
The 32B model costs more per token than the 7B model.

---

## Models used

| Model ID | Provider | Size |
|---|---|---|
| `allenai/olmo-3-7b-instruct` | AllenAI | 7B parameters |
| `allenai/olmo-3-32b-instruct` | AllenAI | 32B parameters |
| `meta-llama/llama-3.2-3b-instruct` | Meta | 3B parameters (optional swap cell) |

> Model IDs on OpenRouter can change as new versions are released.
> Check [openrouter.ai/models](https://openrouter.ai/models) if a model ID stops working.

---

## Learning outcomes

By the end of this notebook, students will be able to:

- Explain what OpenRouter is and why it is useful for teaching and experimentation
- Call a language model through the OpenRouter API using the OpenAI Python SDK
- Distinguish open-weight models (like OLMo) from closed models (like GPT or Claude)
- Use system messages to shape a model's behavior
- Understand model IDs and swap between models with a single code change
- Build and send a multi-turn conversation

---

## Notebook 2

This notebook is followed by **OpenRouter_ModelComparison.ipynb** (in the `Comparision/` folder),
which uses the same technical foundation to compare multiple different models on the same prompt.
