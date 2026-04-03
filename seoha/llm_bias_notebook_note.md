# LLM Bias Notebook Notes

## Changes

- **Dataset loading:** Updated the dataset loading code because some older Hugging Face dataset scripts no longer work with current `datasets` versions to make the notebook run again.

- **Default model:** Changed the default model from `gpt2` to `Qwen/Qwen2.5-0.5B-Instruct` because instruction-tuned models behave more naturally on BBQ-style question-answer tasks and are easier for students to interpret.

- **BBQ scoring:** Improved the BBQ scoring logic so it focuses more directly on the answer choices, and expanded the list of `"unknown"` answers to include forms like `"Can't answer"` and `"Cannot be determined"` to make the BBQ section closer to the benchmark's intended behavior.

- **ToxiGen stability:** Made the ToxiGen section stop crashing when no group metrics could be computed, but the current dataset setup still does not support a strong fairness analysis.

## Need to review

- **ToxiGen summary values:** The notebook still shows `nan%` in the final ToxiGen scorecard when no valid group metrics are available. 

- **StereoSet scoring method:** StereoSet still scores full sequences instead of only the differing completion. This should be improved because the current method can blur the real difference between stereotype and anti-stereotype options.

- **Bonus comparison wording:** The bonus cell still looks like a two-model comparison even though it only scores one model. 
