# llm_values_survey Notes

## Changes

- **Fixed WVS double normalization**
  - The original notebook normalized the WVS human benchmarks twice.
  - This was a real bug because the benchmark values were already on a `0-1` scale.

- **Fixed the WVS similarity plot**
  - The old plot could hide broken negative values by keeping the axis in a clean-looking range.
  - The plot now reflects the actual computed values instead of masking errors.

- **Changed MFQ interpretation**
  - MFQ cosine similarity looked too high even when model profiles were meaningfully different.
  - The notebook now also uses `mean_abs_diff` and tells students to read that first for MFQ.

- **Updated markdown explanations**
  - The metric explanations and discussion prompts were adjusted to match the corrected calculations.

- **Corrected the WVS item count**
  - The intro used to imply a larger WVS subset.
  - It now correctly says the notebook uses **14 selected items**.

## Future Note

- **Model pricing update**
  - A later revision could switch `gpt-4o` to `gpt-5.1` (cheaper) and `Claude Sonnet 4` to `Claude Sonnet 4.5` (same).
  - This is optional because it changes the chosen models, not the correctness of the current notebook.
