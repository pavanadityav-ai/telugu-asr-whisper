# Telugu Speech Recognition with Whisper

> **Status:** in progress. Results will be added as each step is finished.

## Goal

Telugu has over 80 million speakers, but speech recognition for it is much weaker than for English. This project asks:

1. How well does OpenAI's Whisper model recognise Telugu speech **out of the box**?
2. How much does **fine-tuning** on Telugu audio improve it?
3. **What kinds of mistakes** does the model still make, and why?

## Plan

| Step | What I will do | Output |
|---|---|---|
| 1. Data | Load a public Telugu speech dataset (Google FLEURS `te_in`), listen to samples, check clip lengths and text | Data notes in `notebooks/01_data.ipynb` |
| 2. Baseline | Run pre-trained `whisper-small` on the test set, no training | Baseline WER and CER |
| 3. Fine-tune | Fine-tune `whisper-small` on the Telugu training set using a free Colab GPU | Fine-tuned model + training curve |
| 4. Evaluate | Compare baseline vs fine-tuned on the same test set | Results table |
| 5. Error analysis | As a native speaker, group the errors by type: similar-sounding letters, English words mixed into Telugu, word-joining (sandhi), long words, noisy audio | Error categories with counts and examples |
| 6. Write-up | Summarise what worked, what did not, and what I would try next | Final section of this README |

## Metrics

- **WER (Word Error Rate):** share of words the model got wrong. Lower is better.
- **CER (Character Error Rate):** the same, but per character. Useful for Telugu because one long word can hide a single small mistake.

## Results

_Coming soon._

| Model | WER | CER |
|---|---|---|
| whisper-small (no fine-tuning) | – | – |
| whisper-small (fine-tuned) | – | – |

## Planned layout

```
notebooks/   step-by-step Colab notebooks (data, baseline, fine-tuning, evaluation)
src/         reusable Python code (data loading, metrics)
results/     tables, plots and error-analysis files
```

## How to run

```
pip install -r requirements.txt
```

Training needs a GPU. I use Google Colab (free tier).

## References

- Radford et al., 2022. *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper).
- Conneau et al., 2022. *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech.*
