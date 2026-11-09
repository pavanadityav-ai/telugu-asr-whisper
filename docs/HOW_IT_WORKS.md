# How this project works (plain English)

This guide explains every step in simple words, so I can explain the project clearly in an interview or a statement of purpose.

---

## 1. The task

**Automatic speech recognition (ASR):** audio goes in, text comes out.
Here: Telugu speech in → Telugu text out.

## 2. The data: Google FLEURS (Telugu, `te_in`)

- About 3,000 short clips of people reading Telugu sentences, each with the correct text.
- Only **7.9 hours** of training audio, which is tiny compared with the 680,000 hours Whisper was first trained on.
- Split into **train** (2,302), **validation** (311) and **test** (472).
  - **Train:** the model learns from these.
  - **Validation:** used during training to choose the best version of the model.
  - **Test:** used only once, at the very end, for the final score.
- **Golden rule:** never train on the test set. If you do, the score is fake, because the model has already seen the answers.

## 3. The model: Whisper

Whisper is a **transformer** model from OpenAI, trained on 680,000 hours of audio from the internet. It has two halves:

| Part | Job | Analogy |
|---|---|---|
| **Encoder** | Listens to the audio and builds an understanding of it | The ears |
| **Decoder** | Writes the text one token at a time, using what the encoder heard | The hand that writes |

**What the encoder actually reads:** a **log-Mel spectrogram**, which is a picture of sound. It has 80 rows (frequency bands, low to high pitch) and 3,000 columns (time, 30 seconds). Bright spots show which sounds are present at each moment.

**Why 30 seconds and 448 tokens?** Whisper was built that way. It reads 30 s of audio at most, and it can write at most 448 tokens. So we remove clips that are longer than that.

**Why `whisper-small`?** It has 244 million parameters. That is big enough to learn, but small enough to train on a free Colab T4 GPU.

## 4. Tokens: why Telugu is harder

Models don't read letters; they read **tokens** (pieces of words). Whisper's tokenizer was built mostly from English, so:

- English: one common word ≈ 1 token.
- Telugu: about **2 tokens per letter**. In notebook 01, the same sentence needed 13 tokens in English but 112 in Telugu.

So a Telugu sentence is a much longer sequence for the decoder to write, which means more chances to make a mistake.

## 5. Fine-tuning

Whisper already "knows" speech in general. **Fine-tuning** means continuing its training on our Telugu clips, so it adapts to Telugu sounds, spelling and vocabulary.

How one training step works:
1. Take a batch of 16 clips.
2. The model predicts the text.
3. The **loss** measures how wrong the prediction is (cross-entropy: how much probability it gave to the correct next token).
4. **Backpropagation** works out how each parameter should change to reduce the loss.
5. The optimiser (AdamW) changes the parameters a tiny amount, set by the **learning rate**.

Settings I used, and why:

| Setting | Value | Why |
|---|---|---|
| Learning rate | 1e-5 | Small, so we adjust the model gently instead of destroying what it already knows |
| Warm-up | 150 steps | The learning rate starts near zero and rises slowly, for stable early training |
| Batch size | 16 | Fits in a T4 GPU's memory |
| Steps | 1,500 | About 10 passes over the training data |
| fp16 | on | 16-bit numbers: faster and uses half the memory |
| Gradient checkpointing | on | Saves memory by recomputing some values instead of storing them |
| Best checkpoint | lowest validation CER | Protects against **overfitting** |

**Overfitting:** the model memorises the training clips instead of learning general patterns. Signs: training loss keeps falling, but validation loss starts rising. That is why we keep the checkpoint with the best *validation* score, not the last one.

## 6. Measuring errors: WER and CER

Both come from **edit distance**: the smallest number of changes to turn the model's output into the correct text.

- **S** substitution: a wrong word (or character)
- **D** deletion: a missing one
- **I** insertion: an extra one

```
error rate = (S + D + I) / number of words (or characters) in the correct text
```

**Example (WER):**
Correct: `నేను ఇంటికి వెళ్ళాను` (3 words)
Model: `నేను ఇంటికి వచ్చాను`
One substitution → WER = 1/3 = **33%**.

**Why CER too?** In the example above, the model got most letters right, but WER counts the whole word as wrong. Telugu words are long, so **CER** (per character) shows how close the model really was.

**Why can WER be above 100%?** If the model writes many extra words (insertions), the errors can be more than the number of correct words.

**How it is computed:** with **dynamic programming**. A table stores the edit distance between every prefix of the two sentences; each cell takes the cheapest of substitute, delete or insert. Then we walk back through the table to recover *which* operations happened. I reuse that walk-back to find which Telugu characters get confused.

## 7. Text normalisation (an important detail)

Before scoring, both the correct text and the model output are cleaned the same way: punctuation removed, spaces tidied.

**The catch:** Whisper's built-in normaliser deletes every Unicode "mark" character. In Telugu, vowel signs such as ా, ి, ు are marks. So it would turn `నాయకత్వం` into broken letters and make every score wrong. So this project uses a custom normaliser that keeps all Telugu letters and vowel signs. This is a real problem for Indian languages, and checking for it is part of doing the evaluation properly.

## 8. Error analysis

One number (WER) doesn't tell you *why* the model fails. Notebook 04 looks deeper:

- **Character confusions:** for example ి vs ీ (short vs long "i"), క vs ఖ (ka vs kha), త vs ట (dental vs retroflex t). These are sounds that are close in speech.
- **Seen vs unseen words:** is the model worse on words it never saw in training?
- **Length:** are longer clips harder?
- **Male vs female voices:** 72% of the training clips are female voices. Is the model worse on male voices?
- **Native-speaker labels:** I read the 40 worst sentences and labelled the main cause of each error (similar sounds, foreign names, English words, numbers, missing words, and so on). Only a Telugu speaker can do this well.

---

## Interview questions I should be able to answer

**1. What did you build?**
I fine-tuned OpenAI's Whisper-small on the Google FLEURS Telugu dataset, compared it with the original model using WER and CER, and analysed the remaining errors as a native speaker.

**2. Why Whisper and not a model trained from scratch?**
With only about 3,000 clips, training from scratch would fail. Whisper already learned general speech from 680,000 hours, so fine-tuning only needs to adapt it to Telugu. This is **transfer learning**.

**3. What is the difference between validation and test sets?**
Validation is used during training to pick the best checkpoint. Test is used once at the end. If I chose the checkpoint using the test set, the final score would be too optimistic.

**4. Why both WER and CER?**
Telugu words are long and have many forms, so one wrong vowel sign makes a whole word wrong. CER shows partial correctness.

**5. What was the hardest technical problem?**
The text normaliser: Whisper's default one deletes Telugu vowel signs, which silently breaks the evaluation. The project uses a custom normaliser, with unit tests that check vowel signs are kept.

**6. How did you avoid overfitting?**
A small learning rate with warm-up, checking the validation set every 500 steps, and keeping the checkpoint with the lowest validation CER.

**7. What would you do next with more time?**
Add more data (for example, IndicVoices or Common Voice Telugu), try `whisper-medium`, try parameter-efficient fine-tuning (LoRA) to train bigger models on small GPUs, and add a language model to fix spelling.

**8. How did you use AI tools?**
I used Claude as a coding assistant to write and debug code faster. I made sure I understood every part, ran all experiments myself, checked the metrics against an independent library, and did the error analysis myself.
