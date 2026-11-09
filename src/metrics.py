"""Word and character error rates, written from scratch.

Both metrics come from the same idea: the *edit distance* between the
correct sentence (reference) and the model's output (hypothesis).
Edit distance = the smallest number of substitutions (S), deletions (D)
and insertions (I) needed to turn one into the other.

    error rate = (S + D + I) / number of units in the reference

For WER the units are words; for CER the units are characters.
The notebooks also compute the scores with the ``jiwer`` library and check
that both give the same answer.
"""

from collections import Counter


def align(ref, hyp):
    """Align two sequences with dynamic programming (Levenshtein distance).

    Returns a list of operations. Each operation is a tuple
    ``(op, ref_item, hyp_item)`` where op is:
      "C" correct, "S" substitution, "D" deletion, "I" insertion.
    """
    n, m = len(ref), len(hyp)
    # cost[i][j] = edit distance between ref[:i] and hyp[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        row, prev = cost[i], cost[i - 1]
        r = ref[i - 1]
        for j in range(1, m + 1):
            sub = prev[j - 1] + (r != hyp[j - 1])
            row[j] = min(sub, prev[j] + 1, row[j - 1] + 1)

    # Walk back from the bottom-right corner to recover the operations.
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            op = "C" if ref[i - 1] == hyp[j - 1] else "S"
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1
    ops.reverse()
    return ops


def error_counts(ref, hyp):
    """Count substitutions, deletions, insertions and correct items."""
    counts = Counter(op for op, _, _ in align(ref, hyp))
    return {k: counts.get(k, 0) for k in ("S", "D", "I", "C")}


def _rate(refs, hyps, split):
    errors, total = 0, 0
    for ref, hyp in zip(refs, hyps):
        r, h = split(ref), split(hyp)
        if not r:
            continue  # an empty reference has nothing to score
        c = error_counts(r, h)
        errors += c["S"] + c["D"] + c["I"]
        total += len(r)
    return errors / total if total else 0.0


def wer(refs, hyps):
    """Corpus-level word error rate (0.25 means 25% of words wrong)."""
    return _rate(refs, hyps, str.split)


def cer(refs, hyps):
    """Corpus-level character error rate (spaces count as characters)."""
    return _rate(refs, hyps, list)


def sentence_scores(ref, hyp):
    """WER and CER for one sentence."""
    return {"wer": wer([ref], [hyp]), "cer": cer([ref], [hyp])}


def char_confusions(refs, hyps):
    """Which characters get swapped, dropped or added most often.

    Returns a Counter with keys like ("S", "క", "ఖ"), ("D", "ు", None)
    or ("I", None, "ం").
    """
    confusions = Counter()
    for ref, hyp in zip(refs, hyps):
        for op, r, h in align(list(ref), list(hyp)):
            if op != "C" and (r != " " and h != " "):
                confusions[(op, r, h)] += 1
    return confusions


def word_hits(refs, hyps):
    """For every reference word, was it recognised correctly?

    Returns a list of (word, correct: bool). Used to compare words the model
    saw during training with words it never saw.
    """
    hits = []
    for ref, hyp in zip(refs, hyps):
        for op, r, _ in align(ref.split(), hyp.split()):
            if op in ("C", "S", "D"):
                hits.append((r, op == "C"))
    return hits
