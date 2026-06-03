"""Build a confound-controlled eval set for the prompt-injection classifier.

The training data confounds label with surface form: benign examples skew
short and interrogative, malicious examples skew long and declarative. A model
can score well on the in-distribution test split by learning that shortcut
instead of detecting injection. This script rebuilds an eval set from the SAME
held-out data, rebalanced so label is independent of form.

Held-out = validation + test from both source datasets. train.py dedups train
against both (train.py:93-102), so these rows are unseen during training.

Form is operationalised as: 'question' if the text ends with '?', else
'statement' — the exact axis measured as confounded with the label.

Outputs (deterministic, seed=42):
  eval/natural_heldout.jsonl      all held-out rows, natural proportions (baseline)
  eval/confound_controlled.jsonl  equal n per (label x form) cell
"""
import json, pathlib, random
from datasets import load_dataset

SEED = 42
SRC = [("neuralchemy/Prompt-injection-dataset", "full"), ("deepset/prompt-injections", None)]
OUT = pathlib.Path(__file__).parent


def form(text):
    return "question" if text.strip().endswith("?") else "statement"


def load_heldout():
    rows, seen = [], set()
    for ds_id, cfg in SRC:
        d = load_dataset(ds_id, cfg) if cfg else load_dataset(ds_id)
        for split in d.keys():
            if split == "train":
                continue
            s = d[split]
            for t, l in zip(s["text"], s["label"]):
                if not (t and t.strip()):
                    continue
                t = t.strip()
                if t in seen:
                    continue
                seen.add(t)
                rows.append({"text": t, "label": int(l), "form": form(t), "source": ds_id})
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rows = load_heldout()
    write_jsonl(OUT / "natural_heldout.jsonl", rows)

    # group into the four label x form cells
    cells = {}
    for r in rows:
        cells.setdefault((r["label"], r["form"]), []).append(r)
    n = min(len(v) for v in cells.values())  # balance to the smallest cell

    rng = random.Random(SEED)
    balanced = []
    for key in sorted(cells):
        sample = rng.sample(cells[key], n)
        balanced.extend(sample)
    rng.shuffle(balanced)
    write_jsonl(OUT / "confound_controlled.jsonl", balanced)

    print(f"held-out pool: {len(rows)} rows")
    for key in sorted(cells):
        lab = "benign" if key[0] == 0 else "malicious"
        print(f"  {lab:9s} {key[1]:9s}: available={len(cells[key]):>4}")
    print(f"balanced: {n} per cell x 4 = {len(balanced)} rows -> eval/confound_controlled.jsonl")


if __name__ == "__main__":
    main()
