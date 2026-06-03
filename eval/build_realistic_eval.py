"""Build a REALISTIC eval set for the prompt-injection classifier.

The training datasets' benign class is synthetic, templated filler
("Write a Python function to climate change"), so the in-distribution
macro-F1 (~0.96) does not reflect real-world behaviour. On genuine benign
prompts the shipped checkpoint false-positives heavily.

This set draws each class from realistic sources, then balances label x form
so we can read off two numbers that matter:
  - benign recall      -> real-world false-positive rate = 1 - benign recall
  - malicious-question recall -> can it catch injections that are NOT imperatives

Benign:    real instructions from tatsu-lab/alpaca and databricks/databricks-dolly-15k,
           restricted to standalone prompts (no separate input/context field).
Malicious: real injections from the held-out (val+test) splits of the two
           training datasets, plus jailbreak prompts from
           jackhhao/jailbreak-classification for harder, politer attacks.

Form: 'question' if the text ends with '?', else 'statement'.
Deterministic (seed=42). Output: eval/realistic_eval.jsonl
"""
import json, pathlib, random
from datasets import load_dataset

SEED = 42
OUT = pathlib.Path(__file__).parent
INJECTION_SRC = [("neuralchemy/Prompt-injection-dataset", "full"), ("deepset/prompt-injections", None)]


def form(text):
    return "question" if text.strip().endswith("?") else "statement"


def clean(t):
    return " ".join(t.split()).strip()


def benign_pool():
    rows = []
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for r in alpaca:
        if not r["input"].strip() and r["instruction"].strip():
            rows.append(clean(r["instruction"]))
    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for r in dolly:
        if not r["context"].strip() and r["instruction"].strip():
            rows.append(clean(r["instruction"]))
    return rows


def malicious_pool():
    rows = []
    for ds_id, cfg in INJECTION_SRC:
        d = load_dataset(ds_id, cfg) if cfg else load_dataset(ds_id)
        for split in d.keys():
            if split == "train":  # held-out only, mirrors confound_controlled
                continue
            s = d[split]
            for t, l in zip(s["text"], s["label"]):
                if int(l) == 1 and t and t.strip():
                    rows.append(clean(t))
    jb = load_dataset("jackhhao/jailbreak-classification", split="train")
    for r in jb:
        if r["type"] == "jailbreak" and r["prompt"].strip():
            rows.append(clean(r["prompt"]))
    return rows


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    benign = list(dict.fromkeys(benign_pool()))   # dedup, keep order
    malicious = list(dict.fromkeys(malicious_pool()))
    rng.shuffle(benign)
    rng.shuffle(malicious)

    cells = {(0, "question"): [], (0, "statement"): [], (1, "question"): [], (1, "statement"): []}
    for t in benign:
        cells[(0, form(t))].append(t)
    for t in malicious:
        cells[(1, form(t))].append(t)

    print("available per cell:")
    for k in sorted(cells):
        lab = "benign" if k[0] == 0 else "malicious"
        print(f"  {lab:9s} {k[1]:9s}: {len(cells[k])}")

    n = min(len(v) for v in cells.values())  # balance to smallest cell
    eval_rows = []
    for k in sorted(cells):
        for t in cells[k][:n]:
            eval_rows.append({"text": t, "label": k[0], "form": k[1],
                              "source": "alpaca+dolly" if k[0] == 0 else "injection+jailbreak"})
    rng.shuffle(eval_rows)
    write_jsonl(OUT / "realistic_eval.jsonl", eval_rows)
    print(f"\nbalanced: {n} per cell x 4 = {len(eval_rows)} rows -> eval/realistic_eval.jsonl")


if __name__ == "__main__":
    main()
