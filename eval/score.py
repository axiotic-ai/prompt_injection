"""Score the shipped checkpoint on a jsonl eval set, broken down by label x form.

Usage:
    python eval/score.py eval/confound_controlled.jsonl
    python eval/score.py eval/natural_heldout.jsonl

Runs on CPU. The encoder produces NaN under accumulated MPS state in long-lived
sessions; CPU is numerically stable and this model is tiny, so CPU is the safe
default for evaluation.
"""
import json, pathlib, sys
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

DEVICE = "cpu"
CKPT_DIR = pathlib.Path(__file__).parent.parent / "ogma-prompt-injection"


def load_model():
    ckpt = torch.load(CKPT_DIR / "classifier.pt", map_location="cpu", weights_only=False)
    mid, hidden, num_labels = ckpt["model_id"], ckpt["hidden"], ckpt["num_labels"]
    tok_src = CKPT_DIR if (CKPT_DIR / "tokenizer_config.json").exists() else mid
    tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(mid, trust_remote_code=True)

    class OgmaClassifier(nn.Module):
        def __init__(self, encoder, hidden, num_labels, dropout=0.1):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, num_labels)

        def forward(self, input_ids, attention_mask):
            return self.head(self.dropout(self.encoder(input_ids=input_ids, attention_mask=attention_mask)))

    model = OgmaClassifier(encoder, hidden, num_labels)
    model.load_state_dict(ckpt["state_dict"])

    # The encoder's rotary cos/sin caches are register_buffer(persistent=False),
    # so from_pretrained's fast-init leaves them as uninitialised memory (NaN or
    # zeros, varying per process). Rebuild them from the saved inv_freq so
    # inference is deterministic. See embeddings.py RotaryPositionalEncoding.
    for mod in model.modules():
        if hasattr(mod, "_build_cache") and hasattr(mod, "cos_cached"):
            mod._build_cache(mod.cos_cached.shape[0])

    return model.to(DEVICE).eval(), tokenizer, ckpt["max_len"]


@torch.no_grad()
def predict(model, tokenizer, max_len, texts, batch=32):
    preds = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        logits = model(enc["input_ids"], enc["attention_mask"])
        preds.extend(logits.argmax(-1).cpu().tolist())
    return preds


def main(path):
    rows = [json.loads(l) for l in open(path)]
    model, tokenizer, max_len = load_model()
    preds = predict(model, tokenizer, max_len, [r["text"] for r in rows])
    y = [r["label"] for r in rows]

    overall_acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    print(f"\n=== {path}  (n={len(rows)}) ===")
    print(f"overall accuracy : {overall_acc:.4f}")
    print(f"macro-F1         : {macro_f1:.4f}")

    if any("form" in r for r in rows):
        print("\nper label x form cell (accuracy = fraction classified correctly):")
        for lab in (0, 1):
            for f in ("question", "statement"):
                idx = [i for i, r in enumerate(rows) if r["label"] == lab and r.get("form") == f]
                if not idx:
                    continue
                correct = sum(1 for i in idx if preds[i] == y[i])
                nm = "benign" if lab == 0 else "malicious"
                aligned = "(shortcut-aligned)" if (f == "question") == (lab == 0) else "(shortcut-MISALIGNED)"
                print(f"  {nm:9s} {f:9s} n={len(idx):>4}  acc={correct/len(idx):.3f}  {aligned}")

        # balanced accuracy: mean of per-class recall, immune to cell-size skew
        rec0 = accuracy_score([1]*0 + [yy for yy, pp in zip(y, preds) if yy == 0],
                              [pp for yy, pp in zip(y, preds) if yy == 0]) if any(v == 0 for v in y) else 0
        rec1 = accuracy_score([yy for yy, pp in zip(y, preds) if yy == 1],
                              [pp for yy, pp in zip(y, preds) if yy == 1]) if any(v == 1 for v in y) else 0
        print(f"\nbalanced accuracy: {(rec0 + rec1) / 2:.4f}  (benign recall {rec0:.3f}, malicious recall {rec1:.3f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path(__file__).parent / "confound_controlled.jsonl"))
