"""Headless training: fine-tune axiotic/ogma-base on the union of
neuralchemy/Prompt-injection-dataset and deepset/prompt-injections,
with cross-dataset dedup, best-val checkpointing, and early stopping."""
import argparse, json, os, pathlib, platform, random, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset, concatenate_datasets, Value
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_ID = "axiotic/ogma-base"
DATASETS = [
    ("neuralchemy/Prompt-injection-dataset", "full"),
    ("deepset/prompt-injections", None),
]
MAX_LEN = 512
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = int(os.environ.get("EPOCHS", "5"))
PATIENCE = int(os.environ.get("PATIENCE", "2"))
LR_HEAD = 5e-4
LR_ENCODER = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
NUM_LABELS = 2
FREEZE_ENCODER = False
SEED = 42
SAVE_DIR = "./ogma-prompt-injection"


def pick_device():
    forced = os.environ.get("FORCE_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OgmaClassifier(nn.Module):
    """Linear classification head on top of ogma's L2-normalised (B, d_output) embedding."""

    def __init__(self, encoder, hidden, num_labels, freeze_encoder=False, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, num_labels)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, input_ids, attention_mask, labels=None):
        emb = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.head(self.dropout(emb))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return loss, logits


def load_and_combine():
    """Load every DATASETS entry, concat per split, dedupe cross-dataset.

    Dedup priority: test > val > train. A text that appears in test is
    removed from train and val; a text in val is removed from train.
    Returns dict {split: Dataset} with columns text/label/source.
    """
    per_split = {"train": [], "validation": [], "test": []}
    for ds_id, config in DATASETS:
        loaded = load_dataset(ds_id, config) if config else load_dataset(ds_id)
        print(f"[load] {ds_id}{('/' + config) if config else ''}: "
              f"{ {k: len(v) for k, v in loaded.items()} }", flush=True)
        for split_name in per_split:
            if split_name not in loaded:
                continue
            d = loaded[split_name].select_columns(["text", "label"])
            d = d.cast_column("label", Value("int64"))  # align dtypes across datasets
            d = d.add_column("source", [ds_id] * len(d))
            per_split[split_name].append(d)

    combined = {}
    for split_name, parts in per_split.items():
        combined[split_name] = concatenate_datasets(parts) if parts else None

    def _texts(ds):
        return {t.strip() for t in ds["text"]} if ds is not None else set()

    test_texts = _texts(combined["test"])
    before_val = len(combined["validation"]) if combined["validation"] else 0
    if combined["validation"] is not None:
        combined["validation"] = combined["validation"].filter(
            lambda x: x["text"].strip() not in test_texts
        )
    val_texts = _texts(combined["validation"])
    excluded = test_texts | val_texts
    before_train = len(combined["train"])
    combined["train"] = combined["train"].filter(lambda x: x["text"].strip() not in excluded)

    print(f"[dedup] val: {before_val} -> {len(combined['validation']) if combined['validation'] else 0}", flush=True)
    print(f"[dedup] train: {before_train} -> {len(combined['train'])}", flush=True)
    print(f"[dedup] test: {len(combined['test'])} (kept whole)", flush=True)
    return combined


def make_collate(tokenizer):
    pad_id = tokenizer.pad_token_id or 0

    def collate(batch):
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attn, labels = [], [], []
        for x in batch:
            ids = x["input_ids"].tolist()
            a = x["attention_mask"].tolist()
            pad = max_len - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append(a + [0] * pad)
            labels.append(int(x["labels"]))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def tokenize_factory(tokenizer):
    def tokenize(batch):
        enc = tokenizer([t or "" for t in batch["text"]], truncation=True, max_length=MAX_LEN, padding=False)
        enc["labels"] = batch["label"]
        return enc

    return tokenize


def main(smoke: bool):
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    cache = pathlib.Path.cwd() / ".hf-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))

    device = pick_device()
    print(f"[setup] device={device} torch={torch.__version__} epochs={EPOCHS} patience={PATIENCE} smoke={smoke}", flush=True)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    print(f"[load] tokenizer {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"[load] encoder {MODEL_ID}", flush=True)
    encoder = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)

    with torch.no_grad():
        probe = tokenizer("hello", return_tensors="pt", truncation=True, max_length=MAX_LEN)
        out = encoder(**probe)
        hidden = out.shape[-1]
    print(f"[load] hidden={hidden}", flush=True)

    combined = load_and_combine()

    # Keep `source` on the dataset so we can break test results down per-source.
    keep_in_loader = ["input_ids", "attention_mask", "labels"]
    tokenized = {}
    for split, d in combined.items():
        if d is None:
            tokenized[split] = None
            continue
        t = d.map(tokenize_factory(tokenizer), batched=True,
                  remove_columns=[c for c in d.column_names if c != "source"])
        t.set_format(type="torch", columns=keep_in_loader, output_all_columns=False)
        tokenized[split] = t

    if smoke:
        for k in tokenized:
            if tokenized[k] is not None:
                tokenized[k] = tokenized[k].select(range(min(8, len(tokenized[k]))))

    collate = make_collate(tokenizer)
    train_loader = DataLoader(tokenized["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(tokenized["validation"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(tokenized["test"], batch_size=EVAL_BATCH_SIZE, shuffle=False, collate_fn=collate)
    print(f"[ready] batches train={len(train_loader)} val={len(val_loader)} test={len(test_loader)}", flush=True)

    model = OgmaClassifier(encoder, hidden, NUM_LABELS, freeze_encoder=FREEZE_ENCODER).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[ready] trainable params={trainable:,}", flush=True)

    encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    groups = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": LR_ENCODER, "weight_decay": WEIGHT_DECAY})
    groups.append({"params": head_params, "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY})
    optim = AdamW(groups)
    epochs = 1 if smoke else EPOCHS
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optim, int(total_steps * WARMUP_RATIO), total_steps)

    @torch.no_grad()
    def evaluate(loader, name):
        model.eval()
        preds, trues = [], []
        loss_sum, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_loss, logits = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
            bs = batch["labels"].size(0)
            loss_sum += float(batch_loss) * bs
            n += bs
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            trues.extend(batch["labels"].cpu().tolist())
        avg_loss = loss_sum / max(1, n)
        acc = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average="macro")
        print(f"[eval] {name} loss={avg_loss:.4f} acc={acc:.4f} macro-f1={f1:.4f}", flush=True)
        return preds, trues, avg_loss, acc, f1

    t0 = time.time()
    best_f1, best_epoch, best_state = -1.0, -1, None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss, _ = model(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); scheduler.step(); optim.zero_grad()
            running += loss.item()
            if step % 25 == 0 or step == len(train_loader):
                print(f"[train] e{epoch} {step}/{len(train_loader)} loss={running/step:.4f} elapsed={time.time()-t0:.1f}s", flush=True)
        _, _, val_loss, val_acc, val_f1 = evaluate(val_loader, f"val e{epoch}")
        history.append({"epoch": epoch, "train_loss": running / max(1, len(train_loader)),
                        "val_loss": val_loss, "val_acc": val_acc, "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_epoch = val_f1, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print(f"[best] new best val macro-f1={val_f1:.4f} at epoch {epoch}", flush=True)
        else:
            no_improve += 1
            print(f"[best] no improvement ({no_improve}/{PATIENCE})", flush=True)
            if not smoke and no_improve >= PATIENCE:
                print(f"[early-stop] stopping after epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[restore] loaded best weights from epoch {best_epoch} (val macro-f1={best_f1:.4f})", flush=True)

    # Final test pass on the combined set + per-source breakdown.
    preds, trues, test_loss, test_acc, test_f1 = evaluate(test_loader, "test (combined)")
    print(classification_report(trues, preds, target_names=["benign", "malicious"], digits=4), flush=True)

    per_source = {}
    sources = combined["test"]["source"]  # parallel to test_loader since shuffle=False
    if len(sources) == len(preds):
        for src in sorted(set(sources)):
            idx = [i for i, s in enumerate(sources) if s == src]
            p = [preds[i] for i in idx]
            t = [trues[i] for i in idx]
            acc = accuracy_score(t, p)
            f1 = f1_score(t, p, average="macro")
            per_source[src] = {"n": len(idx), "acc": acc, "macro_f1": f1}
            print(f"[test] {src} n={len(idx)} acc={acc:.4f} macro-f1={f1:.4f}", flush=True)
    else:
        print(f"[warn] source/pred length mismatch ({len(sources)} vs {len(preds)}); skipping per-source breakdown", flush=True)

    if not smoke:
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "hidden": hidden,
            "num_labels": NUM_LABELS,
            "model_id": MODEL_ID,
            "max_len": MAX_LEN,
        }, os.path.join(SAVE_DIR, "classifier.pt"))
        tokenizer.save_pretrained(SAVE_DIR)
        metadata = {
            "base_model": MODEL_ID,
            "datasets": [f"{d}{('/' + c) if c else ''}" for d, c in DATASETS],
            "splits_after_dedup": {
                "train": len(combined["train"]),
                "validation": len(combined["validation"]) if combined["validation"] else 0,
                "test": len(combined["test"]),
            },
            "hyperparameters": {
                "max_len": MAX_LEN, "batch_size": BATCH_SIZE, "epochs_run": len(history),
                "epochs_configured": EPOCHS, "patience": PATIENCE,
                "lr_head": LR_HEAD, "lr_encoder": LR_ENCODER,
                "weight_decay": WEIGHT_DECAY, "warmup_ratio": WARMUP_RATIO,
                "freeze_encoder": FREEZE_ENCODER, "seed": SEED,
            },
            "best": {"epoch": best_epoch, "val_macro_f1": best_f1},
            "test_combined": {"loss": test_loss, "acc": test_acc, "macro_f1": test_f1},
            "test_per_source": per_source,
            "history": history,
        }
        with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"[save] {SAVE_DIR} (classifier.pt + tokenizer + metadata.json)", flush=True)

    print(f"[done] total={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny single-epoch run for verification")
    args = ap.parse_args()
    main(smoke=args.smoke)
