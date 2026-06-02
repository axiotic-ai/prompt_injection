"""Headless training script — mirrors finetune_ogma.ipynb."""
import argparse, os, pathlib, platform, random, sys, time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_ID = "axiotic/ogma-base"
DATASET_ID = "neuralchemy/Prompt-injection-dataset"
DATASET_CONFIG = "full"
MAX_LEN = 512
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 64
EPOCHS = 3
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
    """Linear classification head on top of ogma's L2-normalised (B, d_output) embedding.

    The encoder already prepends the SYM task token, pools, and L2-normalises —
    so we feed input_ids directly and treat its output as the sentence embedding.
    """

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
        # No "[SYM] " prefix — the model prepends the task token internally.
        enc = tokenizer([t or "" for t in batch["text"]], truncation=True, max_length=MAX_LEN, padding=False)
        enc["labels"] = batch["label"]
        return enc

    return tokenize


def main(smoke: bool):
    # Env tweaks before heavy imports already happened, set what is still useful.
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    cache = pathlib.Path.cwd() / ".hf-cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))

    device = pick_device()
    print(f"[setup] device={device} torch={torch.__version__} smoke={smoke}", flush=True)

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
    print(f"[load] hidden={hidden} out_shape={tuple(out.shape)}", flush=True)

    print(f"[load] dataset {DATASET_ID}/{DATASET_CONFIG}", flush=True)
    ds = load_dataset(DATASET_ID, DATASET_CONFIG)
    print(f"[load] splits={ {k: len(v) for k, v in ds.items()} }", flush=True)

    keep = ["input_ids", "attention_mask", "labels"]
    tokenized = {}
    for split in ds:
        t = ds[split].map(tokenize_factory(tokenizer), batched=True, remove_columns=ds[split].column_names)
        t.set_format(type="torch", columns=keep)
        tokenized[split] = t

    if smoke:
        # Trim to a tiny slice — one forward + backward.
        tokenized["train"] = tokenized["train"].select(range(min(8, len(tokenized["train"]))))
        tokenized["validation"] = tokenized["validation"].select(range(min(8, len(tokenized["validation"]))))
        tokenized["test"] = tokenized["test"].select(range(min(8, len(tokenized["test"]))))

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
        return preds, trues

    t0 = time.time()
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
        evaluate(val_loader, f"val e{epoch}")

    preds, trues = evaluate(test_loader, "test")
    print(classification_report(trues, preds, target_names=["benign", "malicious"], digits=4), flush=True)

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
        print(f"[save] {SAVE_DIR}", flush=True)

    print(f"[done] total={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny single-epoch run for verification")
    args = ap.parse_args()
    main(smoke=args.smoke)
