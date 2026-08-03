"""Claim 5 (scaled): Mistral-7B fine-tuned on OpenBookQA with the DAPPr loss vs cross-entropy.

Full fine-tuning of Mistral-7B is infeasible on 2x T4-16GB (paper/IB-EDL use larger GPUs);
this is a scaled reproduction: 4-bit QLoRA (rank 16) fine-tune of Mistral-7B-Instruct on
OBQA (4957 train / 500 val / 500 test), per-token DAPPr loss vs CE, reporting accuracy,
NLL and OOD AUROC against ARC-Challenge (as in Table 5). Backbone: HF
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
"""
import json, os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          TrainingArguments, Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import roc_auc_score
import dappr


def build_dataset(tokenizer, max_len=256):
    ds = load_dataset("allenai/openbookqa", "main")

    def fmt(example):
        options = example["choices"]["text"]
        labels = example["choices"]["label"]
        q = example["question_stem"]
        ans = example["answerKey"]
        prompt = f"Question: {q}\nOptions:\n" + "\n".join(
            f"{l}. {t}" for l, t in zip(labels, options)) + f"\nAnswer:"
        return {"prompt": prompt, "answer_key": ans, "answer": ans}

    def tok(example):
        prompt = example["prompt"]
        ans = example["answer"]
        text = prompt + " " + ans
        enc = tokenizer(text, max_length=max_len, truncation=True, padding=False)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    train = ds["train"].map(fmt).map(tok, remove_columns=ds["train"].column_names)
    val = ds["validation"].map(fmt).map(tok, remove_columns=ds["validation"].column_names)
    test = ds["test"].map(fmt).map(tok, remove_columns=ds["test"].column_names)
    return train, val, test


def dappr_data_collator(tokenizer, max_len=256, pad_to_multiple_of=8):
    """Pads to max_len; returns input_ids, attention_mask, labels (labels = input_ids)."""

    def collate(features):
        input_ids = []
        attention_mask = []
        for f in features:
            ids = f["input_ids"][:max_len]
            pad = [tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0] * (max_len - len(ids))
            mask = [1] * len(ids) + [0] * (max_len - len(ids))
            input_ids.append(ids + pad)
            attention_mask.append(mask)
        return {"input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(input_ids)}
    return collate


class DAPPrTrainer(Trainer):
    """Trainer with DAPPr loss: loss = mean over tokens of (log g_psi(p*|x) + lamb*reg)."""

    def __init__(self, *args, lamb=2e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.lamb = lamb

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()          # [B, T-1, V]
        labels = input_ids[:, 1:].contiguous()                    # [B, T-1]
        valid = (labels != -100).to(logits.dtype)
        if valid.sum() == 0:
            return (outputs.loss if return_outputs else torch.tensor(0.0, device=logits.device))
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        loss = dappr.DAPPr_loss(flat_logits, flat_labels, self.lamb)
        return (loss, outputs) if return_outputs else loss


@torch.no_grad()
def evaluate(model, tokenizer, test, ood_ds, device, max_len=256):
    """Accuracy + NLL + OOD AUROC (epistemic EU vs OOD), Table-5 style."""
    model.eval()
    K = model.config.vocab_size
    n_correct, n_total = 0, 0
    nll_sum, nll_n = 0.0, 0
    id_eu, ood_eu = [], []
    for split, coll in (("id", test), ("ood", ood_ds)):
        eus = []
        for ex in coll:
            prompt = ex["prompt"]
            enc = tokenizer(prompt, return_tensors="pt").to(device)
            logits = model(**enc).logits[:, -1, :]               # next-token logits
            alpha = torch.nn.functional.softplus(logits) + 1
            eu = (K / alpha.sum(1)).item()
            eus.append(eu)
            if split == "id":
                pred_token = logits.argmax(1).item()
                pred_letter = tokenizer.decode([pred_token]).strip().upper()
                n_correct += int(any(l in pred_letter for l in ex["answer_key"]))
                n_total += 1
                ans_tok = tokenizer.encode(" " + ex["answer_key"], add_special_tokens=False)
                if ans_tok:
                    logp = torch.log_softmax(logits, -1)[0, ans_tok[-1]].item()
                    nll_sum += -logp
                    nll_n += 1
        if split == "id":
            id_eu = eus
        else:
            ood_eu = eus
    acc = 100 * n_correct / n_total
    nll = nll_sum / max(nll_n, 1)
    ood_auroc = float(roc_auc_score([1] * len(id_eu) + [0] * len(ood_eu), -np.array(id_eu + ood_eu)))
    return {"test_acc": round(acc, 2), "NLL": round(nll, 3), "OOD_AUROC_ARC": round(ood_auroc, 2)}


def run(config):
    device = config.get("device", "cuda:0")
    seed = config.get("seed", 42)
    method = config.get("method", "DAPPr")      # DAPPr | CE
    lamb = config.get("lamb", 2e-4)
    max_len = config.get("max_len", 256)
    num_epochs = config.get("epochs", 5)
    lr = config.get("lr", 2e-4)
    bs = config.get("bs", 4)
    max_steps = config.get("max_steps")
    torch.manual_seed(seed); np.random.seed(seed)

    model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant,
                                                 device_map={"": int(device.split(":")[1])},
                                                 torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM"))
    model.config.use_cache = False

    train, val, test = build_dataset(tokenizer, max_len=max_len)
    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")
    arc_test = arc["test"]
    ood = arc_test.map(lambda e: {
        "prompt": f"Question: {e['question']}\nOptions:\n" + "\n".join(
            f"{l}. {t}" for l, t in zip(e["choices"]["label"], e["choices"]["text"])) + "\nAnswer:"})
    ood = ood.select(range(500)) if len(ood) > 500 else ood

    train_args = TrainingArguments(
        output_dir=f"outs/obqa_{method}",
        per_device_train_batch_size=bs, gradient_accumulation_steps=2,
        num_train_epochs=num_epochs, max_steps=max_steps,
        learning_rate=lr, warmup_ratio=0.05, lr_scheduler_type="cosine",
        logging_steps=20, save_strategy="no", report_to=[],
        fp16=True, gradient_checkpointing=True, remove_unused_columns=False)
    collator = dappr_data_collator(tokenizer, max_len=max_len)
    if method == "CE":
        collator = DataCollatorForSeq2Seq(tokenizer, padding="max_length", max_length=max_len)
        trainer = Trainer(model=model, args=train_args, train_dataset=train,
                          eval_dataset=val, data_collator=collator)
    else:
        trainer = DAPPrTrainer(model=model, args=train_args, train_dataset=train,
                               eval_dataset=val, data_collator=collator, lamb=lamb)
    trainer.train()

    metrics = evaluate(model, tokenizer, test, ood, device, max_len=max_len)
    metrics = {"config": config, "method": method, **metrics}
    print("\nFINAL_RESULTS_JSON\n" + json.dumps(metrics, indent=2) + "\nEND_RESULTS_JSON", flush=True)
    with open(f"outs/obqa_{method}_seed{seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
