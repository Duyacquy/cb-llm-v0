#!/usr/bin/env python3
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    RobertaTokenizerFast, RobertaModel,
    GPT2TokenizerFast, GPT2Model,
    AutoTokenizer, BertModel
)
from datasets import load_dataset
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL, BERTCBL
from utils import normalize, get_labels, eos_pooling

# ------------------ Argument parsing ------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cbl_path", type=str, required=True,
                    help="Path to trained CBL model (e.g., mpnet_acs/Duyacquy_Pubmed-20k/bert_cbm/cbl_no_backbone_acc.pt)")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ------------------ Parse dataset/backbone from path ------------------
parts = args.cbl_path.split("/")
if len(parts) < 4:
    raise ValueError(f"Unexpected cbl_path: {args.cbl_path}")

safe_dataset = parts[-3]                 # e.g. Duyacquy_Pubmed-20k
dataset = safe_dataset.replace("_", "/") # => Duyacquy/Pubmed-20k
backbone = parts[-2].replace("_cbm", "") # roberta | gpt2 | bert
cbl_name = parts[-1]

print(f"[Info] Dataset: {dataset}, Backbone: {backbone}, Model: {cbl_name}")

# ------------------ Load dataset ------------------
print("[Info] Loading test split...")
test_dataset = load_dataset(dataset, split="test")

# Determine text column name
text_col = CFG.dataset_config.get(dataset, {}).get("text_column", None)
if text_col is None:
    text_col = "text" if "text" in test_dataset.column_names else test_dataset.column_names[0]

print(f"[Info] Using text column: {text_col}")

test_texts = test_dataset[text_col]

# ------------------ Tokenization ------------------
print("[Info] Tokenizing test data...")
if "roberta" in backbone:
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
elif "gpt2" in backbone:
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
elif "bert" in backbone:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
else:
    raise ValueError("Unsupported backbone")

enc = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    max_length=args.max_length,
    return_tensors="pt"
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"]),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=max(0, args.num_workers),
)

# ------------------ Load CBL / Backbone ------------------
concept_set = CFG.concept_set[dataset]
print("[Info] Preparing model(s)...")

if "no_backbone" in cbl_name:
    print("[Info] Loading CBL only...")
    cbl = CBL(len(concept_set), args.dropout).to(device)
    cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
    cbl.eval()

    if "roberta" in backbone:
        preLM = RobertaModel.from_pretrained("roberta-base").to(device)
    elif "gpt2" in backbone:
        preLM = GPT2Model.from_pretrained("gpt2").to(device)
    elif "bert" in backbone:
        preLM = BertModel.from_pretrained("bert-base-uncased").to(device)
    preLM.eval()
else:
    print(f"[Info] Loading {backbone}+CBL...")
    if "roberta" in backbone:
        backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
    elif "gpt2" in backbone:
        backbone_cbl = GPT2CBL(len(concept_set), args.dropout).to(device)
    elif "bert" in backbone:
        backbone_cbl = BERTCBL(len(concept_set), args.dropout).to(device)
    backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
    backbone_cbl.eval()

# ------------------ Compute activations ------------------
print("[Info] Computing concept activations...")
FL_test_features = []

for input_ids, attention_mask in test_loader:
    batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}

    with torch.no_grad():
        if "no_backbone" in cbl_name:
            feats = preLM(**batch).last_hidden_state
            if "gpt2" in backbone:
                feats = eos_pooling(feats, batch["attention_mask"])
            else:  # roberta / bert → CLS pooling
                feats = feats[:, 0, :]
            out = cbl(feats)
        else:
            out = backbone_cbl(**batch)

    FL_test_features.append(out.detach().cpu())

test_c = torch.cat(FL_test_features, dim=0)
print(f"[Info] Feature shape: {test_c.shape}")

# ------------------ Normalize & relu ------------------
acs = parts[0]  # e.g. mpnet_acs
prefix = f"./{acs}/{dataset.replace('/', '_')}/{backbone}/"
model_name = cbl_name[3:]

train_mean = torch.load(prefix + "train_mean" + model_name)
train_std = torch.load(prefix + "train_std" + model_name)

test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
test_c = F.relu(test_c)

label = test_dataset["label"]
error_rate = []

for i in range(test_c.T.size(0)):
    error = total = 0
    value, s = test_c.T[i].topk(5)
    for j in range(5):
        if value[j] > 1.0:
            total += 1
            if get_labels(i, dataset) != label[s[j]]:
                error += 1
    if total > 0:
        error_rate.append(error / total)

print(f"[Info] Avg error rate: {sum(error_rate) / len(error_rate):.4f}")

# ------------------ Write activations ------------------
out_path = prefix + "Concept_activation" + args.cbl_path.split("/")[-1][3:-3] + ".txt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    for i in range(test_c.T.size(0)):
        f.write(CFG.concept_set[dataset][i] + "\n")
        value, s = test_c.T[i].topk(5)
        for j in range(5):
            if value[j] > 0.0:
                f.write(test_dataset[text_col][s[j]] + "\n")
            else:
                f.write("\n")
        for j in range(5):
            f.write(f"{float(value[j]):.4f}\n" if value[j] > 0.0 else "\n")
        f.write("\n")

print(f"[Done] Concept activations saved to {out_path}")