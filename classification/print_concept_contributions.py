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
from dataset_utils import preprocess

# ---------------------- ARGPARSE ----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--cbl_path", type=str, required=True,
                    help="Path to trained CBL model (e.g. mpnet_acs/Duyacquy_Pubmed-20k/bert_cbm/cbl_no_backbone_acc.pt)")
parser.add_argument("--sparse", action=argparse.BooleanOptionalAction,
                    help="Use sparse final layer weights (W_g_sparse, b_g_sparse)")
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------- PARSE FROM PATH ----------------------
parts = args.cbl_path.split("/")
if len(parts) < 4:
    raise ValueError(f"Unexpected cbl_path: {args.cbl_path}")

acs = parts[0]                          # e.g., "mpnet_acs"
safe_dataset = parts[-3]                # "Duyacquy_Pubmed-20k"
dataset = safe_dataset.replace("_", "/")# "Duyacquy/Pubmed-20k"

backbone_dir = parts[-2]                # "bert_cbm" (THƯ MỤC THẬT)
backbone = backbone_dir.replace("_cbm", "")   # "bert" (LOẠI MODEL)
cbl_name = parts[-1]

print(f"[Info] Dataset: {dataset}, Backbone: {backbone}, Model: {cbl_name}")

# ---------------------- LOAD DATASET ----------------------
print("[Info] Loading test split...")
test_dataset = load_dataset(dataset, split="test")

text_col  = CFG.dataset_config[dataset]["text_column"]
label_col = CFG.dataset_config[dataset]["label_column"]

test_dataset = preprocess(test_dataset, dataset, text_col, label_col)
test_texts = test_dataset[text_col]

# ---------------------- TOKENIZATION ----------------------
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

# tokenize dựa trên text_col thay vì hard-code
enc = tokenizer(
    test_dataset[text_col],
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

# ---------------------- LOAD MODELS ----------------------
concept_set = CFG.concept_set[dataset]
print("[Info] Preparing models...")

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

# ---------------------- COMPUTE CONCEPT ACTIVATIONS ----------------------
print("[Info] Computing activations...")
FL_test_features = []

for input_ids, attention_mask in test_loader:
    batch = {"input_ids": input_ids.to(device), "attention_mask": attention_mask.to(device)}

    with torch.no_grad():
        if "no_backbone" in cbl_name:
            feats = preLM(**batch).last_hidden_state
            if "gpt2" in backbone:
                feats = eos_pooling(feats, batch["attention_mask"])
            else:  # roberta/bert
                feats = feats[:, 0, :]
            out = cbl(feats)
        else:
            out = backbone_cbl(**batch)

    FL_test_features.append(out.detach().cpu())

test_c = torch.cat(FL_test_features, dim=0)
test_c = F.relu(test_c)
print(f"[Info] test_c shape: {test_c.shape}")

# ---------------------- LOAD FINAL LINEAR LAYER ----------------------
acs = parts[0]
prefix = f"./{acs}/{dataset.replace('/', '_')}/{backbone_dir}/"
model_name = cbl_name[3:]

W_g_path = prefix + "W_g"
b_g_path = prefix + "b_g"
if args.sparse:
    W_g_path += "_sparse"
    b_g_path += "_sparse"
W_g_path += model_name
b_g_path += model_name

print(f"[Info] Loading final weights:\n - {W_g_path}\n - {b_g_path}")
W_g = torch.load(W_g_path)
b_g = torch.load(b_g_path)

final = torch.nn.Linear(len(concept_set), CFG.class_num[dataset])
final.load_state_dict({"weight": W_g, "bias": b_g})
final.eval()

# ---------------------- PREDICTIONS ----------------------
with torch.no_grad():
    logits = final(test_c)
pred = torch.argmax(logits, dim=-1).numpy()
label = np.array(test_dataset[label_col])

correct_indices = np.where(pred == label)[0]
mispred_indices = np.where(pred != label)[0]
print(f"[Info] Correct: {len(correct_indices)}, Incorrect: {len(mispred_indices)}")

# ---------------------- CONTRIBUTION SCORES ----------------------
m = test_c.unsqueeze(1) * W_g.unsqueeze(0)  # (N, num_labels, k)

error_rate = []
for i in correct_indices:
    error = total = 0
    value, c = m[i][label[i]].topk(5)
    for j in range(len(c)):
        if value[j] > 0.0:
            total += 1
            if get_labels(c[j], dataset) != label[i]:
                error += 1
    if total != 0:
        error_rate.append(error / total)
print(f"[Info] Avg error rate: {sum(error_rate)/len(error_rate):.4f}")

# ---------------------- WRITE OUTPUT ----------------------
out_path = prefix + "Concept_contribution" + W_g_path.split("/")[-1][3:-3] + ".txt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    for i in range(m.size(0)):
        f.write(test_dataset[text_col][i] + "\n")
        c = m[i][label[i]].topk(5)[1]
        n = m[i][label[i]].topk(5)[0]
        for j in range(len(c)):
            if n[j] > 0.0 and i not in mispred_indices:
                f.write(CFG.concept_set[dataset][c[j]] + "\n")
            else:
                f.write("\n")
        for j in range(len(c)):
            if n[j] > 0.0 and i not in mispred_indices:
                f.write(f"{float(n[j]):.4f}\n")
            else:
                f.write("\n")
        if i not in mispred_indices:
            f.write(str(pred[i]))
        else:
            f.write("incorrect")
        f.write("\n\n")

print(f"[Done] Concept contributions saved to {out_path}")