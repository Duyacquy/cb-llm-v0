import argparse, os, sys, json
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

acs = parts[0]                           # e.g., "mpnet_acs"
safe_dataset = parts[-3]                 # "Duyacquy_Pubmed-20k"
dataset = safe_dataset.replace("_", "/") # "Duyacquy/Pubmed-20k"

backbone_dir = parts[-2]                 # "bert_cbm"
backbone = backbone_dir.replace("_cbm", "")   # "bert"
cbl_name = parts[-1]

print(f"[Info] Dataset: {dataset}, Backbone: {backbone}, Model: {cbl_name}")
print(f"[Info] ACS folder: {acs}")

# === DEBUG: cảnh báo nếu ACS folder không khớp một trong {mpnet_acs, simcse_acs, angle_acs, llm_labeling}
if not any(acs.startswith(x) for x in ["mpnet_acs", "simcse_acs", "angle_acs", "llm_labeling"]):
    print(f"[WARN][DEBUG] Unknown ACS folder '{acs}'. Check you didn't mix up folders.", file=sys.stderr)

# ------------------ Load dataset ------------------
print("[Info] Loading test split...")
test_dataset = load_dataset(dataset, split="test")

text_col  = CFG.dataset_config[dataset]["text_column"]
label_col = CFG.dataset_config[dataset]["label_column"]

# === DEBUG: in thứ tự tên nhãn nếu có ClassLabel
label_names = None
try:
    label_names = test_dataset.features[label_col].names
    print(f"[DEBUG] Label names order from dataset: {label_names}")
except Exception:
    print("[DEBUG] Dataset label column is not ClassLabel, skipping label name print.")

test_dataset = preprocess(test_dataset, dataset, text_col, label_col)

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

# ------------------ Load CBL / Backbone ------------------
concept_set = CFG.concept_set[dataset]
k_cfg = len(concept_set)
print("[Info] Preparing model(s)...")
print(f"[DEBUG] concept_set size (k from config): {k_cfg}")
print(f"[DEBUG] First 5 concepts: {list(concept_set[:5])}")

if "no_backbone" in cbl_name:
    print("[Info] Loading CBL only...")
    cbl = CBL(k_cfg, args.dropout).to(device)
    state = torch.load(args.cbl_path, map_location=device)
    cbl.load_state_dict(state)
    cbl.eval()

    if "roberta" in backbone:
        preLM = RobertaModel.from_pretrained("roberta-base").to(device)
    elif "gpt2" in backbone:
        preLM = GPT2Model.from_pretrained("gpt2").to(device)
    elif "bert" in backbone:
        preLM = BertModel.from_pretrained("bert-base-uncased").to(device)
    preLM.eval()

    # === DEBUG: in kích thước FC của CBL (k trong checkpoint)
    fc_weight = cbl.fc.weight.detach().cpu()
    k_ckpt = fc_weight.shape[0]
    print(f"[DEBUG] CBL fc weight shape: {tuple(fc_weight.shape)} (k_ckpt={k_ckpt}, d={fc_weight.shape[1]})")
    if k_ckpt != k_cfg:
        print(f"[ERR][DEBUG] k mismatch: ckpt={k_ckpt}, config={k_cfg}. You must retrain or align concepts.", file=sys.stderr)
else:
    print(f"[Info] Loading {backbone}+CBL...")
    if "roberta" in backbone:
        backbone_cbl = RobertaCBL(k_cfg, args.dropout).to(device)
    elif "gpt2" in backbone:
        backbone_cbl = GPT2CBL(k_cfg, args.dropout).to(device)
    elif "bert" in backbone:
        backbone_cbl = BERTCBL(k_cfg, args.dropout).to(device)
    state = torch.load(args.cbl_path, map_location=device)
    backbone_cbl.load_state_dict(state)
    backbone_cbl.eval()

# === DEBUG: đếm concept/label theo mapping get_labels
label_counts = {}
for j in range(k_cfg):
    yj = get_labels(j, dataset)
    label_counts[yj] = label_counts.get(yj, 0) + 1
print(f"[DEBUG] concept counts by label via get_labels(): {label_counts}")

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

# === DEBUG: thống kê activation trước normalize
def tstats(t, name):
    t_np = t.float().view(-1).numpy()
    print(f"[DEBUG] {name}: min={t_np.min():.4f}, max={t_np.max():.4f}, mean={t_np.mean():.4f}, std={t_np.std():.4f}")

tstats(test_c, "test_c (raw, pre-normalize)")

# ------------------ Normalize & ReLU ------------------
prefix = f"./{acs}/{dataset.replace('/', '_')}/{backbone_dir}/"
model_name = cbl_name[3:]  # e.g. "_no_backbone_acc.pt"

mean_path = prefix + "train_mean" + model_name
std_path  = prefix + "train_std"  + model_name

# === DEBUG: an toàn nếu thiếu mean/std
if not (os.path.exists(mean_path) and os.path.exists(std_path)):
    print(f"[WARN][DEBUG] Missing train_mean/std at {mean_path} or {std_path}. Will skip normalize and only ReLU.", file=sys.stderr)
    test_c_norm = test_c.clone()
    mean_used = torch.zeros(test_c.shape[1])
    std_used  = torch.ones(test_c.shape[1])
else:
    train_mean = torch.load(mean_path, map_location="cpu")
    train_std  = torch.load(std_path,  map_location="cpu")
    # === DEBUG: check shape
    if train_mean.numel() != test_c.shape[1] or train_std.numel() != test_c.shape[1]:
        print(f"[ERR][DEBUG] mean/std size mismatch: mean={train_mean.numel()}, std={train_std.numel()}, k={test_c.shape[1]}", file=sys.stderr)
    test_c_norm, mean_used, std_used = normalize(test_c, d=0, mean=train_mean, std=train_std)

tstats(test_c_norm, "test_c (after normalize, pre-ReLU)")
test_c_relu = F.relu(test_c_norm)
tstats(test_c_relu, "test_c (after ReLU)")

# ------------------ labels as numpy int64 (fix torch.from_numpy error) ------------------
raw_label = test_dataset[label_col]
try:
    # nếu là tên chuỗi, map về id theo label_names nếu có
    if len(raw_label) > 0 and isinstance(raw_label[0], str):
        if label_names is None:
            raise ValueError("Label column is str but dataset has no ClassLabel names to map.")
        name2id = {n: i for i, n in enumerate(label_names)}
        label = np.array([name2id[x] for x in raw_label], dtype=np.int64)
    else:
        label = np.array(raw_label, dtype=np.int64)
except Exception as e:
    print(f"[ERR][DEBUG] Failed to cast labels to int64: {e}", file=sys.stderr)
    # fallback: try pandas-like coercion
    label = np.array(raw_label)
    # nếu vẫn không phải số, stop sớm
    if label.dtype.kind not in ("i", "u"):
        raise

# === DEBUG: in phân bố nhãn
unique, counts = np.unique(label, return_counts=True)
print(f"[DEBUG] label distribution: " + ", ".join([f"{int(u)}:{int(c)}" for u, c in zip(unique, counts)]))

# === DEBUG: thống kê ngưỡng >0 và >1 theo từng concept
above0 = (test_c_relu > 0).sum(dim=0).numpy()
above1 = (test_c_relu > 1.0).sum(dim=0).numpy()
print(f"[DEBUG] per-concept count(samples with act>0): min={above0.min()}, max={above0.max()}, mean={above0.mean():.2f}")
print(f"[DEBUG] per-concept count(samples with act>1.0): min={above1.min()}, max={above1.max()}, mean={above1.mean():.2f}")

# ------------------ Error rate (purity-style) ------------------
# Lưu ý: code gốc chỉ tính với value>1.0. In thêm biến thể với >0.0 để biết ngưỡng có quá gắt không.
error_rate_strict = []
error_rate_loose  = []

for i in range(test_c_relu.T.size(0)):
    # strict: >1.0
    err = tot = 0
    value, s = test_c_relu.T[i].topk(5)
    for j in range(5):
        if value[j] > 1.0:
            tot += 1
            if get_labels(i, dataset) != label[s[j]]:
                err += 1
    if tot > 0:
        error_rate_strict.append(err / tot)

    # loose: >0.0
    err2 = tot2 = 0
    for j in range(5):
        if value[j] > 0.0:
            tot2 += 1
            if get_labels(i, dataset) != label[s[j]]:
                err2 += 1
    if tot2 > 0:
        error_rate_loose.append(err2 / tot2)

def avg_or_nan(arr):
    return float(sum(arr)/len(arr)) if len(arr)>0 else float("nan")

print(f"[Info] Avg error rate (strict >1.0): {avg_or_nan(error_rate_strict):.4f}  (counted concepts: {len(error_rate_strict)})")
print(f"[Info] Avg error rate (loose  >0.0): {avg_or_nan(error_rate_loose):.4f}  (counted concepts: {len(error_rate_loose)})")

# === DEBUG (MAPPING): xem thứ tự tên nhãn & phân tích majority label cho mỗi concept
if label_names is not None:
    print(f"[DEBUG][MAP] Dataset label id -> name order: {list(enumerate(label_names))}")

K = 5
num_concepts = test_c_relu.shape[1]
num_labels = int(label.max()) + 1
topk_label_counts = np.zeros((num_concepts, num_labels), dtype=int)

for i in range(num_concepts):
    vals, idxs = test_c_relu.T[i].topk(K)
    labs = label[idxs.numpy()]
    for lb in labs:
        topk_label_counts[i, lb] += 1

mismatch_all = True
for i in range(num_concepts):
    supposed = get_labels(i, dataset)
    maj_lb = int(topk_label_counts[i].argmax())
    maj_cnt = int(topk_label_counts[i, maj_lb])
    print(f"[DEBUG][MAP] concept#{i:02d}  supposed={supposed}  topK-major={maj_lb} (count={maj_cnt}/{K})")
    if maj_lb == supposed and maj_cnt > 0:
        mismatch_all = False

if mismatch_all:
    print("[DEBUG][MAP] WARNING: For ALL concepts, top-K major label != supposed label -> likely label-id permutation or boundaries mismatch.")

supp_L = np.array([get_labels(i, dataset) for i in range(num_concepts)])
block_counts = np.zeros((num_labels, num_labels), dtype=int)  # rows: supposed, cols: actual-majority

for i in range(num_concepts):
    supposed = supp_L[i]
    maj_lb = int(topk_label_counts[i].argmax())
    block_counts[supposed, maj_lb] += 1

print("[DEBUG][MAP] block_counts (rows = supposed label, cols = actual-majority label):")
print(block_counts)

suggest_perm = block_counts.argmax(axis=1).tolist()  # supposed->actual
print(f"[DEBUG][MAP] suggested permutation (supposed_id -> actual_id): {suggest_perm}")
if label_names is not None and len(label_names) >= len(suggest_perm):
    print("[DEBUG][MAP] suggested mapping (name): " +
          ", ".join([f"{label_names[r]} -> {label_names[c]}" for r, c in enumerate(suggest_perm)]))

# ------------------ (Optional) cosine với ACC(ACS_test) nếu có ------------------
acs_test_path = f"./{acs}/{safe_dataset}/concept_labels_test.npy"
if os.path.exists(acs_test_path):
    Sc = torch.from_numpy(np.load(acs_test_path)).float()
    # ACC mask cho test (đồng bộ với train_CBL)
    concept_labels = np.array([get_labels(j, dataset) for j in range(k_cfg)])
    label_matches = (label[:, None] == concept_labels[None, :])
    Sc[~torch.from_numpy(label_matches)] = 0.0
    Sc.clamp_min_(0.0)
    # chuẩn hoá để so cosine
    Sc_n = F.normalize(Sc, dim=1)
    Tc_n = F.normalize(test_c_relu, dim=1)
    cos = (Sc_n * Tc_n).sum(dim=1).numpy()
    print(f"[DEBUG] cosine(test_c_relu, ACC(ACS_test)): mean={cos.mean():.4f}, std={cos.std():.4f}, min={cos.min():.4f}, max={cos.max():.4f}")
else:
    print(f"[DEBUG] No ACS test file at {acs_test_path}; skipping cosine-to-ACS check.")

# ------------------ Write activations ------------------
out_path = prefix + "Concept_activation" + args.cbl_path.split("/")[-1][3:-3] + ".txt"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    for i in range(test_c_relu.T.size(0)):
        f.write(CFG.concept_set[dataset][i] + "\n")
        value, s = test_c_relu.T[i].topk(5)
        for j in range(5):
            if value[j] > 0.0:
                f.write(test_dataset[text_col][s[j]] + "\n")
            else:
                f.write("\n")
        for j in range(5):
            f.write(f"{float(value[j]):.4f}\n" if value[j] > 0.0 else "\n")
        f.write("\n")

print(f"[Done] Concept activations saved to {out_path}")