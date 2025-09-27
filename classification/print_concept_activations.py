import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    RobertaTokenizerFast, RobertaModel,
    GPT2TokenizerFast, GPT2Model,
    AutoTokenizer, AutoModel
)
from datasets import load_dataset
import config as CFG
from modules import CBL, RobertaCBL, GPT2CBL
from utils import normalize, get_labels, eos_pooling

parser = argparse.ArgumentParser()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--cbl_path", type=str, default="mpnet_acs/SetFit_sst2/roberta_cbm/cbl.pt")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--dropout", type=float, default=0.1)

# ---------------- Collate an toàn: ép mọi field trong batch thành torch.Tensor ----------------
def hf_collate(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]

        # Nếu là chuỗi (hoặc list toàn chuỗi) thì giữ nguyên dạng list
        if isinstance(v0, (str, bytes)) or (
            isinstance(v0, (list, tuple)) and len(v0) > 0 and isinstance(v0[0], (str, bytes))
        ):
            out[k] = vals
            continue

        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(v0, (list, tuple)):
            try:
                out[k] = torch.tensor(vals)
            except Exception:
                # fallback nếu không chuyển được (ví dụ đối tượng lẫn lộn)
                out[k] = vals
        elif isinstance(v0, (int, float, np.integer, np.floating)):
            out[k] = torch.tensor(vals)
        elif isinstance(v0, np.ndarray):
            out[k] = torch.from_numpy(np.stack(vals))
        else:
            try:
                out[k] = torch.as_tensor(vals)
            except Exception:
                out[k] = vals
    return out


# ---------------- Dataset wrapper: chỉ expose cột cần dùng ----------------
class ClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, ds, columns):
        self.ds = ds
        self.columns = columns  # ví dụ: ['input_ids','attention_mask','label']

    def __getitem__(self, idx):
        return {k: self.ds[k][idx] for k in self.columns}

    def __len__(self):
        return self.ds.num_rows

def build_loader(ds, columns, mode):
    dataset = ClassificationDataset(ds, columns)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=(mode == "train"),
        collate_fn=hf_collate,
        pin_memory=torch.cuda.is_available(),
    )

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args, _ = parser.parse_known_args()

    # ---- Parse cbl_path ----
    cbl_path = args.cbl_path.strip()
    parts = cbl_path.split("/")
    if len(parts) < 3:
        raise ValueError(f"--cbl_path không hợp lệ: {cbl_path}")

    acs = parts[0]              # mpnet_acs / simcse_acs / angle_acs
    dataset_dir = parts[1]      # ví dụ: Duyacquy_Pubmed-20k
    cbl_name = parts[-1]        # ví dụ: cbl_no_backbone_acc.pt
    backbone_dir = parts[2]     # ví dụ: roberta_cbm, gpt2_cbm, bert ...

    # Phục hồi 'org/dataset' nếu bị '_' lần đầu
    if "/" not in dataset_dir and "_" in dataset_dir:
        org, rest = dataset_dir.split("_", 1)
        dataset_hf = f"{org}/{rest}"
    else:
        dataset_hf = dataset_dir

    # Suy ra backbone
    lower_path = cbl_path.lower()
    if "roberta" in lower_path:
        backbone = "roberta"
    elif "gpt2" in lower_path:
        backbone = "gpt2"
    elif "bert" in lower_path:
        backbone = "bert"
    else:
        seg = parts[2].lower() if len(parts) > 2 else ""
        if any(k in seg for k in ["roberta", "gpt2", "bert"]):
            backbone = "roberta" if "roberta" in seg else ("gpt2" if "gpt2" in seg else "bert")
        else:
            raise Exception(f"Cannot infer backbone from cbl_path='{cbl_path}' (need roberta/gpt2/bert).")

    print("------------------------CONCEPT_ACTIVATION---------------------")
    print("loading data...")
    test_dataset = load_dataset(dataset_hf, split='test')
    print("test data len:", len(test_dataset))

    # --- Lấy text column key từ CFG ---
    cfg_key = dataset_hf if dataset_hf in CFG.example_name else dataset_hf.replace("/", "_")
    if cfg_key not in CFG.example_name:
        raise KeyError(f"Không tìm thấy text key cho '{cfg_key}' trong CFG.example_name")
    text_key = CFG.example_name[cfg_key]

    # --- Tokenizer ---
    print("tokenizing...")
    if backbone == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif backbone == 'gpt2':
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
    elif backbone == 'bert':
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    else:
        raise Exception("backbone should be roberta, gpt2 or bert")

    # --- Tokenize 1 lần ---
    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[text_key], padding=True, truncation=True, max_length=args.max_length),
        batched=True, batch_size=len(test_dataset)
    )
    encoded_test_dataset = encoded_test_dataset.remove_columns([text_key])
    # Một số dataset có cột phụ cần bỏ
    if cfg_key in ('SetFit/sst2', 'SetFit_sst2') and 'label_text' in encoded_test_dataset.column_names:
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if cfg_key == 'dbpedia_14' and 'title' in encoded_test_dataset.column_names:
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])

    # --- Nhãn: PubMed dùng 'target'; nếu không có, fallback ---
    raw_cols = set(test_dataset.column_names)
    label_col = "target" if "target" in raw_cols else None
    if label_col is None:
        for cand in ["label", "labels", "class", "y"]:
            if cand in raw_cols:
                label_col = cand
                break
    if label_col is None:
        raise ValueError(f"Không tìm thấy cột nhãn trong {test_dataset.column_names}")

    # Lưu nhãn gốc trước khi lỡ xoá
    orig_labels = test_dataset[label_col]
    try:
        orig_labels = [int(x) for x in orig_labels]
    except Exception:
        pass

    # ---- Giữ đúng cột cần thiết & khôi phục label nếu bị thiếu ----
    keep_cols = [c for c in ["input_ids", "attention_mask", "token_type_ids", label_col]
                 if c in encoded_test_dataset.column_names]
    drop_cols = [c for c in encoded_test_dataset.column_names if c not in keep_cols]
    if drop_cols:
        encoded_test_dataset = encoded_test_dataset.remove_columns(drop_cols)
    if label_col not in encoded_test_dataset.column_names:
        encoded_test_dataset = encoded_test_dataset.add_column(label_col, orig_labels)
        if label_col not in keep_cols:
            keep_cols.append(label_col)

    # Cho HF Dataset trả tensor trực tiếp
    encoded_test_dataset = encoded_test_dataset.with_format(type="torch", columns=keep_cols)

    print("creating loader...")
    test_loader = build_loader(encoded_test_dataset, keep_cols, mode="test")

    # ---- Chuẩn bị mô hình (CBL / Backbone+CBL) ----
    concept_set = CFG.concept_set[cfg_key]
    if backbone == 'roberta':
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = RobertaModel.from_pretrained('roberta-base').to(device)
            preLM.eval()
        else:
            print("preparing backbone(roberta)+CBL...")
            backbone_cbl = RobertaCBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    elif backbone == 'gpt2':
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = GPT2Model.from_pretrained('gpt2').to(device)
            preLM.eval()
        else:
            print("preparing backbone(gpt2)+CBL...")
            backbone_cbl = GPT2CBL(len(concept_set), args.dropout).to(device)
            backbone_cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            backbone_cbl.eval()
    elif backbone == 'bert':
        if 'no_backbone' in cbl_name:
            print("preparing CBL only...")
            cbl = CBL(len(concept_set), args.dropout).to(device)
            cbl.load_state_dict(torch.load(args.cbl_path, map_location=device))
            cbl.eval()
            preLM = AutoModel.from_pretrained('bert-base-uncased').to(device)
            preLM.eval()
        else:
            raise Exception("BERT with backbone_cbl not implemented in this script. Use 'cbl_no_backbone_*.pt'.")
    else:
        raise Exception("backbone should be roberta, gpt2 or bert")

    # ---- Trích concept activations ----
    print("get concept features...")
    FL_test_features = []
    for batch in test_loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            if 'no_backbone' in cbl_name:
                test_features = preLM(input_ids=batch["input_ids"],
                                      attention_mask=batch["attention_mask"]).last_hidden_state
                if backbone in ['roberta', 'bert']:
                    test_features = test_features[:, 0, :]  # [CLS]
                elif backbone == 'gpt2':
                    test_features = eos_pooling(test_features, batch["attention_mask"])
                else:
                    raise Exception("backbone should be roberta, gpt2 or bert")
                test_features = cbl(test_features)
            else:
                test_features = backbone_cbl(batch)  # nhận dict
            FL_test_features.append(test_features)
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()  # [N, num_concepts]

    # ---- Load mean/std từ cùng thư mục với cbl_path ----
    base_dir = os.path.dirname(args.cbl_path) + "/"
    model_name = os.path.basename(args.cbl_path)[3:]  # 'cbl_xxx.pt' -> '_xxx.pt'
    train_mean = torch.load(base_dir + 'train_mean' + model_name)
    train_std  = torch.load(base_dir + 'train_std'  + model_name)

    # ---- Normalize + ReLU ----
    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    # ---- Lấy nhãn ----
    label = encoded_test_dataset[label_col]  # torch.LongTensor

    # ---- Tính error rate theo code gốc (trên activations) ----
    error_rate = []
    for i in range(test_c.T.size(0)):  # từng concept
        error = 0
        total = 0
        value, s = test_c.T[i].topk(5)
        for j in range(5):
            if value[j] > 1.0:  # ngưỡng như code gốc bạn đang dùng
                total += 1
                if get_labels(i, dataset_hf) != int(label[int(s[j])]):
                    error += 1
        if total != 0:
            error_rate.append(error / total)
    print("avg error rate:", (sum(error_rate) / len(error_rate)) if error_rate else "N/A")

    # ---- Ghi file Concept_activation ----
    out_path = base_dir + 'Concept_activation' + model_name[:-3] + '.txt'
    with open(out_path, 'w') as f:
        for i in range(test_c.T.size(0)):
            f.write(CFG.concept_set[cfg_key][i])
            f.write('\n')
            value, s = test_c.T[i].topk(5)
            for j in range(5):
                if value[j] > 0.0:
                    f.write(test_dataset[text_key][int(s[j])])
                    f.write('\n')
                else:
                    f.write('\n')
            for j in range(5):
                if value[j] > 0.0:
                    f.write("{:.4f}".format(float(value[j])))
                    f.write('\n')
                else:
                    f.write('\n')
            f.write('\n')
    print("Saved:", out_path)
    print("encoded_test_dataset columns:", encoded_test_dataset.column_names)
    print("sample types:", {k: type(encoded_test_dataset[k][0]) for k in encoded_test_dataset.column_names})