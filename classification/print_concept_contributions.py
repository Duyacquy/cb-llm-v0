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
parser.add_argument('--sparse', action=argparse.BooleanOptionalAction, help="Dùng W_g_sparse/b_g_sparse nếu bật")


# ---------------- Collate an toàn (ép mọi thứ thành torch.Tensor) ----------------
def hf_collate(batch):
    out = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(v0, (list, tuple)):
            out[k] = torch.tensor(vals)
        elif isinstance(v0, (int, float, np.integer, np.floating)):
            out[k] = torch.tensor(vals)
        elif isinstance(v0, np.ndarray):
            out[k] = torch.from_numpy(np.stack(vals))
        else:
            out[k] = torch.as_tensor(vals)
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

    acs = parts[0]              # mpnet_acs/simcse_acs/angle_acs
    dataset_dir = parts[1]      # ví dụ: Duyacquy_Pubmed-20k
    cbl_name = parts[-1]        # ví dụ: cbl_no_backbone_acc.pt
    backbone_dir = parts[2]     # ví dụ: roberta_cbm, gpt2_cbm hoặc bert

    # Phục hồi 'org/dataset' nếu bị '_' ở lần đầu
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
            raise Exception(f"Cannot infer backbone from cbl_path='{cbl_path}' (need roberta/gpt2/bert in path).")

    print("------------------------CONCEPT_CONTRIBUTED---------------------")
    print("loading data...")
    test_dataset = load_dataset(dataset_hf, split='test')
    print("test data len: ", len(test_dataset))

    # Key text theo CFG
    cfg_key = dataset_hf if dataset_hf in CFG.example_name else dataset_hf.replace("/", "_")
    if cfg_key not in CFG.example_name:
        raise KeyError(f"Không tìm thấy text key cho '{cfg_key}' trong CFG.example_name")
    text_key = CFG.example_name[cfg_key]

    # Tokenizer
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

    # Tokenize
    encoded_test_dataset = test_dataset.map(
        lambda e: tokenizer(e[text_key], padding=True, truncation=True, max_length=args.max_length),
        batched=True, batch_size=len(test_dataset)
    )
    encoded_test_dataset = encoded_test_dataset.remove_columns([text_key])

    # Bỏ cột thừa (nếu còn)
    if cfg_key in ('SetFit/sst2', 'SetFit_sst2') and 'label_text' in encoded_test_dataset.column_names:
        encoded_test_dataset = encoded_test_dataset.remove_columns(['label_text'])
    if cfg_key == 'dbpedia_14' and 'title' in encoded_test_dataset.column_names:
        encoded_test_dataset = encoded_test_dataset.remove_columns(['title'])

    # ---- Lấy/giữ lại cột nhãn ----
    raw_cols = set(test_dataset.column_names)
    # PubMed dùng 'target'; nếu không có thì fallback
    label_col = "target" if "target" in raw_cols else None
    if label_col is None:
        for cand in ["label", "labels", "class", "y"]:
            if cand in raw_cols:
                label_col = cand
                break
    if label_col is None:
        raise ValueError(f"Không tìm thấy cột nhãn trong {test_dataset.column_names}")

    # Lưu nhãn gốc để phòng bị remove
    orig_labels = test_dataset[label_col]

    # Giữ đúng cột cần thiết + khôi phục label nếu đã mất
    keep_cols = [c for c in ["input_ids", "attention_mask", "token_type_ids", label_col]
                 if c in encoded_test_dataset.column_names]
    drop_cols = [c for c in encoded_test_dataset.column_names if c not in keep_cols]
    if drop_cols:
        encoded_test_dataset = encoded_test_dataset.remove_columns(drop_cols)
    if label_col not in encoded_test_dataset.column_names:
        encoded_test_dataset = encoded_test_dataset.add_column(label_col, orig_labels)
        if label_col not in keep_cols:
            keep_cols.append(label_col)

    # (Tuỳ chọn) Để HF trả về luôn tensor cho các cột keep_cols
    encoded_test_dataset = encoded_test_dataset.with_format(type="torch", columns=keep_cols)

    print("creating loader...")
    test_loader = build_loader(encoded_test_dataset, keep_cols, mode="test")

    # ---- Chuẩn bị mô hình ----
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

    # ---- Extract concept features ----
    print("get concept features...")
    FL_test_features = []
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
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
                # backbone_cbl nhận dict batch (đã có input_ids, attention_mask, ...)
                test_features = backbone_cbl(batch)
            FL_test_features.append(test_features)
    test_c = torch.cat(FL_test_features, dim=0).detach().cpu()

    # ---- Load mean/std & W_g, b_g ngay cạnh cbl_path ----
    base_dir = os.path.dirname(args.cbl_path) + "/"
    model_name = os.path.basename(args.cbl_path)[3:]  # 'cbl_xxx.pt' -> '_xxx.pt'
    train_mean = torch.load(base_dir + 'train_mean' + model_name)
    train_std  = torch.load(base_dir + 'train_std'  + model_name)

    test_c, _, _ = normalize(test_c, d=0, mean=train_mean, std=train_std)
    test_c = F.relu(test_c)

    # Nhãn (torch tensor)
    label = encoded_test_dataset[label_col]  # đã là torch.Tensor

    # Final layer để shape + dùng W_g trực tiếp
    final = torch.nn.Linear(in_features=len(concept_set), out_features=CFG.class_num[cfg_key])

    W_g_path = base_dir + ( "W_g_sparse" if args.sparse else "W_g" ) + model_name
    b_g_path = base_dir + ( "b_g_sparse" if args.sparse else "b_g" ) + model_name
    W_g = torch.load(W_g_path)
    b_g = torch.load(b_g_path)
    final.load_state_dict({"weight": W_g, "bias": b_g})

    with torch.no_grad():
        logits = final(test_c)
        pred = np.argmax(logits.detach().cpu().numpy(), axis=-1)

    correct_indices = np.where(pred == label.cpu().numpy())[0]
    mispred_indices = np.where(pred != label.cpu().numpy())[0]

    # Mảng đóng góp: [N, num_classes, num_concepts]
    m = test_c.unsqueeze(1) * W_g.unsqueeze(0)
    print("contrib tensor size:", m.size())

    # Tính error rate như code gốc
    error_rate = []
    for i in correct_indices:
        error = 0
        total = 0
        gold = int(label[i])
        value, c = m[i][gold].topk(5)
        for j in range(len(c)):
            if value[j] > 0.0:
                total += 1
                if get_labels(int(c[j]), dataset_hf) != gold:
                    error += 1
        if total != 0:
            error_rate.append(error/total)

    print("avg error rate:", (sum(error_rate)/len(error_rate)) if error_rate else "N/A")

    # ---- Ghi file Concept_contribution ----
    out_path = base_dir + 'Concept_contribution' + model_name[:-3] + '.txt'
    with open(out_path, 'w') as f:
        for i in range(m.size(0)):
            f.write(test_dataset[text_key][i])
            f.write('\n')
            gold = int(label[i])
            c = m[i][gold].topk(5)[1]
            n = m[i][gold].topk(5)[0]
            for j in range(len(c)):
                if n[j] > 0.0 and i not in mispred_indices:
                    f.write(CFG.concept_set[cfg_key][int(c[j])])
                    f.write('\n')
                else:
                    f.write('\n')
            for j in range(len(c)):
                if n[j] > 0.0 and i not in mispred_indices:
                    f.write("{:.4f}".format(float(n[j])))
                    f.write('\n')
                else:
                    f.write('\n')
            if i not in mispred_indices:
                f.write(str(int(pred[i])))
            else:
                f.write("incorrect")
            f.write('\n\n')
    print("Saved:", out_path)