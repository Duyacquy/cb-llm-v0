#!/usr/bin/env bash
set -euo pipefail

# Usage: ./train.sh <dataset> <acs_model: mpnet|simcse|angle> <backbone: roberta|gpt2|bert>
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <dataset> <acs_model: mpnet|simcse|angle> <backbone: roberta|gpt2|bert>"
  exit 1
fi

DATASET="$1"           # ví dụ: Duyacquy/Pubmed-20k
ACS="$2"               # mpnet | simcse | angle
BACKBONE="$3"          # roberta | gpt2 | bert

# Tên an toàn cho thư mục
SAFE_DATASET="${DATASET//\//_}"
SAFE_BACKBONE="${BACKBONE//\//_}"

# --- Chọn thư mục cơ sở cho concept labels (ACS output) ---
CONCEPT_BASE="mpnet_acs"
case "$ACS" in
  *simcse* ) CONCEPT_BASE="simcse_acs" ;;
  *angle*  ) CONCEPT_BASE="angle_acs"  ;;
  *mpnet*  ) CONCEPT_BASE="mpnet_acs"  ;;
  * ) echo "[Warn] Unknown ACS '$ACS', default to mpnet_acs"; CONCEPT_BASE="mpnet_acs" ;;
esac

CONCEPT_DIR="${CONCEPT_BASE}/${SAFE_DATASET}"
TRAIN_NPY="${CONCEPT_DIR}/concept_labels_train.npy"
VAL_NPY="${CONCEPT_DIR}/concept_labels_val.npy"

# --- Bước 2: ACS (nếu chưa có .npy) ---
if [[ ! -f "$TRAIN_NPY" || ! -f "$VAL_NPY" ]]; then
  echo "[Info] Running ACS to create concept labels..."
  python get_concept_labels.py --dataset="${DATASET}" --concept_text_sim_model="${ACS}"
else
  echo "[Info] Concept label .npy files already exist. Skipping get_concept_labels.py."
fi

# --- Đường dẫn CBL (Step 4) ---
CBL_DIR="${CONCEPT_DIR}/${SAFE_BACKBONE}_cbm"
mkdir -p "$CBL_DIR"
CBL_PATH="${CBL_DIR}/cbl_no_backbone_acc.pt"

# --- Bước 4: Train CBL (+ACC), freeze backbone với --tune_cbl_only ---
if [[ ! -f "$CBL_PATH" ]]; then
  echo "[Info] Training CBL (ACC on, tune_cbl_only)..."
  python train_CBL.py --automatic_concept_correction --dataset="${DATASET}" --backbone="${BACKBONE}" --tune_cbl_only
else
  echo "[Info] CBL model already exists at ${CBL_PATH}. Skipping train_CBL.py."
fi

# Bắt buộc phải tồn tại CBL trước khi sang Step 5
if [[ ! -f "$CBL_PATH" ]]; then
  echo "[Error] Missing CBL checkpoint at ${CBL_PATH}."
  exit 2
fi

# --- Bước 5: Train Final Linear Layer (Elastic-Net) ---
echo "[Info] Training Final Linear (sparse &/or dense)..."
python train_FL.py --cbl_path="${CBL_PATH}" --dataset="${DATASET}" --backbone="${BACKBONE}"

# --- In concept activations & contributions ---
# Hai script print_* hiện chỉ hỗ trợ roberta/gpt2 khi tự load backbone
if [[ "$BACKBONE" == "roberta" || "$BACKBONE" == "gpt2" ]]; then
  echo "[Info] Printing concept activations..."
  python print_concept_activations.py --cbl_path "${CBL_PATH}" || echo "[Warn] print_concept_activations failed."

  echo "[Info] Printing concept contributions (sparse weights)..."
  python print_concept_contributions.py --cbl_path "${CBL_PATH}" --sparse || echo "[Warn] print_concept_contributions failed."
else
  echo "[Info] Skipping print_* scripts for backbone='${BACKBONE}' (not supported by these scripts)."
fi

echo "[Done] Pipeline completed for dataset='${DATASET}', ACS='${ACS}', backbone='${BACKBONE}'."