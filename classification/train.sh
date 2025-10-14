#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./train.sh <dataset> <acs_model: mpnet|simcse|angle> <backbone: roberta|gpt2|bert>
#
# Ví dụ:
#   ./train.sh "Duyacquy/Pubmed-20k" mpnet bert

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <dataset> <acs_model: mpnet|simcse|angle> <backbone: roberta|gpt2|bert>"
  exit 1
fi

DATASET="$1"         # ví dụ: Duyacquy/Pubmed-20k
ACS="$2"             # mpnet | simcse | angle
BACKBONE="$3"        # roberta | gpt2 | bert

SAFE_DATASET="${DATASET//\//_}"
SAFE_BACKBONE="${BACKBONE//\//_}"

timestamp() { date +"%F %T"; }
info() { echo "[Info $(timestamp)] $*"; }
warn() { echo "[Warn $(timestamp)] $*" >&2; }
err()  { echo "[Error $(timestamp)] $*" >&2; }

# --- Chọn thư mục cơ sở cho concept labels (ACS output) ---
CONCEPT_BASE="mpnet_acs"
case "$ACS" in
  *simcse* ) CONCEPT_BASE="simcse_acs" ;;
  *angle*  ) CONCEPT_BASE="angle_acs"  ;;
  *mpnet*  ) CONCEPT_BASE="mpnet_acs"  ;;
  * ) warn "Unknown ACS '$ACS', default to mpnet_acs"; CONCEPT_BASE="mpnet_acs" ;;
esac

CONCEPT_DIR="${CONCEPT_BASE}/${SAFE_DATASET}"
TRAIN_NPY="${CONCEPT_DIR}/concept_labels_train.npy"
VAL_NPY="${CONCEPT_DIR}/concept_labels_val.npy"

# --- Step 2: ACS (nếu chưa có .npy) ---
if [[ ! -f "$TRAIN_NPY" || ! -f "$VAL_NPY" ]]; then
  info "Running ACS to create concept labels with model='${ACS}'..."
  mkdir -p "${CONCEPT_DIR}"
  python get_concept_labels.py \
    --dataset="${DATASET}" \
    --concept_text_sim_model="${ACS}"
else
  info "Concept label .npy files already exist. Skipping get_concept_labels.py."
fi

# --- Đường dẫn CBL (Step 4) ---
CBL_DIR="${CONCEPT_DIR}/${SAFE_BACKBONE}_cbm"
mkdir -p "${CBL_DIR}"
CBL_PATH="${CBL_DIR}/cbl_no_backbone_acc.pt"

# --- Step 4: Train CBL (+ACC), freeze backbone với --tune_cbl_only ---
if [[ ! -f "$CBL_PATH" ]]; then
  info "Training CBL (ACC on, tune_cbl_only) for backbone='${BACKBONE}'..."
  python train_CBL.py \
    --automatic_concept_correction \
    --dataset="${DATASET}" \
    --backbone="${BACKBONE}" \
    --tune_cbl_only
else
  info "CBL model already exists at ${CBL_PATH}. Skipping train_CBL.py."
fi

# Kiểm tra checkpoint CBL
if [[ ! -f "$CBL_PATH" ]]; then
  err "Missing CBL checkpoint at ${CBL_PATH}."
  exit 2
fi

# --- Step 5: Train Final Linear Layer (Elastic-Net) ---
info "Training Final Linear (sparse &/or dense)..."
python train_FL.py \
  --cbl_path="${CBL_PATH}" \
  --dataset="${DATASET}" \
  --backbone="${BACKBONE}"