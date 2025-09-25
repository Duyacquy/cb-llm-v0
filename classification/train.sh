safe_dataset="${1//\//_}"
safe_model="${3//\//_}"

# Determine concept embedding model folder
concept_base="mpnet_acs"
if [[ "$2" == *"simcse"* ]]; then
    concept_base="simcse_acs"
elif [[ "$2" == *"angle"* ]]; then
    concept_base="angle_acs"
fi

concept_dir="${concept_base}/${safe_dataset}"
train_npy="${concept_dir}/concept_labels_train.npy"
val_npy="${concept_dir}/concept_labels_val.npy"

# Check if .npy files exist
if [[ ! -f "$train_npy" || ! -f "$val_npy" ]]; then
    python get_concept_labels.py --dataset="${1}" --concept_text_sim_model="${2}"
else
    echo "Concept label .npy files already exist. Skipping get_concept_labels.py."
fi

# Define the output CBL model path
cbl_model_path="${concept_dir}/${safe_model}_cbm/cbl_no_backbone_acc.pt"

# Check if the CBL model already exists
if [[ ! -f "$cbl_model_path" ]]; then
    python train_CBL.py --automatic_concept_correction --dataset="${1}" --backbone="${3}" --tune_cbl_only
else
    echo "CBL model already exists at $cbl_model_path. Skipping train_CBL.py."
fi

# Continue with training final layer
python train_FL.py --cbl_path="$cbl_model_path" --dataset="${1}" --backbone="${3}"
python print_concept_contributions.py --cbl_path="$cbl_model_path" --dataset="${1}"
