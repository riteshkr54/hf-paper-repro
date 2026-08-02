import os

# Canonical data cache on this machine (pre-seeded; falls back to ./data)
DATA_ROOT = os.environ.get("DAPPR_DATA_ROOT", "/home/avesha/.cache/dappr-data")
TORCH_ROOT = os.path.join(DATA_ROOT, "torch")

if not os.path.isdir(DATA_ROOT):
    DATA_ROOT = os.path.join(os.getcwd(), "data")
    TORCH_ROOT = os.path.join(DATA_ROOT, "torch")

os.makedirs(TORCH_ROOT, exist_ok=True)

# Official-repo layout: <DATA_ROOT>/datasets/<name>, <DATA_ROOT>/splits/<ds>/{train,val,test}.json
DATASETS_DIR = os.path.join(DATA_ROOT, "datasets")
SPLITS_DIR = os.path.join(DATA_ROOT, "splits")
os.makedirs(DATASETS_DIR, exist_ok=True)
