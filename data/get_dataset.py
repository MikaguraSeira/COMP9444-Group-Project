from huggingface_hub import snapshot_download
from datasets import load_dataset

# Set local_dir to current directory "."
local_dir = "."
snapshot_download(
    repo_id="gymprathap/Breast-Cancer-Ultrasound-Images-Dataset", 
    repo_type="dataset", 
    local_dir=local_dir
)

# Load the dataset explicitly using the "imagefolder" builder
ds = load_dataset("imagefolder", data_dir=local_dir)

print(ds)