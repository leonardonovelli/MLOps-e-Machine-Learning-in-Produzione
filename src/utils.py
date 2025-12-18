# push_to_hf.py
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

def push_to_hf(local_dir):
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token is None:
        raise ValueError("HF_TOKEN non trovato.")

    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=os.environ.get("HF_REPO_ID"),
        repo_type="model",
        token=hf_token,
        path_in_repo=""
    )
    print("Modello pushato su HuggingFace Hub!")

if __name__ == "__main__":
    push_to_hf("model_data")
