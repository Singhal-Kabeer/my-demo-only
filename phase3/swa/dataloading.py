from huggingface_hub import login
from huggingface_hub import hf_hub_download
import json

def load_rr():
    HF_TOKEN = "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"
    login(token=HF_TOKEN)

    print("Hugging Face login successful.")

    HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"

    LABEL_VOCAB = "Phase_3_RR/Train/label_vocab.json"

    CL_TRAIN = "Phase_3_RR/Train/CL_train.json"
    IT_TRAIN = "Phase_3_RR/Train/IT_train.json"
    CL_DEV = "Phase_3_RR/Train/CL_dev.json"
    IT_DEV = "Phase_3_RR/Train/IT_dev.json"
    CL_TEST = "Phase_3_RR/Test/CL_test.json"
    IT_TEST = "Phase_3_RR/Test/IT_test.json"

    LABEL_VOCAB = hf_hub_download(repo_id=HF_DATASET_REPO, filename=LABEL_VOCAB, repo_type="dataset", token=HF_TOKEN)
    CL_TRAIN = hf_hub_download(repo_id=HF_DATASET_REPO, filename=CL_TRAIN, repo_type="dataset", token=HF_TOKEN)
    CL_DEV = hf_hub_download(repo_id=HF_DATASET_REPO, filename=CL_DEV, repo_type="dataset", token=HF_TOKEN)
    CL_TEST = hf_hub_download(repo_id=HF_DATASET_REPO, filename=CL_TEST, repo_type="dataset", token=HF_TOKEN)
    IT_TRAIN = hf_hub_download(repo_id=HF_DATASET_REPO, filename=IT_TRAIN, repo_type="dataset", token=HF_TOKEN)
    IT_DEV = hf_hub_download(repo_id=HF_DATASET_REPO, filename=IT_DEV, repo_type="dataset", token=HF_TOKEN)
    IT_TEST = hf_hub_download(repo_id=HF_DATASET_REPO, filename=IT_TEST, repo_type="dataset", token=HF_TOKEN)

    with open(LABEL_VOCAB,"r", encoding="utf-8") as f:
        label_vocab = json.load(f)
    with open(CL_TRAIN,"r", encoding="utf-8") as f:
        cl_train = json.load(f)
    with open(CL_DEV,"r", encoding="utf-8") as f:
        cl_dev = json.load(f)
    with open(CL_TEST,"r", encoding="utf-8") as f:
        cl_test = json.load(f)
    with open(IT_TRAIN,"r", encoding="utf-8") as f:
        it_train = json.load(f)
    with open(IT_DEV,"r", encoding="utf-8") as f:
        it_dev = json.load(f)
    with open(IT_TEST,"r", encoding="utf-8") as f:
        it_test = json.load(f)

    return label_vocab, cl_train, cl_dev, cl_test, it_train, it_dev, it_test