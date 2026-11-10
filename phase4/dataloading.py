import json

def load_rr():
    with open("/Users/kabeersinghal/CAPSTONE/phase4/local_tokenized_train.json","r", encoding="utf-8") as f:
        train = json.load(f)
    with open("/Users/kabeersinghal/CAPSTONE/phase4/local_tokenized_dev.json","r", encoding="utf-8") as f:
        dev = json.load(f)
    with open("/Users/kabeersinghal/CAPSTONE/phase4/local_tokenized_test.json","r", encoding="utf-8") as f:
        test = json.load(f)

    return train, dev, test