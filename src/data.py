from datasets import load_dataset, concatenate_datasets, DatasetDict

def load_imdb():
    return load_dataset("imdb")

def load_tweeteval_binary():
    ds = load_dataset("tweet_eval", "sentiment")
    # drop neutral (1) and map 0->0 (NEG), 2->1 (POS)
    def keep_binary(x): return x["label"] != 1
    def map_lbl(x): x["label"] = 0 if x["label"] == 0 else 1; return x
    train = ds["train"].filter(keep_binary).map(map_lbl)
    val = ds["validation"].filter(keep_binary).map(map_lbl)
    test = ds["test"].filter(keep_binary).map(map_lbl)
    return DatasetDict({
        "train": train,
        "validation": val,
        "test": test
    })