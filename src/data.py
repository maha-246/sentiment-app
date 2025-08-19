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

def make_binary_corpus(imdb_sample=20000, tweet_sample=20000, seed=42):
    imdb = load_imdb()
    tw   = load_tweeteval_binary()


    train = concatenate_datasets([
        imdb["train"].shuffle(seed=seed).select(range(min(imdb_sample, len(imdb["train"])))),
        tw["train"].shuffle(seed=seed).select(range(min(tweet_sample, len(tw["train"])))),
    ])

    test = concatenate_datasets([
        imdb["test"].shuffle(seed=seed).select(range(5000)),
        tw["test"].shuffle(seed=seed).select(range(5000)),
    ])

    split = train.train_test_split(test_size=0.1, seed=seed, stratify_by_column="label")
    return DatasetDict(train=split["train"], validation=split["test"], test=test)
