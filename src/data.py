# src/data.py
from datasets import load_dataset, concatenate_datasets, DatasetDict, Value  # ← add Value

def load_imdb():
    return load_dataset("imdb")

def load_tweeteval_binary():
    ds = load_dataset("tweet_eval", "sentiment")
    # drop neutral (1) and map 0->0 (NEG), 2->1 (POS)
    def keep_binary(x): return x["label"] != 1
    def map_lbl(x):
        x["label"] = 0 if x["label"] == 0 else 1
        return x

    train = ds["train"].filter(keep_binary).map(map_lbl)
    val   = ds["validation"].filter(keep_binary).map(map_lbl)
    test  = ds["test"].filter(keep_binary).map(map_lbl)

    # ← force label to simple integer type so it aligns with IMDb after casting
    train = train.cast_column("label", Value("int64"))
    val   = val.cast_column("label", Value("int64"))
    test  = test.cast_column("label", Value("int64"))

    return DatasetDict(train=train, validation=val, test=test)

def make_binary_corpus(imdb_sample=20000, tweet_sample=20000, seed=42):
    imdb = load_dataset("imdb")
    tw   = load_dataset("tweet_eval", "sentiment")

    # --- TweetEval → drop neutral (1) and map 0→0, 2→1 ---
    def keep_binary(x): return x["label"] != 1
    def map_lbl(x): return {"text": x["text"], "label": 0 if x["label"] == 0 else 1}

    tw_train = tw["train"].filter(keep_binary).map(map_lbl, remove_columns=tw["train"].column_names)
    tw_test  = tw["test"].filter(keep_binary).map(map_lbl,  remove_columns=tw["test"].column_names)

    # --- cast BOTH sources to the SAME simple schema so concatenation works ---
    imdb_train = imdb["train"].cast_column("label", Value("int64"))
    imdb_test  = imdb["test"].cast_column("label",  Value("int64"))
    tw_train   = tw_train.cast_column("label",      Value("int64"))
    tw_test    = tw_test.cast_column("label",       Value("int64"))

    # --- build combined splits ---
    train = concatenate_datasets([
        imdb_train.shuffle(seed=seed).select(range(min(imdb_sample, len(imdb_train)))),
        tw_train.shuffle(seed=seed).select(range(min(tweet_sample, len(tw_train)))),
    ])
    test = concatenate_datasets([
        imdb_test.shuffle(seed=seed).select(range(5000)),
        tw_test.shuffle(seed=seed).select(range(5000)),
    ])

    # --- stratified 90/10 split needs ClassLabel -> re-encode just before splitting ---
    train = train.class_encode_column("label")
    split = train.train_test_split(test_size=0.1, seed=seed, stratify_by_column="label")

    return DatasetDict(train=split["train"], validation=split["test"], test=test)

# Normalization functions for text preprocessing
import re
_URL    = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_HANDLE = re.compile(r"@\w+")
_HASHTAG= re.compile(r"#(\w+)")
_WS     = re.compile(r"\s+")

def clean_for_transformers(t: str) -> str:
    t = t.strip()
    t = _URL.sub(" URL ", t)
    t = _HANDLE.sub(" USER ", t)
    t = _WS.sub(" ", t)
    return t

def clean_for_tfidf(t: str) -> str:
    t = t.strip().lower()
    t = _URL.sub(" url ", t)
    t = _HANDLE.sub(" user ", t)
    t = _HASHTAG.sub(r"\1", t)
    t = _WS.sub(" ", t)
    return t

