# src/data.py
from datasets import load_dataset, concatenate_datasets, DatasetDict, Value  # ← add Value

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

def _need_datasets():
    raise ImportError("`datasets` is required for data loading. Install with `pip install datasets`.")

def load_imdb():
    try:
        from datasets import load_dataset, DatasetDict, Value, Features
    except Exception:
        _need_datasets()
    imdb = load_dataset("imdb")
    feats = Features({"text": Value("string"), "label": Value("int64")})
    imdb_train = imdb["train"].cast(feats)
    imdb_test  = imdb["test"].cast(feats)
    return DatasetDict(train=imdb_train, test=imdb_test)

def load_tweeteval_binary():
    try:
        from datasets import load_dataset, DatasetDict, Value, Features
    except Exception:
        _need_datasets()
    ds = load_dataset("tweet_eval", "sentiment")
    def keep_binary(x): return x["label"] != 1
    def map_lbl(x): return {"text": x["text"], "label": 0 if x["label"] == 0 else 1}
    feats = Features({"text": Value("string"), "label": Value("int64")})
    train = ds["train"].filter(keep_binary).map(map_lbl, remove_columns=ds["train"].column_names).cast(feats)
    val   = ds["validation"].filter(keep_binary).map(map_lbl, remove_columns=ds["validation"].column_names).cast(feats)
    test  = ds["test"].filter(keep_binary).map(map_lbl, remove_columns=ds["test"].column_names).cast(feats)
    return DatasetDict(train=train, validation=val, test=test)

def make_binary_corpus(imdb_sample=20000, tweet_sample=20000, seed=42):
    try:
        from datasets import concatenate_datasets, DatasetDict
    except Exception:
        _need_datasets()
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
    train = train.class_encode_column("label")
    split = train.train_test_split(test_size=0.1, seed=seed, stratify_by_column="label")
    return DatasetDict(train=split["train"], validation=split["test"], test=test)


