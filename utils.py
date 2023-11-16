from typing import List, Optional,Tuple
from collections import defaultdict, Counter
import datasets
import transformers
import logging
import random
import numpy as np
import os
import torch
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import itertools

import matplotlib

logging.basicConfig()
LOG = logging.getLogger(__name__)

datasets.logging.set_verbosity_error()

RESULTS_DIR = (
    "results" if "RESULTS_DIR" not in os.environ else os.environ["RESULTS_DIR"]
)
print(f"Using results dir: {RESULTS_DIR}")
glue_datasets = ["sst2", "cola", "mrpc", "qnli", "rte", "wnli"]


def is_classification(dataset):
    ### name of original dataset should not contain _aug ###
    dataset_dict = {"amazon": 5, "sst2": 2, "cola": 2, "mrpc": 2, "qnli": 2, "rte": 2, 
    "wnli": 2, "yelp_polarity": 2}
    if dataset in dataset_dict.keys():
        return dataset_dict[dataset]
    if dataset.startswith(tuple([name+"_aug" for name in dataset_dict.keys()])):
        return dataset_dict[dataset.split("_aug")[0]]
    return -1


def model2hfname(model: str) -> str:
    return {
        "bert-tiny": "prajjwal1/bert-tiny",
        "bert-med": "prajjwal1/bert-medium",
        "small": "gpt2",
        "med": "gpt2-medium",
        "large": "gpt2-large",
        "full": "gpt2-xl",
        "gpt2-sm": "gpt2",
        "gpt2-med": "gpt2-medium",
        "gpt2-lg": "gpt2-large",
        "gpt2": "gpt2-xl",
        "neo": "EleutherAI/gpt-neo-2.7B",
        "sst2": "gchhablani/bert-base-cased-finetuned-sst2",
        "cola": "gchhablani/bert-base-cased-finetuned-cola",
        "mrpc": "gchhablani/bert-base-cased-finetuned-mrpc",
        "qnli": "gchhablani/bert-base-cased-finetuned-qnli",
        "rte": "gchhablani/bert-base-cased-finetuned-rte",
        "wnli": "gchhablani/bert-base-cased-finetuned-wnli",
        "imdb": "Wakaka/bert-finetuned-imdb",
        "yelp_polarity": "textattack/bert-base-uncased-yelp-polarity"
    }[model]


def dataset2hfname(dataset: str) -> str:
    return {
        "mnli": ("multi_nli",),
        "amazon": ("amazon_us_reviews", "Video_v1_00"),
        "cnn": ("cnn_dailymail", "3.0.0"),
        "math": ("math_qa",),
        "tos": ("ought/raft", "terms_of_service"),
        "xsum": ("xsum",),
        "babi": ("babi_qa", "en-valid-10k-qa1"),
    }[dataset]


def get_dataset(dataset: str, n_train: int, n_val: int = 100):
    if dataset == "cnn":
        n_train = 64
        d = datasets.load_dataset("cnn_dailymail", "3.0.0", split="train")
        filter_fn = lambda rows: [
            "VIDEO" not in a
            and len(a.split(" ")) < 110
            and len(a.split(" ")) > 35
            and len(s.split(" ")) < 25
            for a, s in zip(rows["article"], rows["highlights"])
        ]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        d = d.rename_columns({"article": "x", "highlights": "y"})

        def strip_target(row):
            y = row["y"]
            y = y.replace(" .", ".")
            if ". " in y:
                y = y[: y.index(". ")]
            if "\n" in y:
                y = y[: y.index("\n")]
            row["y"] = y
            return row

        d = d.map(strip_target)
        d = d.add_column("simple_y", d["y"])
        return d[:n_train], d[n_train : n_train + n_val]
    elif dataset == "trivia":
        n_train = 256
        d = datasets.load_dataset("trivia_qa", "rc.nocontext", split="train[:1%]")
        targets = [
            [a["normalized_value"]] + a["normalized_aliases"] for a in d["answer"]
        ]
        d = d.add_column("simple_y", [t[0] for t in targets])
        d = d.add_column("y", targets)
        d = d.rename_column("question", "x")
        offset = 0
        return (
            d[offset : offset + n_train],
            d[offset + n_train : offset + n_train + n_val],
        )
    elif dataset == "babi":
        n_train = 256
        d = datasets.load_dataset("babi_qa", "en-valid-10k-qa1", split="train")
        answer_idxs = []
        for story in d["story"]:
            for idx, answer in enumerate(story["answer"]):
                if answer:
                    answer_idxs.append(idx)
                    break

        perm = np.random.permutation(len(d["story"]))
        answers = [story["answer"][idx] for idx, story in zip(answer_idxs, d["story"])]
        stories = [
            " ".join(story["text"][: idx + 1])
            for idx, story in zip(answer_idxs, d["story"])
        ]

        answers = [answers[idx] for idx in perm]
        stories = [stories[idx] for idx in perm]
        data = {"x": stories, "y": answers, "simple_y": answers}
        d = datasets.Dataset.from_dict(data)
        return d[:n_train], d[n_train : n_train + n_val]
    elif dataset == "amazon":
        # d = datasets.load_dataset("amazon_us_reviews", "Video_v1_00")["train"]
        data_files = "data/amazon_reviews_us_Video_v1_00.csv"
        if not os.path.exists(data_files):
            data_files = "starter_code/data/amazon_reviews_us_Video_v1_00.csv"
        try:
            d = datasets.load_dataset("csv", data_files=data_files)["train"]
        except FileNotFoundError:
            print(
                "PLEASE DOWNLOAD THE AMAZON DATASET FROM https://drive.google.com/file/d/1RLCPCEvJVTvUbn-D426Avwg6hynSBgU3/view?usp=sharing AND PLACE IT IN data/amazon_reviews_us_Video_v1_00.csv"
            )
            exit(1)
        filter_fn = lambda rows: ["sex" not in r.lower() for r in rows["review_body"]]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        x = d["review_body"]
        y = [s - 1 for s in d["star_rating"]]
        train = defaultdict(lambda: [None] * 5 * n_train)
        val = defaultdict(lambda: [None] * 5 * n_val)
        counts = defaultdict(int)
        for idx in range(len(y)):
            c = counts[y[idx]]
            if c < n_train:
                train["x"][c * 5 + y[idx]] = x[idx]
                train["y"][c * 5 + y[idx]] = y[idx]
                counts[y[idx]] += 1
            elif c < n_train + n_val:
                val["x"][(c - n_train) * 5 + y[idx]] = x[idx]
                val["y"][(c - n_train) * 5 + y[idx]] = y[idx]
                counts[y[idx]] += 1
        return train, val
    elif dataset == "xsum":
        n_train = 256
        d = datasets.load_dataset("xsum", split="train")
        filter_fn = lambda rows: [
            len(a.split(" ")) + len(s.split(" ")) < 100
            for a, s in zip(rows["document"], rows["summary"])
        ]
        d = d.filter(filter_fn, batched=True, batch_size=None)
        d = d.rename_columns({"document": "x", "summary": "y"})
        d = d.add_column("simple_y", d["y"])
        return d[:n_train], d[n_train : n_train + n_val]

    elif dataset in glue_datasets+ ["yelp_polarity"] or dataset.startswith(tuple([i+"_aug" for i in glue_datasets])):
        if dataset in glue_datasets:
            d = datasets.load_dataset("glue", dataset)
        elif dataset.startswith(tuple([i+"_aug" for i in glue_datasets])):
            d = datasets.load_dataset('csv', data_files={"train": "datasets/"+dataset+"_train.csv", "validation":  "datasets/"+dataset+"_val.csv"})
        else:
            d = datasets.load_dataset(dataset)
        
        # Among subsets of the validation set corresponding to each label, get size of subset with min size
        n_val = min(list(Counter(d[list(d.keys())[1]]["label"]).values())) 

        # Perform similar validation for n_train
        n_train_check = min(list(Counter(d[list(d.keys())[0]]["label"]).values())) 
        if n_train>n_train_check:
            print("Reducing n_train to {} to ensure equal number of samples for each class in train set".format(n_train_check))
            n_train = n_train_check

        train = defaultdict(lambda: [None] * num_labels * n_train)
        val = defaultdict(lambda: [None] * num_labels * n_val)

        counts_train = defaultdict(int)
        counts_val = defaultdict(int)
        num_labels = is_classification(dataset)

        x_feature_name = list(d["train"][0].keys())[0]
        # if len(list(d["train"][0].keys()))>3:
        #     raise ValueError("Number of features longer than expected!!")

        x_train = [i[x_feature_name] for i in d["train"]]
        y_train = [i["label"] for i in d["train"]]

        x_val = [i[x_feature_name] for i in d[list(d.keys())[1]]]
        y_val = [i["label"] for i in d[list(d.keys())[1]]]

        for idx in range(len(y_train)):
            c = counts_train[y_train[idx]]
            if c < n_train:
                train["x"][c * num_labels + y_train[idx]] = x_train[idx]
                train["y"][c * num_labels + y_train[idx]] = y_train[idx]
                counts_train[y_train[idx]] += 1
        for idx in range(len(y_val)):
            c = counts_val[y_val[idx]]
            if c < n_val:
                val["x"][c * num_labels + y_val[idx]] = x_val[idx]
                val["y"][c * num_labels + y_val[idx]] = y_val[idx]
                counts_val[y_val[idx]] += 1 
        print(train["y"][:10], val["y"][:10])
        return train, val
    else:
        raise NotImplementedError(f"{dataset}")


def metric_for_dataset(dataset: str):
    return {
        "cnn": "rouge",
        "xsum": "rouge",
        "trivia": "exact match",
        "babi": "exact match",
        "amazon": "classification accuracy",
        "sst2": "classification accuracy",
        "cola": "classification accuracy",
        "mrpc": "classification accuracy",
        "qnli": "classification accuracy",
        "rte": "classification accuracy", 
        "wnli": "classification accuracy"
    }.get(dataset, "exact match")


def max_sampled_tokens_for_dataset(dataset: str) -> int:
    return {"cnn": 30, "trivia": 12, "babi": 6, "xsum": 30}.get(dataset, 30)


def early_stop_thresold(dataset: str):
    return {"cnn": 0.8, "trivia": 0.7, "babi": 0.9, "amazon": 0.75, "xsum": 0.55}.get(dataset, 0.8)


def get_prefix(dataset: str):
    input_prefix = {}.get(dataset, "")
    label_prefix = {"trivia": " In the", "babi": " In the"}.get(dataset, " TL;DR:")
    label_suffix = {"trivia": ".", "babi": "."}.get(dataset, "")
    
    return input_prefix, label_prefix, label_suffix


################ MODEL UTILS: Most likely no need to change ################

class IrisColormap(matplotlib.colors.ListedColormap):
    """Official IRIS lab plotting color palette. Palette author: Chelsea Finn."""

    def __init__(self, N: Optional[int] = None):
        """See matplotlib.colors.Colormap for N argument docs."""
        hex_colors = ["#FF6150", "#134E6F", "#1AC0C6", "#FFA822", "#DEE0E6", "#091A29"]

        rgb_colors = [matplotlib.colors.to_rgb(c) for c in hex_colors]
        super().__init__(rgb_colors, name="iris", N=N)


def get_model_and_tokenizer(model: str, Cls, **model_kwargs):
    hf_model_name = model2hfname(model)

    m = Cls.from_pretrained(hf_model_name, **model_kwargs)
    if isinstance(m, transformers.GPT2LMHeadModel):
        m.transformer.gradient_checkpointing_enable()

    tok = transformers.AutoTokenizer.from_pretrained(hf_model_name)

    if tok.pad_token_id is None:
        if Cls == transformers.AutoModelForCausalLM:
            tok.pad_token = tok.eos_token
        else:
            print("Adding pad token to tokenizer")
            tok.add_special_tokens({"pad_token": "[PAD]"})
            tok.pad_token = "[PAD]"
    return m, tok


def stop_tokens(tokenizer, stop_string: str = ".") -> List[int]:
    tokens = []
    for idx in range(len(tokenizer)):
        if tokenizer.decode(idx) == stop_string:
            tokens.append(idx)
    return tokens


def fix_random_seeds(seed=123, set_system=True, set_torch=True):
    """
    Fix random seeds for reproducibility.
    Parameters
    ----------
    seed : int
        Random seed to be set.
    set_system : bool
        Whether to set `np.random.seed(seed)` and `random.seed(seed)`
    set_torch : bool
        Whether to set `torch.manual_seed(seed)`
    """
    # set system seed
    if set_system:
        random.seed(seed)
        np.random.seed(seed)

    # set torch seed
    if set_torch:
        torch.manual_seed(seed)


# def plot_ft(models, datasets, ks, modes, output_path: str):
#     data = defaultdict(lambda: defaultdict(list))
#     question = "ft"

#     x_vals = set()
#     for dataset in datasets:
#         for model, mode in itertools.product(models, modes):
#             for k in ks:
#                 fn = "_".join([model, dataset, str(k), mode])
#                 id_ = "_".join([model, dataset, mode])
#                 with open(f"{RESULTS_DIR}/{question}/{fn}.json", "r") as f:
#                     score = json.load(f)["metric"]
#                     data[id_]["x"].append(k)
#                     x_vals.add(k)
#                     data[id_]["y"].append(score)

#         for k, v in data.items():
#             plt.plot(v["x"], v["y"], label=k)

#     if max(x_vals) > 4:
#         plt.xscale("symlog")
#     ax = plt.gca()
#     ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
#     ax.xaxis.set_ticks(sorted(x_vals))
#     plt.legend()
#     plt.title(" & ".join(datasets))
#     plt.ylabel("/".join([metric_for_dataset(dataset) for dataset in datasets]))
#     plt.xlabel("Number of support examples")
#     # plt.show()
#     plt.savefig(output_path, bbox_inches="tight")


def plot_ft(models, datasets, ks, modes, output_path: str):
    data = defaultdict(lambda: defaultdict(list))
    question = "ft"

    x_vals = set()
    k = ks[0]
    model = models[0]
    for dataset in datasets:
        for mode in modes:
            fn = "_".join([model, dataset, str(k), mode])
            id_ = "_".join([model, dataset])
            with open(f"{RESULTS_DIR}/{question}/{fn}.json", "r") as f:
                score = json.load(f)["metric"]
                if mode=="all":
                    mode = "-1"
                data[id_]["x"].append(int(mode))
                x_vals.add(k)
                data[id_]["y"].append(score)
    
    for k, v in data.items():
        print(v["x"], v["y"], k)
        plt.plot(v["x"], v["y"], label=k)

    # if max(x_vals) > 4:
    #     plt.xscale("symlog")
    # ax = plt.gca()
    # ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    # ax.xaxis.set_ticks(sorted(x_vals))
    plt.legend()
    plt.title("{} model fine-tuned on augmented datasets".format(model, dataset))
    plt.ylabel("Classification Accuracy")
    plt.xlabel("Fine-tuned layer number")
    # plt.show()
    plt.savefig(output_path, bbox_inches="tight")


def do_sample(
    model: transformers.GPT2LMHeadModel,
    input_ids: torch.Tensor,
    stop_tokens: List[int],
    max_tokens: int,
) -> List[int]:
    """
    Sample from the model using the given input_ids as a prefix until we either
    hit the stop token or we have sampled max_tokens tokens.
    Args:
        model: A transformers.PreTrainedModel that we will sample from.
        input_ids: An integer tensor of shape [1, prefix_len]
        stop_tokens: A list of token ids that indicates that we should stop sampling (e.g., a period)
        max_tokens: Stop sampling if we've sampled this many tokens

    Returns:
        The sampled tokens (a python list of ints/zero-dim tensors), not including the input_ids prefix
          OR the stop token (if we hit the stop token before max_tokens)
    """
    sampled_tokens = []
    with torch.inference_mode():
        model_input = input_ids
        past_key_values = None
        for i in range(max_tokens):
            outputs = model(
                model_input, past_key_values=past_key_values, use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1)
            if next_token.item() in stop_tokens:
                break
            sampled_tokens.append(next_token.item())
            model_input = next_token.unsqueeze(-1)
    return sampled_tokens


def add_prefixes(x: List[str], y: List[str], dataset: str) -> Tuple[List[str], List[str]]:
    input_prefix, label_prefix, label_suffix = get_prefix(dataset)
    x = [input_prefix + x_.replace("\n", " ") + label_prefix for x_ in x]
    y = [" " + y_.replace("\n", " ") + label_suffix for y_ in y]

    return x, y