from typing import List, Iterable
import argparse
import torch
import transformers
import torch.nn as nn

try:
    import utils
except ModuleNotFoundError:
    from . import utils
import copy
import numpy as np
import os
import json
import itertools

try:
    from lora import LoRALayerWrapper
except ModuleNotFoundError:
    from .lora import LoRALayerWrapper
import tqdm
from rouge_score import rouge_scorer
import random

parser = argparse.ArgumentParser()
parser.add_argument("--task")
parser.add_argument("--model")
parser.add_argument("--dataset")
parser.add_argument("--k")
parser.add_argument(
    "--mode", default="all", type=str, help="all, last, first, middle, integer"
)
parser.add_argument("--debug", action="store_true")
parser.add_argument("--repeats", default=1, type=int)
parser.add_argument("--device", default="cuda")
parser.add_argument("--plot_name", default="plot.png")
parser.add_argument("--save_model", default=False, action="store_true")
parser.add_argument("--evaluate", default=False, action="store_true")
parser.add_argument(
    "--num_layers",
    help="number of layers to train",
)
parser.add_argument(
    "--lora_mode",default="all",type=str,help="all, last, first, middle, integer, none, same"
)
parser.add_argument(
    '--lora_k',
)
args = parser.parse_args()


if os.environ.get("FORCE_DEVICE", False):
    DEVICE = torch.device(os.environ["FORCE_DEVICE"])
else:
    DEVICE = torch.device(args.device)

print("Fine-tuning using device: ", DEVICE)

def add_lora(model: nn.Module, mode: str, lora_k: int, num_layers: int = 1) -> nn.Module:
    """
    Add the LORA wrapper to the model
    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)
    Returns:
        new model with LORA wrapper
    """
    if model.__class__.__name__ == "GPT2LMHeadModel":
        layers = model.transformer.h
    elif model.__class__.__name__ == "BertLMHeadModel" or model.__class__.__name__ == "BertForSequenceClassification":
        layers = model.bert.encoder.layer
    else:
        raise ValueError(
            f"Unrecognized model class {model.__class__.__name__}. Check add_lora function in ft.py"
        )
    layers_to_add = []
    if mode == "all":
        layers_to_add = layers
    elif mode == "last":
        layers_to_add = layers[-num_layers:]
    elif mode == "first":
        layers_to_add = layers[: num_layers]
    elif mode == "middle":
        num_layers = len(layers)
        layers_to_add = layers[(num_layers+1) // 2 - (num_layers+1) // 2: (num_layers+1) // 2 + num_layers // 2]
    elif mode == "none":
        layers_to_add = []
    elif mode.isnumeric():
        layers_to_add = layers[int(mode):int(mode)+num_layers]
    else:
        raise ValueError(f"Unrecognized fine-tuning mode {mode}")
    for m in layers_to_add:
            if model.__class__.__name__ == "GPT2LMHeadModel":
                m.mlp.c_fc = LoRALayerWrapper(m.mlp.c_fc, lora_k)
                m.mlp.c_proj = LoRALayerWrapper(m.mlp.c_proj, lora_k)
                m.attn.c_attn = LoRALayerWrapper(m.attn.c_attn, lora_k)
                m.attn.c_proj = LoRALayerWrapper(m.attn.c_proj, lora_k)
            else:
                m.attention.self.query = LoRALayerWrapper(m.attention.self.query, lora_k)
                m.attention.self.key = LoRALayerWrapper(m.attention.self.key, lora_k)
                m.attention.self.value = LoRALayerWrapper(m.attention.self.value, lora_k)
                m.attention.output.dense = LoRALayerWrapper(m.attention.output.dense, lora_k)
                m.intermediate.dense = LoRALayerWrapper(m.intermediate.dense, lora_k)
                m.output.dense = LoRALayerWrapper(m.output.dense, lora_k)

    return layers_to_add, model

def parameters_to_fine_tune(model: nn.Module, mode: str, num_layers=1, lora_layers = None) -> Iterable[nn.Parameter]:
    """
    Select the parameters in `model` that should be fine-tuned in mode `mode`.
    Args:
      model: the model we're fine-tuning
      mode: the fine-tuning mode we're using; may be 'all', 'last', 'first',
        'middle', or 'loraN' (where N is an integer)
    Returns:
      A list of nn.Parameters of `model` that should be fine-tuned in the given
        fine-tuning mode.
    """
    if model.__class__.__name__ == "GPT2LMHeadModel":
        layers = model.transformer.h
    elif model.__class__.__name__ == "BertLMHeadModel" or model.__class__.__name__ == "BertForSequenceClassification":
        layers = model.bert.encoder.layer
    else:
        raise ValueError(
            f"Unrecognized model class {model.__class__.__name__}. Check parameters_to_fine_tune function in ft.py"
        )
    parameters_to_fine_tune: List[nn.Parameter] = None
    if mode == "all":
        layers_to_fine_tune = layers
    elif mode == "last":
        layers_to_fine_tune = layers[-num_layers:]
    elif mode == "first":
        layers_to_fine_tune = layers[: num_layers]
    elif mode == "middle":
        num_layers = len(layers)
        layers_to_fine_tune = layers[(num_layers+1) // 2 - (num_layers+1) // 2: (num_layers+1) // 2 + num_layers // 2]
    elif mode.isnumeric():
        layers_to_fine_tune = layers[int(mode):int(mode)+num_layers]
    elif mode.startswith("lora"):
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized fine-tuning mode {mode}")
    
    layers_to_fine_tune = set(layers_to_fine_tune)
    lora_layers = set(lora_layers)
    parameters_to_fine_tune = []
    
    non_lora_layers = layers_to_fine_tune - lora_layers
    lora_layers = layers_to_fine_tune - non_lora_layers
    
    for layer in non_lora_layers:
        parameters_to_fine_tune.extend(list(layer.parameters()))
    
    for layer in lora_layers:
        for m in layer.modules():
            if isinstance(m, LoRALayerWrapper):
                parameters_to_fine_tune.extend([m.lora_A, m.lora_B])

    return parameters_to_fine_tune


def get_loss(unnormalized_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss for either sequence classification or generation.
    Args:
      unnormalized_logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor
        of *UNNORMALIZED* logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.

    Returns:
      A zero-dim tensor (scalar) representing the average cross-entropy loss over all batch
        elements (and sequence timesteps, if applicable)
    """
    loss: torch.Tensor = None
    if unnormalized_logits.dim() == 2:
        loss = nn.functional.cross_entropy(unnormalized_logits, targets)
    elif unnormalized_logits.dim() == 3:
        unnormalized_logits = unnormalized_logits[:, :-1, :]
        targets = targets[:, 1:]
        loss = nn.functional.cross_entropy(
            unnormalized_logits.view(-1, unnormalized_logits.shape[-1]),
            targets.view(-1),
            ignore_index=-100,
        )
    else:
        raise ValueError(
            f"Logits should either be 2-dim (for classification) or 3-dim (for generation); got {unnormalized_logits.dim()}"
        )

    assert (
        loss is not None and loss.dim() == 0
    ), "Loss should be a scalar tensor. It should be the mean loss over the batch"
    return loss


def get_acc(unnormalized_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the exact match accuracy for either sequence classification or generation. i.e.,
      the fraction of predictions for which the most likely class/token equals the target.
    Args:
      unnormalized_logits: a 2D [batch_size, n_classes] (for classification) or 3D
        [batch_size, sequence_length, vocab_size] (for generation) tensor of logits
      targets: a 1D [batch_size] (for classification) or 2D [batch_size, sequence_length]
        (for generation) tensor of target indices. For the generation case, may contain
        -100 in some positions, meaning that the loss for this timestep should be ignored.
    Returns:
      A *scalar* representing the average exact-match accuracy over all non-masked batch
        elements (and sequence timesteps, if applicable)
    """
    accuracy: torch.Tensor = None
    if unnormalized_logits.dim() == 2:
        accuracy = (unnormalized_logits.argmax(dim=-1) == targets).float().mean()
    elif unnormalized_logits.dim() == 3:
        unnormalized_logits = unnormalized_logits[:, :-1, :]
        targets = targets[:, 1:]
        accuracy = (unnormalized_logits.argmax(dim=-1) == targets).float()
        accuracy = accuracy[targets != -100].mean()
    else:
        raise ValueError(
            f"Logits should either be 2-dim (for classification) or 3-dim (for generation); got {unnormalized_logits.dim()}"
        )

    assert (
        accuracy is not None and accuracy.dim() == 0
    ), "Accuracy should be a scalar tensor. It should be the mean accuracy over the batch"
    return accuracy.item()


def get_performance_metric(
    predictions, targets, metric: str
) -> float:
    if metric == "rouge":
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = []
        for p, t in zip(predictions, targets):
            score = scorer.score(p, t)["rouge1"].fmeasure
            scores.append(score)
        return sum(scores) / len(scores)
    elif metric == "exact match":
        if isinstance(targets[0], str):
            return sum(
                [p.strip() == t.strip() for p, t in zip(predictions, targets)]
            ) / len(predictions)
        else:

            def _normalize(prediction):
                if prediction.endswith("Q"):
                    prediction = prediction[:-1]
                elif "Q:" in prediction:
                    prediction = prediction[: prediction.index("Q:")]
                return prediction.strip(". ").lower()

            normalized = [_normalize(p) for p in predictions]

            def contains(key, candidates):
                for c in candidates:
                    if key in c:
                        return True
                return False

            return sum([contains(n, t) for n, t in zip(normalized, targets)]) / len(
                normalized
            )
    elif metric == "classification accuracy":
        return get_acc(predictions, targets)
    else:
        raise NotImplementedError()


def update_relative_gradient_norm(model, relative_gradient_norm_list):
    if model.__class__.__name__ == "GPT2LMHeadModel":
        layers = model.transformer.h
    elif model.__class__.__name__ == "BertLMHeadModel" or model.__class__.__name__ == "BertForSequenceClassification":
        layers = model.bert.encoder.layer

    num_layers = len(layers)
    new_list = [max([torch.norm(i.grad)/torch.norm(i.data) for i in model.bert.encoder.layer[j].parameters()]) for j in range(12)]
    if relative_gradient_norm_list is None:
        relative_gradient_norm_list = new_list
    else:
        relative_gradient_norm_list = [relative_gradient_norm_list[i]+new_list[i] for i in range(len(new_list))]
    # print(new_list)
    return relative_gradient_norm_list

def update_snr(model, relative_gradient_norm_list):
    if model.__class__.__name__ == "GPT2LMHeadModel":
        layers = model.transformer.h
    elif model.__class__.__name__ == "BertLMHeadModel" or model.__class__.__name__ == "BertForSequenceClassification":
        layers = model.bert.encoder.layer

    num_layers = len(layers)
    new_list = [max([torch.square(torch.sum(i.grad))/torch.sum(torch.square(i.grad)) for i in model.bert.encoder.layer[j].parameters()]) for j in range(12)]
    if relative_gradient_norm_list is None:
        relative_gradient_norm_list = new_list
    else:
        relative_gradient_norm_list = [relative_gradient_norm_list[i]+new_list[i] for i in range(len(new_list))]
    # print(relative_gradient_norm_list)
    return relative_gradient_norm_list

def ft_classification(model, tok, x, y, mode, num_layers, lora_mode, lora_k, batch_size=8):
    model = copy.deepcopy(model)
    print("Size of training set =", len(x), len(y))

    if lora_mode == "same":
        lora_mode = mode
    lora_layers, _ = add_lora(model, lora_mode, lora_k, num_layers)
    # if mode.startswith("lora"):
    #     for m in model.transformer.h:
    #         m.mlp.c_fc = LoRALayerWrapper(m.mlp.c_fc, int(mode[4:]))
    #         m.mlp.c_proj = LoRALayerWrapper(m.mlp.c_proj, int(mode[4:]))

    model.to(DEVICE)
    relative_gradient_norm_list = None
    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode, num_layers, lora_layers), lr=1e-4)
    all_x = tok(
        x, return_tensors="pt", padding=True, truncation=True, max_length=100
    ).to(DEVICE)
    all_y = torch.tensor(y, device=DEVICE)
    pbar = tqdm.tqdm(range(1000))
    for step in pbar:
        batch = np.random.randint(0, len(x), batch_size)
        x_ = tok(
            [x[i] for i in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=100,
        ).to(DEVICE)
        y_ = torch.tensor([y[i] for i in batch], device=DEVICE)
        logits = model(**x_).logits
        loss = get_loss(logits, y_)
        loss.backward()
        # relative_gradient_norm_list = update_snr(model, relative_gradient_norm_list)
        optimizer.step()
        optimizer.zero_grad()
        if args.debug:
            break

        if step % 100 == 0: # Time-consuming part - increase the number to reduce frequency of validation
            with torch.inference_mode():
                total_acc = get_acc(model(**all_x).logits, all_y)
            pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")
            if total_acc >0.995:
                break

    print(relative_gradient_norm_list)
    return model

def tokenize_gpt2_batch(
    tokenizer: transformers.GPT2Tokenizer, x: List[str], y: List[str]
):
    """
    Tokenization step for a batch of examples for GPT-2.

    Args:
        tokenizer: a GPT2Tokenizer that you can call and receive a dictionary of:
          - input_ids: a list (or tensor) of token ids
          - attention_mask: a list (or tensor) of 1s and 0s indicating which tokens
              are padding (if you requested padding and tensors from the tokenizer)
        x: a list of strings, each of which is the input for a single example
        y: a list of strings, each of which is a *target* for a single example

    Returns:
        A dictionary with the following keys:
            - input_ids: a tensor of shape [batch_size, sequence_length]
                containing the token ids
            - attention_mask: a tensor of shape [batch_size, sequence_length]
                containing 1s and 0s indicating which tokens are padding
            - labels: a tensor of shape [batch_size, sequence_length] containing
                the target token ids, with -100 for non-target tokens (i.e., the
                tokens in the input part of each example or padding tokens)
    """
    combined_sequences = tokenizer(
        [x_ + y_ for x_, y_ in zip(x, y)], return_tensors="pt", padding=True
    )
    combined_sequences["labels"] = combined_sequences["input_ids"].clone()
    combined_sequences["labels"][
        combined_sequences["labels"] == tokenizer.pad_token_id
    ] = -100

    tokenized_x = tokenizer(x)
    for i in range(len(combined_sequences["labels"])):
        combined_sequences["labels"][i][: len(tokenized_x["input_ids"][i])] = -100
    combined_sequences = combined_sequences.to(DEVICE)
    return combined_sequences


def ft_generation(model, tok, x, y, mode, dataset, num_layers, lora_mode, lora_k, batch_size=8, grad_accum=8):
    x, y = utils.add_prefixes(x, y, dataset)

    model = copy.deepcopy(model)
    if lora_mode == "same":
        lora_mode = mode
    lora_layers, _ = add_lora(model, lora_mode, lora_k, num_layers)

    # if mode.startswith("lora"):
    #     # if model is GPT2
    #     if hasattr(model, "transformer"):
    #         for m in model.transformer.h:
    #             m.mlp.c_fc = LoRALayerWrapper(m.mlp.c_fc, int(mode[4:]))
    #             m.mlp.c_proj = LoRALayerWrapper(m.mlp.c_proj, int(mode[4:]))
    #             m.attn.c_attn = LoRALayerWrapper(m.attn.c_attn, int(mode[4:]))

    model.to(DEVICE)

    optimizer = torch.optim.Adam(parameters_to_fine_tune(model, mode, num_layers, lora_layers), lr=2e-5)
    all_both = tokenize_gpt2_batch(tok, x, y)
    max_n = len(x) * 10
    pbar = tqdm.tqdm(range(max_n))
    idxs = []
    for step in pbar:
        model.train()

        if len(idxs) < batch_size // grad_accum:
            idxs = list(range(len(x)))
            random.shuffle(idxs)
        batch_idxs = idxs[: batch_size // grad_accum]
        idxs = idxs[batch_size // grad_accum :]

        minibatch_x = [x[i] for i in batch_idxs]
        minibatch_y = [y[i] for i in batch_idxs]

        minibatch = tokenize_gpt2_batch(tok, minibatch_x, minibatch_y)

        model_output = model(**minibatch, use_cache=False)

        loss = get_loss(model_output.logits, minibatch["labels"]) / grad_accum
        loss.backward()
        if step % grad_accum == 0 and step != 0:
            optimizer.step()
            optimizer.zero_grad()
            
        if step % (grad_accum * 5) == 0:
            with torch.inference_mode():
                model.eval()
                accs = []
                for idx in range(len(list(all_both.values())[0])):
                    d = {k: v[idx: idx + 1] for k, v in all_both.items()}
                    acc = get_acc(model(**d).logits, d["labels"])
                    accs.append(acc)
                total_acc = sum(accs) / len(accs)
                pbar.set_description(f"Fine-tuning acc: {total_acc:.04f}")

            if total_acc >= utils.early_stop_thresold(dataset):
                print("Early stopping!")
                break
    return model


def eval(model, tok, val_data):
    x = tok(
        val_data["x"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=100,
    ).to(DEVICE)
    y = torch.tensor(val_data["y"], device=DEVICE)
    with torch.inference_mode():
        logits = model(**x).logits
    return get_acc(logits, y)


def run_ft(
    models: List[str],
    datasets: List[str],
    ks: List[int],
    modes: List[str],
    num_layers_: List[int] = [1],
    lora_k_: List[int] = [8],
    lora_mode_: List[str] = ["all"],
    n_val: int = 200,
):
    results = {}
    for dataset in datasets:
        utils.fix_random_seeds()
        if args.debug:
            n_val = 1
        train, val = utils.get_dataset(dataset, max(ks), n_val=n_val)
        for model_name, mode in itertools.product(models, modes):
            utils.fix_random_seeds()
            if utils.is_classification(dataset) != -1:
                model, tokenizer = utils.get_model_and_tokenizer(
                    model_name,
                    transformers.AutoModelForSequenceClassification,
                    num_labels=utils.is_classification(dataset),
                )
            else:
                model, tokenizer = utils.get_model_and_tokenizer(
                    model_name, transformers.AutoModelForCausalLM
                )
            stop_tokens = utils.stop_tokens(tokenizer)

            for num_layers, lora_mode, lora_k, k in itertools.product(num_layers_, lora_mode_, lora_k_, ks):
                utils.fix_random_seeds()
                for repeat in range(args.repeats):
                    if repeat > 0:
                        print(f"Beginning repeat #{repeat}")
                    if utils.is_classification(dataset) != -1:
                        if args.evaluate:
                            fine_tuned = model.to(DEVICE)
                        else:
                            print(f"Fine-tuning {model_name} on {dataset} with k={k} and mode={mode}")
                            fine_tuned = ft_classification(
                                model,
                                tokenizer,
                                train["x"][: k * utils.is_classification(dataset)],
                                train["y"][: k * utils.is_classification(dataset)],
                                mode,
                                num_layers, lora_mode, lora_k
                            )
                        val_acc = eval(fine_tuned, tokenizer, val)
                        results["_".join([model_name, dataset, str(k), mode, str(num_layers), lora_mode, str(lora_k)])] = val_acc
                    else:
                        if k > 0:
                            fine_tuned = ft_generation(
                                model,
                                tokenizer,
                                train["x"][:k],
                                train["simple_y"][:k],
                                mode,
                                dataset,
                            )
                        else:
                            fine_tuned = copy.deepcopy(model)
                            fine_tuned.to(DEVICE)

                        fine_tuned.eval()
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val["x"])))))

                        for row in pbar:
                            test_input = val["x"][row]
                            targets.append(val["y"][row])
                            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
                            prompt = test_input + utils.get_prefix(dataset)[1]
                            input_ids = tokenizer(
                                prompt, return_tensors="pt"
                            ).input_ids.to(DEVICE)
                            sampled_tokens = utils.do_sample(
                                fine_tuned, input_ids, stop_tokens, max_tokens
                            )
                            decoded = tokenizer.decode(sampled_tokens).strip()
                            predictions.append(decoded)
                            metric = get_performance_metric(
                                predictions, targets, utils.metric_for_dataset(dataset)
                            )
                            pbar.set_description(f"Eval: {metric:.04f}")
                        results["_".join([model_name, dataset, str(k), mode, str(num_layers), lora_mode, str(lora_k)])] = metric

                    print(results)
                    question = "ft"
                    if not os.path.exists(f"{utils.RESULTS_DIR}/{question}"):
                        os.makedirs(f"{utils.RESULTS_DIR}/{question}")

                    for k_, v in results.items():
                        print(
                            f"Writing results to: {utils.RESULTS_DIR}/{question}/{k_}.json"
                        )
                        with open(
                            f"{utils.RESULTS_DIR}/{question}/{k_}.json", "a+"
                        ) as f:
                            json.dump({"metric": v}, f)
                    results = {}
                    if args.save_model:
                        print("Saving model...")
                        if not os.path.exists(f"{utils.RESULTS_DIR}/models"):
                            os.makedirs(f"{utils.RESULTS_DIR}/models")
                        torch.save(
                            fine_tuned.state_dict(),
                            f"{utils.RESULTS_DIR}/models/" + "_".join([model_name, dataset, str(k), mode, str(num_layers), lora_mode, str(lora_k)]) + ".pt",
                        )
                    
                    del fine_tuned


def run():
    ks = [int(k) for k in args.k.split(",")]
    num_layers_ = [int(k) for k in args.num_layers.split(",")]
    lora_k_ = [int(k) for k in args.lora_k.split(",")]
    lora_mode_ = args.lora_mode.split(",")
    if args.task == "ft":
        run_ft(args.model.split(","), args.dataset.split(","), ks, args.mode.split(","), num_layers_, lora_k_, lora_mode_)
    elif args.task == "plot":
        utils.plot_ft(
            args.model.split(","),
            args.dataset.split(","),
            ks,
            args.mode.split(","),
            num_layers_,
            lora_k_,
            lora_mode_,
            args.plot_name,
        )


if __name__ == "__main__":
    run()
