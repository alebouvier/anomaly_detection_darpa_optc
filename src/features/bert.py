import importlib
import math
import os
import random
import re

import gensim
import numpy as np
import torch
import torch
from tqdm import tqdm

from utils.utils import load_pickle_file, open_config





def preprocess_path(line, cfg):
    """Normalize a path-like string and split it into its logical components."""
    line_treatmnt = line.replace("\\", "/")  # Normalize path
    line_treatmnt = re.sub(
        re.compile(cfg["MODEL"]["HARD_DEVICE"]),
        cfg["MODEL"]["SIMPLE_HARD_DEVICE"],
        line_treatmnt,
    )
    norm_line = os.path.normpath(line_treatmnt)
    data_lst = re.split(re.compile(cfg["MODEL"]["SPLIT_PATH"]), norm_line)

    is_extension = False  # Check if extension
    match = re.compile(cfg["MODEL"]["SPLIT_EXTENSION"]).match(data_lst[-1])
    if match:
        is_extension = True
        data_lst.remove(data_lst[-1])
        data_lst.append(match.group(1))
        data_lst.append(match.group(2))

    if "" in data_lst:  # Remove trash in list
        data_lst.remove("")
    if "?" in data_lst:
        data_lst.remove("?")
    if "??" in data_lst:
        data_lst.remove("??")

    return data_lst, is_extension


def preprocess_command_line(line, cfg):
    """Split a command line into tokenized path and argument fragments."""
    data_lst_complet = []
    m = re.compile(cfg["MODEL"]["SPLIT_PATH_COMMAND_LINE"])
    data_lst = re.findall(m, line)

    path_lst, _ = preprocess_path(data_lst[0].replace('"', ""), cfg)
    data_lst_complet.extend(path_lst)

    for data in data_lst[1:]:
        data = data.replace('"', "")
        if " " in data:
            data = preprocess_command_line(data, cfg)
            data_lst_complet.extend(data)
        elif "\\" in data or "/" in data:
            path_lst, _ = preprocess_path(data, cfg)
            data_lst_complet.extend(path_lst)
        else:
            data_lst_complet.append(data)

    if "" in data_lst_complet:  # Remove trash in list
        data_lst_complet.remove("")

    return data_lst_complet



def eval_for_encoding(tokenizer, model, data, is_command, cfg):
    """Encode a path or command line into the model embedding space."""
    if data == 0:
        return np.zeros(cfg["MODEL"]["LEN_ENCODE_PATH"])
    else:
        if is_command:
            data_lst = preprocess_command_line(data, cfg)
            ext = False
        else:
            data_lst, ext = preprocess_path(data, cfg)
        tokenizer, model, sentences_emb, _, _, _, _ = eval(
            tokenizer, model, [data_lst, ext, data, None, None], is_command, cfg
        )

        return sentences_emb



def eval_function_coeff_path_2combine(enc, last, h, k):
    """Blend the embedding with a position-dependent coefficient along the path."""
    b = 1
    if k < len(h) - last:
        res = ((-b / (len(h) + 1)) * k + b) * enc
    else:
        res = ((b / last) * k + b * (2 - len(h) / last)) / 2 * enc
    return res


def eval_function_coeff_path_decrois_geo(enc, last, h, k, ulast, q=0.1):
    """Apply the geometric decay rule to weight the current path embedding."""
    if k < len(h) - last:
        res = (q * ulast) * enc
    else:
        res = (q * q) * enc
    return res, q * ulast


def eval_function_coeff_path_const(enc):
    """Return the embedding unchanged."""
    return enc


def get_sequence_embedding(sequence: str, tokenizer, model) -> torch.Tensor:
    """Extract the contextual embedding for a tokenized input sequence."""
    inputs = tokenizer(sequence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    
    seq_embedding = outputs.last_hidden_state[0, 1, :]  # shape: [128]
    return seq_embedding


def encode_text(tokenizer, model, text, cfg):
    """Encode text into a fixed-length vector for downstream feature extraction."""
    if text is None or text == 0 or text == "":
        return np.zeros(cfg["MODEL"]["LEN_ENCODE_PATH"])
    seq_embedding = get_sequence_embedding(text, tokenizer, model)
    return seq_embedding.detach().cpu().numpy()


def eval(tokenizer, model, data, is_command, cfg, f="const", q=0.1):
    """Aggregate a path's token embeddings and return the combined representation."""
    h = data[0]
    ext = data[1]
    if ext:
        last = 1
    else:
        last = 0

    ulast = 1

    # evaluate the embedding of the sequence of tokens
    

    if f == "combine":
        seq_embedding = eval_function_coeff_path_2combine(get_sequence_embedding(" ".join(h), tokenizer, model), last, h, k)

    elif f == "const":
        seq_embedding = eval_function_coeff_path_const(get_sequence_embedding(" ".join(h), tokenizer, model))

    elif f == "geo":
        res, ulast = eval_function_coeff_path_decrois_geo(
            get_sequence_embedding(" ".join(h), tokenizer, model), last, h, k, ulast, q
        )
        

    else:
        seq_embedding = get_sequence_embedding(" ".join(h), tokenizer, model)



    return tokenizer, model, seq_embedding, data[2], data[3], data[4], h


