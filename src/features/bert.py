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


# def eval_unknown(lst, unkown_index, h, model, last, cfg):
#     emb = []
#     index = 1
#     p = 0
#     while p < len(unkown_index):
#         k = unkown_index[p]
#         for i in range(min(k + 1, len(h)), len(h)):
#             if i in unkown_index:
#                 index += 1
#             else:
#                 break

#         if p + 1 < len(unkown_index):
#             next = unkown_index[p + 1]
#         else:
#             next = len(h)

#         for i in range(max(0, k - cfg["MODEL"]["WINDOW"]), k):
#             try:
#                 emb.append(get_word_embedding(h[i], tokenizer, model))
#             except Exception as e:
#                 print(f"Error {e}")
#         for i in range(
#             k + index, min(len(h), k + cfg["MODEL"]["WINDOW"] + index, next)
#         ):
#             try:
#                 emb.append(model(h[i], return_tensors='np'))
#             except Exception as e:
#                 print(f"Error {e}")

#         # if len(emb) > 0:
#         #     context_emb = np.mean(np.array(emb), axis=0)
#         #     most_similar = model.wv.similar_by_vector(context_emb, topn=2)
#         #     unkown_emb = [model(i[0], return_tensors='np') for i in most_similar]
#         #     unkown_emb = np.mean(np.array(unkown_emb), axis=0)

#         #     for j in range(index):
#         #         lst.append(
#         #             eval_function_coeff_path_2combine(unkown_emb, last, h, k + j)
#         #         )
#         #         model.wv.add_vector(h[k + j], unkown_emb)
#         #         model.wv.fill_norms(force=True)
#         # else:
#         #     lst.append(np.mean(model.wv.vectors, axis=0))

#         p += index
#         index = 1
#     return lst


def eval_function_coeff_path_2combine(enc, last, h, k):
    b = 1
    if k < len(h) - last:
        res = ((-b / (len(h) + 1)) * k + b) * enc
    else:
        res = ((b / last) * k + b * (2 - len(h) / last)) / 2 * enc
    return res


def eval_function_coeff_path_decrois_geo(enc, last, h, k, ulast, q=0.1):
    if k < len(h) - last:
        res = (q * ulast) * enc
    else:
        res = (q * q) * enc
    return res, q * ulast


def eval_function_coeff_path_const(enc):
    return enc


def get_word_embedding(word: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(word, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state shape: [batch, tokens, 768]
    # Token 0 = [CLS], 1 = word, 2 = [SEP]
    word_embedding = outputs.last_hidden_state[0, 1, :]  # shape: [768]
    return word_embedding

def eval(tokenizer, model, data, is_command, cfg, f="const", q=0.1):
    h = data[0]
    ext = data[1]
    if ext:
        last = 1
    else:
        last = 0
    lst = []
    ulast = 1
    for k, j in enumerate(h):
        if f == "combine":
            lst.append(eval_function_coeff_path_2combine(get_word_embedding(j, tokenizer, model), last, h, k))

        elif f == "const":
            lst.append(eval_function_coeff_path_const(get_word_embedding(j, tokenizer, model)))

        elif f == "geo":
            res, ulast = eval_function_coeff_path_decrois_geo(
                get_word_embedding(j, tokenizer, model), last, h, k, ulast, q
            )
            lst.append(res)

        else:
            lst.append(get_word_embedding(j, tokenizer, model))

    mean_emb = np.mean(np.array(lst), axis=0)

    return tokenizer, model, mean_emb, data[2], data[3], data[4], h


