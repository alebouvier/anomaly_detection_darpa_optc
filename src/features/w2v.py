import importlib
import math
import os
import random
import re

import gensim
import numpy as np
from tqdm import tqdm

from utils.utils import load_pickle_file, open_config


class MySentences(object):
    def __init__(self, files, sampled_content_file, is_path, cfg):
        self.files = files
        self.sample = sampled_content_file
        self.nb = 0
        self.is_path = is_path
        self.cfg = cfg

    def __iter__(self):
        lines = random.sample(self.files, math.ceil(self.sample * len(self.files)))
        for line in lines:
            line = line[0]
            if self.is_path:
                lst, _ = preprocess_path(line, self.cfg)
                self.nb += len(lst)
                yield lst
            else:
                lst = preprocess_command_line(line, self.cfg)
                self.nb += len(lst)
                yield lst

    def nb_word(self):
        return self.nb

    def nb_line(self):
        return len(self.files)


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


def train_val(data_train, cfg):
    epochs = 100
    model = gensim.models.Word2Vec(
        vector_size=cfg["MODEL"]["LEN_ENCODE_PATH"],
        window=cfg["MODEL"]["WINDOW"],
        sg=0,
    )
    model.build_vocab(data_train)

    # print(len(model.wv.key_to_index.keys()))

    for epoch in tqdm(range(epochs)):
        model.train(
            data_train,
            total_examples=model.corpus_count,
            epochs=1,
            compute_loss=True,
        )
    return model


def eval_for_encoding(model, data, is_command, cfg):
    if data == 0:
        return np.zeros(cfg["MODEL"]["LEN_ENCODE_PATH"])
    else:
        if is_command:
            data_lst = preprocess_command_line(data, cfg)
            ext = False
        else:
            data_lst, ext = preprocess_path(data, cfg)
        model, sentences_emb, _, _, _, _ = eval(
            model, [data_lst, ext, data, None, None], is_command, cfg
        )
        return sentences_emb


def eval_unknown(lst, unkown_index, h, model, last, cfg):
    emb = []
    index = 1
    p = 0
    while p < len(unkown_index):
        k = unkown_index[p]
        for i in range(min(k + 1, len(h)), len(h)):
            if i in unkown_index:
                index += 1
            else:
                break

        if p + 1 < len(unkown_index):
            next = unkown_index[p + 1]
        else:
            next = len(h)

        for i in range(max(0, k - cfg["MODEL"]["WINDOW"]), k):
            try:
                emb.append(model.wv[h[i]])
            except Exception as e:
                print(f"Error {e}")
        for i in range(
            k + index, min(len(h), k + cfg["MODEL"]["WINDOW"] + index, next)
        ):
            try:
                emb.append(model.wv[h[i]])
            except Exception as e:
                print(f"Error {e}")

        if len(emb) > 0:
            context_emb = np.mean(np.array(emb), axis=0)
            most_similar = model.wv.similar_by_vector(context_emb, topn=2)
            unkown_emb = [model.wv[i[0]] for i in most_similar]
            unkown_emb = np.mean(np.array(unkown_emb), axis=0)

            for j in range(index):
                lst.append(
                    eval_function_coeff_path_2combine(unkown_emb, last, h, k + j)
                )
                model.wv.add_vector(h[k + j], unkown_emb)
                model.wv.fill_norms(force=True)
        else:
            lst.append(np.mean(model.wv.vectors, axis=0))

        p += index
        index = 1
    return lst


def eval_function_coeff_path_2combine(enc, last, h, k):
    b = 1
    if k < len(h) - last:
        res = ((-b / (len(h) + 1)) * k + b) * enc
    else:
        res = ((b / last) * k + b * (2 - len(h) / last)) / 2 * enc
    return res


def eval_function_coeff_path_decrois_geo(enc, last, h, k, ulast, q=0.1):
    b = q
    if k < len(h) - last:
        res = (q * ulast) * enc
    else:
        res = (q * q) * enc
    return res, q * ulast


def eval_function_coeff_path_const(enc):
    return enc


def eval(model, data, is_command, cfg, f="const", q=0.1):
    h = data[0]
    ext = data[1]
    if ext:
        last = 1
    else:
        last = 0
    lst = []
    unkown_index = []
    ulast = 1
    for k, j in enumerate(h):
        if j not in model.wv:
            unkown_index.append(k)
        else:
            if f == "combine":
                lst.append(eval_function_coeff_path_2combine(model.wv[j], last, h, k))

            elif f == "const":
                lst.append(eval_function_coeff_path_const(model.wv[j]))

            elif f == "geo":
                res, ulast = eval_function_coeff_path_decrois_geo(
                    model.wv[j], last, h, k, ulast, q
                )
                lst.append(res)

            else:
                lst.append(model.wv[j])

    if len(unkown_index) > 0:
        lst = eval_unknown(lst, unkown_index, h, model, last, cfg)

    if len(lst) == 0:
        lst = [np.mean(model.wv.vectors, axis=0)]

    return model, np.mean(np.array(lst), axis=0), data[2], data[3], data[4], h


def main(
    base, clients, dataset, data, is_path=True, batch=64, sampled_content_file=0.01
):
    cfg = open_config(data)
    utils_dataset = importlib.import_module(f"data.{data}_utils")
    dataset = load_pickle_file(dataset)
    ft = []
    for c in clients:
        time_train, _, _ = utils_dataset.get_sets(dataset[c].keys())
        for t in time_train:
            ft.extend(load_pickle_file(dataset[c][t]))


    mysentences = MySentences(ft, sampled_content_file, is_path=is_path, cfg=cfg)

    model = train_val(mysentences, cfg)

    if is_path:
        model.save(base + f"/feature_data/w2v_model_path.pt")
    else:
        model.save(base + f"/feature_data/w2v_model_cmd.pt")

    return
