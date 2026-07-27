import csv
import gzip
import importlib
from pathlib import Path
from typing import Iterable, Iterator, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from features.train_bert128 import encode_text
from utils.utils import BASE, create_folder, load_word2vec_model, open_config, save_pkl
from features.w2v import eval_for_encoding, preprocess
from transformers import BertForMaskedLM, BertTokenizerFast

OBJECT_TYPES = ["PROCESS", "FILE"]
ACTION_TYPES = ["OPEN", "CREATE", "TERMINATE", "MODIFY", "WRITE", "RENAME", "READ", "DELETE"]


def read_line_csv_file(path: Path) -> Iterator[list]:
    mode = "rt" if path.suffix == ".gz" else "r"
    if path.suffix == ".gz":
        f = gzip.open(path, mode=mode, newline="")
    else:
        f = path.open(mode=mode, newline="")

    with f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            yield [row[0], row[1]]


def count_csv_rows(path: Path) -> int:
    mode = "rt" if path.suffix == ".gz" else "r"
    if path.suffix == ".gz":
        f = gzip.open(path, mode=mode, newline="")
    else:
        f = path.open(mode=mode, newline="")

    with f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for row in reader if len(row) >= 2)


def one_hot_encoding(type_info: str, is_edge: bool) -> List[int]:
    classes = ACTION_TYPES if is_edge else OBJECT_TYPES
    features = [0] * len(classes)
    if type_info in classes:
        features[classes.index(type_info)] = 1
    return features


def build_text_sequence(raw_line: str, cfg) -> str:
    if raw_line is None:
        return ""

    raw_line = str(raw_line).strip()
    if raw_line == "":
        return ""

    tokens = preprocess(raw_line, cfg)
    return " ".join(tokens)


def w2v_encoding(model, cmd_path_info, cfg):
    if not pd.isna(cmd_path_info):
        return eval_for_encoding(model, cmd_path_info, cfg).tolist()
    else:
        return [0] * cfg["MODEL"]["LEN_ENCODE_PATH"]


def bert_encoding(tokenizer, model, cmd_path_info, cfg):
    if pd.isna(cmd_path_info) or cmd_path_info in (None, "", 0):
        return [0] * cfg["MODEL"]["LEN_ENCODE_PATH"]

    text = build_text_sequence(cmd_path_info, cfg)
    if text == "":
        return [0] * cfg["MODEL"]["LEN_ENCODE_PATH"]

    return encode_text(tokenizer, model, text, cfg).tolist()


def create_features(
    line_iter: Iterator[list],
    model,
    cfg,
    is_edge: bool,
    nrows: int,
    output_path: Path,
    use_type: bool = True,
    model_type: str = "w2v",
    bert_tokenizer=None,
    bert_model=None,
) -> np.ndarray:
    dim_type = len(ACTION_TYPES) if is_edge else len(OBJECT_TYPES)
    dim = dim_type + cfg["MODEL"]["LEN_ENCODE_PATH"]
    create_folder(output_path.parent)
    features = np.lib.format.open_memmap(
        str(output_path), dtype=float, mode="w+", shape=(nrows, dim)
    )

    row_index = 0
    desc = "Creating edge features" if is_edge else "Creating node features"
    for line in tqdm(line_iter, total=nrows, desc=desc):
        if len(line) < 2:
            continue

        type_info = line[0]
        cmd_path_info = line[1]
        if use_type:
            type_embedding = one_hot_encoding(type_info, is_edge)
        else:
            type_embedding = [0] * dim_type

        if model_type == "no":
            cmd_path_embedding = [0] * cfg["MODEL"]["LEN_ENCODE_PATH"]
        elif model_type.lower() == "bert":
            cmd_path_embedding = bert_encoding(bert_tokenizer, bert_model, cmd_path_info, cfg)
        else:
            cmd_path_embedding = w2v_encoding(model, cmd_path_info, cfg)
        features[row_index] = type_embedding + cmd_path_embedding
        row_index += 1

    if row_index != nrows:
        return features[:row_index]
    return features


def save_processed_data(output_dir: Path, dataset, client, edge_features: np.ndarray, node_features: np.ndarray) -> None:
    create_folder(output_dir)
    np.save(output_dir / f"ml_{dataset}_{client}.npy", edge_features)
    np.save(output_dir / f"ml_{dataset}_{client}_node.npy", node_features)


def process_client(
    dataset: str,
    client: str,
    cfg,
    use_type: bool = True,
    model_type: str = "w2v",
    model=None,
    bert_tokenizer=None,
    bert_model=None,
) -> None:
    input_dir = Path(BASE) / "processed_data" / f"{dataset}_{client}"

    if model_type.lower() == "no":
        pass
    elif model_type.lower() == "bert":
        if bert_tokenizer is None or bert_model is None:
            raise ValueError("BERT tokenizer and model must be provided for model_type='bert'.")
    else:
        if model is None:
            model_path = f"{BASE}/feature_data/w2v_model.pt"
            model = load_word2vec_model(model_path)

    edge_features_file = input_dir / "edge_features.csv"
    edge_output_path = Path(BASE) / "processed_data" / f"{dataset}_{client}" / f"ml_{dataset}_{client}.npy"
    edge_count = count_csv_rows(edge_features_file)
    create_features(
        read_line_csv_file(edge_features_file),
        model,
        cfg,
        is_edge=True,
        nrows=edge_count,
        output_path=edge_output_path,
        use_type=use_type,
        model_type=model_type,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
    )

    node_features_file = input_dir / "node_features.csv"
    node_output_path = Path(BASE) / "processed_data" / f"{dataset}_{client}" / f"ml_{dataset}_{client}_node.npy"
    node_count = count_csv_rows(node_features_file)
    create_features(
        read_line_csv_file(node_features_file),
        model,
        cfg,
        is_edge=False,
        nrows=node_count,
        output_path=node_output_path,
        use_type=use_type,
        model_type=model_type,
        bert_tokenizer=bert_tokenizer,
        bert_model=bert_model,
    )



def main(dataset, clients, model_type: str = "w2v", bert_dir: str | None = None) -> None:
    cfg = open_config(dataset)

    use_type = False

    if model_type.lower() == "no":
        for client in clients:
            print(f"Processing client {client} for dataset {dataset} without features")
            process_client(
                dataset,
                client,
                cfg,
                use_type=use_type,
                model_type="no",
            )

    elif model_type.lower() == "bert":
        if bert_dir is None:
            bert_dir = f"{BASE}/feature_data/Bert_ft"
        tokenizer = BertTokenizerFast.from_pretrained(bert_dir, do_lower_case=True)
        model = BertForMaskedLM.from_pretrained(bert_dir)
        for client in clients:
            print(f"Processing client {client} for dataset {dataset} using BERT")
            process_client(
                dataset,
                client,
                cfg,
                use_type=use_type,
                model_type="bert",
                model=None,
                bert_tokenizer=tokenizer,
                bert_model=model,
            )
    else:
        model_path = f"{BASE}/feature_data/w2v_model.pt"
        model = load_word2vec_model(model_path)
        for client in clients:
            print(f"Processing client {client} for dataset {dataset} using W2V")
            process_client(dataset, client, cfg, use_type=use_type, model_type="w2v", model=model)


