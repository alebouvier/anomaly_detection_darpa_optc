import math
import os
import random
import re
from pathlib import Path
from typing import List, Union

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils.utils import BASE, create_folder, load_pickle_file, open_config


class PathCommandDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer: BertTokenizerFast, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_special_tokens_mask=True,
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}


def preprocess_path(line: str, cfg: dict):
    line_treatmnt = line.replace("\\", "/")
    line_treatmnt = re.sub(
        re.compile(cfg["MODEL"]["HARD_DEVICE"]),
        cfg["MODEL"]["SIMPLE_HARD_DEVICE"],
        line_treatmnt,
    )
    norm_line = os.path.normpath(line_treatmnt)
    data_lst = re.split(re.compile(cfg["MODEL"]["SPLIT_PATH"]), norm_line)

    match = re.compile(cfg["MODEL"]["SPLIT_EXTENSION"]).match(data_lst[-1])
    if match:
        data_lst.remove(data_lst[-1])
        data_lst.append(match.group(1))
        data_lst.append(match.group(2))

    data_lst = [token for token in data_lst if token not in ("", "?", "??")]
    return data_lst


def preprocess(line: str, cfg: dict):
    data_lst_complet: List[str] = []
    m = re.compile(cfg["MODEL"]["SPLIT_PATH_COMMAND_LINE"])
    data_lst = re.findall(m, line)

    if len(data_lst) == 0:
        return [line]

    data_lst_complet.extend(preprocess_path(data_lst[0].replace('"', ""), cfg))
    for data in data_lst[1:]:
        data = data.replace('"', "")
        if " " in data:
            data_lst_complet.extend(preprocess(data, cfg))
        elif "\\" in data or "/" in data:
            data_lst_complet.extend(preprocess_path(data, cfg))
        else:
            data_lst_complet.append(data)

    return [token for token in data_lst_complet if token != ""]


def normalize_text_item(item: Union[str, List[str]]) -> str:
    if isinstance(item, (list, tuple)) and len(item) > 0:
        return item[0]
    if item is None:
        return ""
    return item


def build_text_sequence(raw_line: str, cfg: dict) -> str:
    raw_line = raw_line.strip()
    if raw_line == "":
        return raw_line

    tokens = preprocess(raw_line, cfg)
    return " ".join(tokens)


def get_sequence_embedding(sequence: str, tokenizer, model) -> torch.Tensor:
    inputs = tokenizer(sequence, return_tensors="pt")

    # move inputs to model device if model on GPU
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Use the encoder portion if loading a task model (e.g., BertForMaskedLM)
    encoder = model.bert if hasattr(model, "bert") else model

    encoder.eval()
    with torch.no_grad():
        outputs = encoder(**inputs)

    # outputs.last_hidden_state shape: (batch, seq_len, hidden)
    last_hidden = outputs.last_hidden_state

    # prefer the token at position 1 if present (consistent with other code),
    # otherwise fall back to CLS (position 0) or mean pooling
    if last_hidden.size(1) > 1:
        seq_embedding = last_hidden[0, 1, :]
    else:
        seq_embedding = last_hidden[0].mean(dim=0)

    return seq_embedding


def encode_text(tokenizer, model, text, cfg):
    """Encode `text` to a numpy vector of length `cfg["MODEL"]["LEN_ENCODE_PATH"]`.

    The function uses the model's encoder to produce a hidden vector, then
    truncates or pads to match the configured length so callers get a fixed size.
    """
    if text is None or text == 0 or text == "":
        return np.zeros(cfg["MODEL"]["LEN_ENCODE_PATH"])

    seq_embedding = get_sequence_embedding(text, tokenizer, model)
    vec = seq_embedding.detach().cpu().numpy()

    target_len = cfg["MODEL"]["LEN_ENCODE_PATH"]
    if vec.shape[0] == target_len:
        return vec
    elif vec.shape[0] > target_len:
        return vec[:target_len]
    else:
        # pad with zeros
        pad = np.zeros(target_len - vec.shape[0], dtype=vec.dtype)
        return np.concatenate([vec, pad])


def collect_texts(clients: List[str], data: str, sampled_content_file: float, cfg: dict) -> List[str]:
    texts: List[str] = []
    for client in clients:
        preprocessing_folder = f"{BASE}/processed_data/{data}_{client}"
        path_cmd_file = Path(preprocessing_folder) / "path_cmd_list.pkl"
        raw_lines = load_pickle_file(str(path_cmd_file))
        raw_lines = [normalize_text_item(line) for line in raw_lines]
        if len(raw_lines) == 0:
            continue

        sample_size = max(1, math.ceil(sampled_content_file * len(raw_lines)))
        sampled_lines = random.sample(raw_lines, sample_size)
        texts.extend(build_text_sequence(line, cfg) for line in sampled_lines if line)
    return texts


def train_bert128(
    clients: List[str],
    data: str,
    sampled_content_file: float = 0.01,
    epochs: int = 3,
    batch_size: int = 32,
    output_dir: str | None = None,
):
    cfg = open_config(data)
    if output_dir is None:
        output_dir = f"{BASE}/feature_data/Bert_ft"

    bert_path = "google/bert_uncased_L-2_H-128_A-2"
    tokenizer = BertTokenizerFast.from_pretrained(bert_path, do_lower_case=True)
    model = BertForMaskedLM.from_pretrained(bert_path)

    texts = collect_texts(clients, data, sampled_content_file, cfg)
    if len(texts) == 0:
        raise ValueError("No text samples found for BERT fine-tuning.")

    dataset = PathCommandDataset(texts, tokenizer, max_length=128)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    training_args = TrainingArguments(
        output_dir=f"{BASE}/feature_data/bert128_training",
        # overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    create_folder(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def main(clients, data, sampled_content_file=0.01, epochs=3, batch_size=32):
    return train_bert128(
        clients=clients,
        data=data,
        sampled_content_file=sampled_content_file,
        epochs=epochs,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a BERT-128 model on path and command line data.")
    parser.add_argument("--data", type=str, required=True, help="Dataset name for config / processed_data folder")
    parser.add_argument("--clients", nargs="+", required=True, help="Client ids to use for training")
    parser.add_argument("--sample", type=float, default=0.001, help="Fraction of lines to sample from each client")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--output_dir", type=str, default=f"{BASE}/feature_data/Bert_ft", help="Directory to save the fine-tuned model")
    args = parser.parse_args()

    print("Starting BERT-128 fine-tuning")
    trained_dir = train_bert128(
        clients=args.clients,
        data=args.data,
        sampled_content_file=args.sample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    print(f"Saved fine-tuned BERT to {trained_dir}")
