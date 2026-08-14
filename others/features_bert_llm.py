import copy
import gc
import importlib
from tqdm import tqdm

from peft import PeftModel

from features.bert import encode_text
from features.features_bert import Encoding_builder as BertEncodingBuilder
from features.LLM_augmented import enrich_process_nodes

from utils.utils import (
    encoding_parent_son,
    encoding_sid,
    load_pickle_file,
    save_pkl,
    open_config,
    BASE,
)


class Encoding_builder(BertEncodingBuilder):
    def encoding_process(self, g, nodes):
        features_e_g = {}
        parent_and_son = [0, 0, 0, 0]

        process_logs = []
        for node in nodes:
            process_logs.append(
                {
                    "command_line": node.command_line if node.command_line != 0 else "",
                    "image_path": node.image_path if node.image_path != 0 else "",
                    "parent_image_path": node.parent_image_path if node.parent_image_path != 0 else "",
                }
            )
        descriptions = enrich_process_nodes(process_logs, batch_size=20)

        for node, description in zip(nodes, descriptions):
            text_embedding = encode_text(self.tokenizer, self.model, description, self.cfg)
            sid = encoding_sid([node.sid])
            p_and_son_ = encoding_parent_son(
                copy.deepcopy(parent_and_son), node.parent_image_path, node.image_path
            )
            features_e_g[node.id] = text_embedding.tolist() + sid + p_and_son_

            parent_and_son = encoding_parent_son(
                parent_and_son, node.parent_image_path, node.image_path
            )

        features_g = None
        return features_g, features_e_g


def main(clients, graphs, model_w2v_path, dataset, g=False, e=False):
    utils_dataset = importlib.import_module(f"data.{dataset}_utils")
    cfg_dataset = open_config(dataset)

    features_g, features_e = {c: {} for c in clients}, {c: {} for c in clients}

    metadata = True

    data = load_pickle_file(graphs)

    bert_path = "bert-base-uncased"
    bert_ft_path = f"{BASE}/feature_data/Bert_ft"

    tokenizer = None
    model = None
    try:
        from transformers import BertTokenizerFast, BertModel

        tokenizer = BertTokenizerFast.from_pretrained(bert_path, do_lower_case=True)
        model = BertModel.from_pretrained(bert_path)
        model = PeftModel.from_pretrained(model, bert_ft_path)
        model.eval()
    except Exception as e:
        print(f"Error loading BERT model: {e}")
        raise

    for c in clients:
        for d, g in tqdm(data[c].items()):
            G = load_pickle_file(g)
            e = Encoding_builder(
                G, tokenizer, model, metadata, utils_dataset, cfg_dataset
            )
            e_g, e_e = e.build_encoding()

            save_pkl(
                {d: e_g},
                f"{BASE}/feature_data/optc_{c}/features/{c}_{d}_g.pkl",
            )
            save_pkl(
                {d: e_e},
                f"{BASE}/feature_data/optc_{c}/features/{c}_{d}_e.pkl",
            )
            features_g[c][d] = f"feature_data/optc_{c}/features/{c}_{d}_g.pkl"
            features_e[c][d] = f"feature_data/optc_{c}/features/{c}_{d}_e.pkl"
            del G, e, e_g, e_e
            gc.collect()

    save_pkl(features_g, f"{BASE}/feature_data/features_g.pkl")
    save_pkl(features_e, f"{BASE}/feature_data/features_e.pkl")

    return
