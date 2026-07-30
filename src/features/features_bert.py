import copy
import gc
import importlib
from xml.parsers.expat import model
import networkx as nx
import numpy as np
from tqdm import tqdm

from features.bert import eval_for_encoding
from peft import PeftModel

from transformers import BertTokenizerFast, BertModel

from utils.utils import (
    encoding_parent_son,
    encoding_sid,
    load_pickle_file,
    load_word2vec_model,
    save_pkl,
    open_config,
    BASE,
)


class Encoding_builder:
    def __init__(self, file, tokenizer, model, metadata, selected_nodes, cfg):
        self.g = file
        self.encoding_size = 0
        self.selected_nodes = selected_nodes
        self.metadata = metadata
        self.cfg = cfg
        if metadata:
            self.tokenizer = tokenizer
            self.model = model

    def build_encoding(self):
        """Build node and edge features for the graph."""

        node_types = set(
            data.get("type_") for _, data in self.g.nodes(data=True) if "type_" in data
        )

        node_types = list(node_types)
        # print(node_types)
        # exit(1)

        ac = [
            n[0]
            for n in self.g.edges(data=True)
            if type(n[0]).__name__ == self.cfg["MODEL"]["SELECTED_NODES"]
        ]
        dst = [
            n[1]
            for n in self.g.edges(data=True)
            if type(n[1]).__name__ == self.cfg["MODEL"]["SELECTED_NODES"]
        ]
        nodes = list(set(ac + dst))
        # print("Nb processes: ", len(nodes))
        if len(nodes) == 0:
            print("Error no process in this graph")
            exit(1)
        try:
            enc_struc_g, enc_struc_e, fe_b = self.encoding(self.g, nodes)
            if self.metadata:
                enc_process_g, enc_process_e = self.encoding_process(self.g, nodes)

                features_g = np.array(enc_struc_g).astype(np.float32)
                features_e = {}
                for k in set(enc_struc_e):
                    features_e[k] = [
                        np.array(enc_process_e.get(k, [])).astype(np.float32),
                        fe_b[k][0],
                    ]
            else:
                features_g = np.array(enc_struc_g).astype(np.float32)
                features_e = {}
                for k in set(enc_struc_e):
                    features_e[k] = [
                        np.array(enc_struc_e.get(k, [])).astype(np.float32),
                        fe_b[k][0],
                    ]

        except Exception as e:
            print(f"Error {e}")
            exit(1)

        return features_g, features_e

    def mean_path(self, lst):
        """Return the mean feature vector for a list of paths."""
        if len(lst) > 0:
            return np.mean(lst, axis=0)
        else:
            return np.zeros(self.cfg["MODEL"]["LEN_ENCODE_PATH"]).tolist()

    def encoding_process(self, g, nodes):
        # graph
        image_paths = []
        """Perform the work of encoding process."""
        parent_image_paths = []
        command_line_paths = []
        sids = []
        parent_and_son = [0, 0, 0, 0]

        # entity
        features_e_g = {}
        p_and_son = [0, 0, 0, 0]

        for node in nodes:  # for each process
            image_path = eval_for_encoding(
                self.tokenizer, self.model, node.image_path, False, cfg=self.cfg
            )
            # pp_path = eval_for_encoding(
            #     self.tokenizer, self.model, node.parent_image_path, False, cfg=self.cfg
            # )
            # cmd_path = eval_for_encoding(
            #     self.tokenizer, self.model, node.command_line, True, cfg=self.cfg
            # )
            sid = encoding_sid([node.sid])
            p_and_son_ = encoding_parent_son(
                copy.deepcopy(p_and_son), node.parent_image_path, node.image_path
            )
            features_e_g[node.id] = (  # Encoding of a process
                image_path.tolist()
                # + pp_path.tolist()
                # + cmd_path.tolist()
                + sid
                + p_and_son_
            )

            
        
                
            # print(len(features_e_g[node.id]))

            sids.append(node.sid)
            image_paths.append(image_path)
            # parent_image_paths.append(pp_path)
            # command_line_paths.append(cmd_path)
            parent_and_son = encoding_parent_son(
                parent_and_son, node.parent_image_path, node.image_path
            )

        # image_path = self.mean_path(image_paths)
        # # parent_image_path = self.mean_path(parent_image_paths)
        # # command_line_path = self.mean_path(command_line_paths)
        # sid_enc = encoding_sid(sids)
        # features_g = (
        #     image_path.tolist()
        #     # + parent_image_path.tolist()
        #     # + command_line_path.tolist()
        #     + sid_enc
        #     + parent_and_son
        # )
        features_g = None
        return features_g, features_e_g

    def degree(self, degrees):
        """Return the maximum and average degree values for a set of nodes."""
        if len(degrees) == 0:
            return 0, 0
        return int(sorted(degrees.values(), reverse=True)[0]), int(
            sum(degrees.values()) / len(degrees)
        )

    def get_degree(self, g, nodes):
        """Compute graph-level and node-level degree features for the selected nodes."""
        c, d = self.degree(dict(g.degree(nodes)))
        e, f = self.degree(dict(g.in_degree(nodes)))
        h, i = self.degree(dict(g.out_degree(nodes)))

        features_e = {}
        for n in nodes:
            degree = nx.degree(g, n)
            in_degree = len(g.in_edges(n))
            out_degree = len(g.out_edges(n))
            features_e[n.id] = [
                degree,
                degree,
                in_degree,
                in_degree,
                out_degree,
                out_degree,
            ]

        features_g = [c, d, e, f, h, i]
        return features_g, features_e

    def encoding(self, g, nodes):
        """Combine multiple feature groups into the final graph encoding."""
        fg_a, fe_a = self.edge_type_for_node_type(g, nodes)
        fg_b, fe_b = self.get_degree(g, nodes)
        fg_c, fe_c = self.lifetimenodes(nodes)
        fg_d, fe_d = self.node_time_for_node_type(g, nodes)

        features_g = fg_a + fg_b + fg_c + fg_d
        features_e = {}
        # for k in fe_a.keys():
        #    features_e[k] = (
        #        fe_a.get(k, []) + fe_b.get(k, []) + fe_c.get(k, []) + fe_d.get(k, [])
        #    )
        for k in fe_a.keys():
            features_e[k] = [x for d in (fe_a, fe_b, fe_c, fe_d) for x in d.get(k, [])]
            # print(len(features_e[k]))

        return features_g, features_e, fe_b

    def lifetimenodes(self, nodes):
        min_life = 15
        """Perform the work of lifetimenodes."""
        max_life = 0
        sum_life = 0

        features_e = {}
        for node in nodes:
            life = node.lifetime()
            features_e[node.id] = [life, life, life]
            if life > max_life:
                max_life = life
            if life < min_life:
                min_life = life
            sum_life += life
        features_g = [min_life, max_life, sum_life]
        return features_g, features_e

    def freq_time_mean(self, node_freq):
        a = []
        """Perform the work of freq time mean."""
        for _, freq in node_freq.items():
            if len(freq) < 1:
                a.append(0)
            else:
                a.append(np.mean(freq))
        return a

    def node_time_for_node_type(self, g, nodes):
        node_time_out_freq = {}
        """Perform the work of node time for node type."""
        node_time_in_freq = {}

        for type_ in self.cfg["MODEL"]["NODES_TYPES_ALL"]:
            node_time_out_freq[type_] = []
            node_time_in_freq[type_] = []

        # entity
        features_e = {}
        node_time_out_freq_e = copy.deepcopy(node_time_out_freq)
        node_time_in_freq_e = copy.deepcopy(node_time_in_freq)

        for node in nodes:
            edges = g.out_edges(node, data=True)
            node_time_out_freq_ = self.freq_time(
                edges, True, copy.deepcopy(node_time_out_freq_e)
            )

            edges = g.in_edges(node, data=True)
            node_time_in_freq_ = self.freq_time(
                edges, False, copy.deepcopy(node_time_in_freq_e)
            )

            features_e[node.id] = self.freq_time_mean(
                node_time_out_freq_
            ) + self.freq_time_mean(node_time_in_freq_)

            for k in node_time_out_freq_:
                node_time_out_freq[k] += node_time_out_freq_.get(k, 0)
                node_time_in_freq[k] += node_time_in_freq_.get(k, 0)

        a = self.freq_time_mean(node_time_out_freq)
        b = self.freq_time_mean(node_time_in_freq)
        features_g = a + b
        return features_g, features_e

    def freq_time(self, edges, out, node_time_freq):
        """Collect lifetime values for edges grouped by node type."""
        for edge in edges:
            if out:
                node_type = type(edge[1]).__name__
                n = edge[1]
            else:
                node_type = type(edge[0]).__name__
                n = edge[0]

            if node_type in node_time_freq:
                node_time_freq[node_type].append(n.lifetime())
            else:
                node_time_freq[node_type] = [n.lifetime()]
        return node_time_freq

    def edge_type_for_node_type(self, g, nodes):
        """Count edge and node types for each selected node."""
        edge_types_out_freq = {}
        edge_types_in_freq = {}
        node_types_out_freq = {}
        node_types_in_freq = {}

        for type_ in self.cfg["MODEL"]["LST_ACTION"]:
            edge_types_out_freq[type_] = 0
            edge_types_in_freq[type_] = 0

        for type_ in self.cfg["MODEL"]["NODES_TYPES_ALL"]:
            node_types_out_freq[type_] = 0
            node_types_in_freq[type_] = 0

        # entity
        features_e = {}
        node_types_out_freq_e = copy.deepcopy(node_types_out_freq)
        node_types_in_freq_e = copy.deepcopy(node_types_in_freq)
        edge_types_out_freq_e = copy.deepcopy(edge_types_out_freq)
        edge_types_in_freq_e = copy.deepcopy(edge_types_in_freq)

        for node in nodes:
            edges = g.out_edges(node, data=True)
            edge_types_out_freq_, node_types_out_freq_ = self.freq_(
                edges,
                True,
                copy.deepcopy(edge_types_out_freq_e),
                copy.deepcopy(node_types_out_freq_e),
            )

            edges = g.in_edges(node, data=True)
            edge_types_in_freq_, node_types_in_freq_ = self.freq_(
                edges,
                False,
                copy.deepcopy(edge_types_in_freq_e),
                copy.deepcopy(node_types_in_freq_e),
            )

            features_e[node.id] = (
                [freq for type, freq in edge_types_out_freq_.items()]
                + [freq for type, freq in edge_types_in_freq_.items()]
                + [freq for type, freq in node_types_out_freq_.items()]
                + [freq for type, freq in node_types_in_freq_.items()]
            )

            for k in edge_types_out_freq_:
                edge_types_out_freq[k] += edge_types_out_freq_.get(k, 0)
                edge_types_in_freq[k] += edge_types_in_freq_.get(k, 0)
            for k in node_types_out_freq_:
                node_types_in_freq[k] += node_types_in_freq_.get(k, 0)
                node_types_out_freq[k] += node_types_out_freq_.get(k, 0)

        features_g = (
            [freq for type, freq in edge_types_out_freq.items()]
            + [freq for type, freq in edge_types_in_freq.items()]
            + [freq for type, freq in node_types_out_freq.items()]
            + [freq for type, freq in node_types_in_freq.items()]
        )
        return features_g, features_e

    def freq_(self, edges, out, edge_types_freq, node_types_freq):
        """Count edge and node type frequencies for the supplied edges."""
        for edge in edges:
            edge_type = edge[2]["action"]
            if edge_type in edge_types_freq:
                edge_types_freq[edge_type] += 1
            else:
                edge_types_freq[edge_type] = 1

            if out:
                node_type = type(edge[1]).__name__
            else:
                node_type = type(edge[0]).__name__

            if node_type in node_types_freq:
                node_types_freq[node_type] += 1
            else:
                node_types_freq[node_type] = 1
        return edge_types_freq, node_types_freq

    def set_g(self, g):
        self.g = g
        """Store set g."""
        return g

    def set_encoding_size(self, encoding_size):
        self.encoding_size = encoding_size
        """Store set encoding size."""
        return encoding_size

    def info(self):
        return f"Graph {self.g}."
        """Perform the work of info."""


def main( clients, graphs, model_w2v_path, dataset, g=False, e=False):
    """Perform the work of main."""
    utils_dataset = importlib.import_module(f"data.{dataset}_utils")
    cfg_dataset = open_config(dataset)

    features_g, features_e = {c: {} for c in clients}, {c: {} for c in clients}

    metadata = True

    data = load_pickle_file(graphs)


    # bert_path = 'google/bert_uncased_L-2_H-128_A-2'

    bert_path = 'bert-base-uncased'
    bert_ft_path = f"{BASE}/feature_data/Bert_ft"

    tokenizer = BertTokenizerFast.from_pretrained(bert_path, do_lower_case=True)
    model = BertModel.from_pretrained(bert_path)
    model = PeftModel.from_pretrained(model, bert_ft_path)
    model.eval()

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
