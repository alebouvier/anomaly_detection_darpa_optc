import gzip
import json
import importlib
import traceback

import networkx as nx
from tqdm import tqdm

from utils.utils import period, round_duration, save_pkl, open_config, BASE




class Graph_builder:
    def __init__(self, data, duration, metadata, node_dataset, cfg):
        """This class Graph_builder create graphs.

        Parameters
        ----------
        data: list of dict
            This is the list of all events of the graph.

        """
        self.data = data
        self.duration = duration
        self.metadata = metadata
        self.nodes = {}
        self.labels = 0
        self.command_line = []
        self.node_dataset = node_dataset
        self.path = []
        self.cfg = cfg
        self.g = self.build_graph(data)

    def get_g(self):
        return self.g

    def get_command_lines(self):
        return self.command_line

    def get_path(self):
        return self.path

    def is_in_line(self, id_, line):
        if id_ in line:
            return line[id_]
        else:
            return 0

    def build_graph(self, data):
        """Create a graph.

        Parameters
        ----------
        data: list of dict
            This is the list of all events of the graph.

        Returns
        -------
        g: nx.MultiDiGraph
            The complete graph.
        """
        g = nx.MultiDiGraph()
        for line in data:
            g = self.add_one_log_into_graph(line, g)  # Add one event to the graph
        # print(nx.info(g), self.labels)
        return g

    def add_one_log_into_graph(self, line, g):
        """Add one event to the graph.

        Parameters:
        line: dict
            The event to be added.
        g: nx.MultiDiGraph
            The graph.

        Returns
        -------
        g: nx.MultiDiGraph
            The graph where the line was added.

        """
        time = self.is_in_line("timestamp", line)
        action = self.is_in_line("action", line)
        # If label are already into the event, extract it.
        if "red" in line:
            red = self.is_in_line("red", line)
        else:
            red = self.is_in_line("label", line)

        # Build nodes
        if self.metadata:
            isnewsrc, src, isnewdest, dest = self.build_nodes_metadata(
                line, time, action
            )
        else:
            isnewsrc, src, isnewdest, dest = self.build_nodes(line, time, action)
        if (src is None) or (dest is None):
            return g

        # Build event object to manage nodes properties
        Event = getattr(self.node_dataset, "Event")
        event = Event(time, action, src, dest, red)
        self.labels += int(red)
        event._manage_time_(action)
        event._manage_red()

        # Add nodes and event to the graph
        if isnewsrc:
            g.add_node(
                src,
                label=src.info(),
                type_=type(src).__name__,
                red=src.red,
                start=src.start,
                end=src.end,
            )
        if isnewdest:
            g.add_node(
                dest,
                label=dest.info(),
                type_=type(dest).__name__,
                red=dest.red,
                start=dest.start,
                end=dest.end,
            )

        # print("Graph: ", g, type(g))
        g.add_edge(
            src,
            dest,
            action=event.action,
            time=event.time,
            red=event.is_red,
            eventid=self.is_in_line("id", line),
        )
        return g

    def build_nodes(self, line, time, action):
        # Build Actor Process
        if self.is_in_line("actor", line) != 0:
            isnewsrc, src = self.build_undividual_node(
                line,
                self.is_in_line("actorID", line),
                self.is_in_line("actor", line),
                time,
                action,
            )
        else:
            isnewsrc, src = self.build_undividual_node(
                line,
                self.is_in_line("actorID", line),
                "PROCESS",
                time,
                action,
            )

        # Build Object
        isnewdest, dest = self.build_undividual_node(
            line,
            self.is_in_line("objectID", line),
            self.is_in_line("object", line),
            time,
            action,
        )

        return isnewsrc, src, isnewdest, dest

    def build_nodes_metadata(self, line, time, action):
        """Build source and destination nodes involved into the event.

        Parameters
        ----------
        line: dict
            The event to be added.
        time: str
            Time of the event, in format "2019-09-23T14:44:53.894-04:00".
        action: str
            Action of the event.

        Returns
        -------
        isnewsrc: bool
            Is the source node not already into the graph.
        src: Python Object
            The source node.
        isnewdest: bool
            Is the destination node not already into the graph.
        dest: Python Object
            The destination node.

        """
        # Object type
        type = self.is_in_line("object", line)

        # Build Actor Process
        if type == "PROCESS" and action == "CREATE":
            act = line.copy()
            act["pid"] = act["ppid"]
            act["ppid"] = 0
            act["properties"]["command_line"] = 0
            act["properties"]["image_path"] = self.is_in_line(
                "parent_image_path", self.is_in_line("properties", line)
            )
            act["properties"]["parent_image_path"] = 0
            isnewsrc, src = self.build_undividual_node(
                act, self.is_in_line("actorID", act), "PROCESS", time, action
            )
        else:
            isnewsrc, src = self.build_undividual_node(
                line, self.is_in_line("actorID", line), "PROCESS", time, action
            )

        # Build Object
        if type == "PROCESS":
            # Add command_line in command_line list
            command = self.is_in_line(
                "command_line", self.is_in_line("properties", line)
            )
            if command != 0:
                # self.command_line.append(command)
                self.command_line.append(
                    [
                        command,
                        self.cfg["MODEL"]["NODES_TYPES_ALL"].index("Process"),
                        self.cfg["MODEL"]["LST_ACTION"].index(action),
                    ]
                )

            # Add parent path in paths list
            p_image_path = self.is_in_line(
                "parent_image_path", self.is_in_line("properties", line)
            )
            if p_image_path != 0:
                # self.path.append(p_image_path)
                self.path.append(
                    [
                        p_image_path,
                        self.cfg["MODEL"]["NODES_TYPES_ALL"].index("Process"),
                        self.cfg["MODEL"]["LST_ACTION"].index(action),
                    ]
                )

            if action != "CREATE":
                line["ppid"] = line["pid"]
                line["pid"] = 0
                line["properties"]["image_path"] = 0
                line["properties"]["parent_image_path"] = 0
                line["properties"]["command_line"] = 0

            # Build process node
            isnewdest, dest = self.build_undividual_node(
                line,
                self.is_in_line("objectID", line),
                self.is_in_line("object", line),
                time,
                action,
            )

        if type == "FILE":
            # Add path in paths list
            file_path = self.is_in_line(
                "file_path", self.is_in_line("properties", line)
            )
            if file_path != 0:
                # self.path.append(file_path)
                self.path.append(
                    [
                        file_path,
                        self.cfg["MODEL"]["NODES_TYPES_ALL"].index("File"),
                        self.cfg["MODEL"]["LST_ACTION"].index(action),
                    ]
                )
            # Add new path in paths list
            new_path = self.is_in_line("new_path", self.is_in_line("properties", line))
            if new_path != 0:
                # self.path.append(new_path)
                self.path.append(
                    [
                        new_path,
                        self.cfg["MODEL"]["NODES_TYPES_ALL"].index("File"),
                        self.cfg["MODEL"]["LST_ACTION"].index(action),
                    ]
                )

            # Build file node
            isnewdest, dest = self.build_undividual_node(
                line,
                self.is_in_line("file_path", self.is_in_line("properties", line)),
                type,
                time,
                action,
            )

        elif type == "MODULE":
            # Add module path is paths list
            module_path = self.is_in_line(
                "module_path", self.is_in_line("properties", line)
            )
            if module_path != 0:
                # self.path.append(module_path)
                self.path.append(
                    [
                        module_path,
                        self.cfg["MODEL"]["NODES_TYPES_ALL"].index("Module"),
                        self.cfg["MODEL"]["LST_ACTION"].index(action),
                    ]
                )

            # Build Module node
            isnewdest, dest = self.build_undividual_node(
                line,
                self.is_in_line("module_path", self.is_in_line("properties", line)),
                type,
                time,
                action,
            )

        elif type == "FLOW":
            # Build Flow node
            isnewdest, dest = self.build_undividual_node(
                line,
                str(self.is_in_line("dest_ip", self.is_in_line("properties", line)))
                + ":"
                + str(
                    self.is_in_line("dest_port", self.is_in_line("properties", line))
                ),
                self.is_in_line("object", line),
                time,
                action,
            )
            # isnewdest, dest = self.build_undividual_node(line, self.is_in_line('dest_ip', self.is_in_line('properties', line)), self.is_in_line('object', line), time)

        elif type == "REGISTRY":
            # Build Registry node
            isnewdest, dest = self.build_undividual_node(
                line,
                self.is_in_line("key", self.is_in_line("properties", line)),
                self.is_in_line("object", line),
                time,
                action,
            )

        elif type == "SHELL":
            # Create identifier
            context = self.is_in_line(
                "context_info", self.is_in_line("properties", line)
            )

            host = context.split("Host ID = ")
            if len(host) > 1:
                host = host[1].split("\n")[0]
            else:
                host = ""

            user = context.split("User = ")
            if len(user) > 1:
                user = user[1].split("\n")[0]
            else:
                user = ""

            cmd = context.split("Command Name = ")
            if len(cmd) > 1:
                cmd = cmd[1].split("\n")[0]
            else:
                cmd = ""

            # Build Shell node
            if host + user + cmd == "":
                isnewdest, dest = self.build_undividual_node(
                    line,
                    self.is_in_line("objectID", line),
                    self.is_in_line("object", line),
                    time,
                    action,
                )
            else:
                isnewdest, dest = self.build_undividual_node(
                    line,
                    host + user + cmd,
                    self.is_in_line("object", line),
                    time,
                    action,
                )

        # Build unknown node type
        else:
            isnewdest, dest = self.build_undividual_node(
                line,
                self.is_in_line("objectID", line),
                self.is_in_line("object", line),
                time,
                action,
            )

        # For all node type, image_path is permanent
        # Add image_path to paths list
        image_path = self.is_in_line("image_path", self.is_in_line("properties", line))
        if image_path != 0:
            self.path.append(
                [
                    image_path,
                    self.cfg["MODEL"]["NODES_TYPES_ALL"].index(type.capitalize()),
                    self.cfg["MODEL"]["LST_ACTION"].index(action),
                ]
            )

        return isnewsrc, src, isnewdest, dest

    def build_undividual_node(self, line, id_, type_, time, action):
        """Build a node.

        Parameters
        ----------
        line: dict
            The event to be added.
        id_: str
            Identifier of the node.
        type_: str
            Type of the node.
        time: str
            Time of the event, in format "2019-09-23T14:44:53.894-04:00".
        action: str
            Action of the event.

        Returns
        -------
        Bool:
            Is the node new ?
        Node: Python Object
            The node created.
        """
        if id_ in self.nodes:
            if self.metadata:
                if type_.capitalize() != type(self.nodes[id_]).__name__:
                    return False, None
                if type_ == "PROCESS":
                    self.nodes[id_].set_pid(self.is_in_line("pid", line))
                    self.nodes[id_].set_ppid(self.is_in_line("ppid", line))
                    self.nodes[id_].set_command_line(
                        self.is_in_line(
                            "command_line", self.is_in_line("properties", line)
                        )
                    )
                    self.nodes[id_].set_image_path(
                        self.is_in_line(
                            "image_path", self.is_in_line("properties", line)
                        )
                    )
                    self.nodes[id_].set_parent_image_path(
                        self.is_in_line(
                            "parent_image_path", self.is_in_line("properties", line)
                        )
                    )
            return False, self.nodes[id_]
        else:
            node = self.build_node_instance(
                type_,
                id_,
                period(time, -self.duration),
                period(time, self.duration),
                self.is_in_line("properties", line),
                self.is_in_line("principal", line),
            )
            self.nodes[id_] = node
            return True, node

    def build_node_instance(self, type, id_, start, end, properties, principal):
        type = type.upper()
        """Build Python Object node.

        Parameters
        ----------
        type: str
            Type of the node.
        id_: str
            Identifier of the node.
        start: str
            Time to set as the start time of the node life.
        end: str
            Time to set as the end time of the node life.
        properties: dict
            Properties such as size, image_path, etc.
        principal: str
            Principal of the process.

        Returns
        -------
        node: Python Object
            The python object node created, with the right type.

        """
        if type == "FLOW":
            Classe = getattr(self.node_dataset, "Flow")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("image_path", properties),
                self.is_in_line("src_ip", properties),
                self.is_in_line("src_port", properties),
                self.is_in_line("dest_ip", properties),
                self.is_in_line("dest_port", properties),
                self.is_in_line("is_inbound", properties),
                self.is_in_line("size", properties),
                self.is_in_line("l4protocol", properties),
            )
        elif type == "FILE":
            Classe = getattr(self.node_dataset, "File")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("image_path", properties),
                self.is_in_line("file_path", properties),
                self.is_in_line("new_path", properties),
                self.is_in_line("info_class", properties),
                self.is_in_line("size", properties),
            )
        elif type == "PROCESS":
            Classe = getattr(self.node_dataset, "Process")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("image_path", properties),
                self.is_in_line("parent_image_path", properties),
                self.is_in_line("command_line", properties),
                self.is_in_line("user", properties),
                self.is_in_line("sid", properties),
                self.is_in_line("pid", properties),
                self.is_in_line("ppid", properties),
                principal,
            )
        elif type == "MODULE":
            Classe = getattr(self.node_dataset, "Module")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("image_path", properties),
                self.is_in_line("module_path", properties),
            )
        elif type == "THREAD":
            Classe = getattr(self.node_dataset, "Thread")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("image_path", properties),
                self.is_in_line("pid", properties),
                self.is_in_line("tid", properties),
                self.is_in_line("is_src_tag", properties),
            )
        elif type == "REGISTRY":
            Classe = getattr(self.node_dataset, "Registry")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("image_path", properties),
                self.is_in_line("key", properties),
                self.is_in_line("type_", properties),
            )
        elif type == "TASK":
            Classe = getattr(self.node_dataset, "Task")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("image_path", properties),
                self.is_in_line("task_process_uuid", properties),
                self.is_in_line("path", properties),
                self.is_in_line("task_name", properties),
            )
        elif type == "SHELL":
            Classe = getattr(self.node_dataset, "Shell")
            return Classe(id_, start, end, self.is_in_line("image_path", properties))
        elif type == "HOST":
            Classe = getattr(self.node_dataset, "Host")
            return Classe(id_, start, end, self.is_in_line("image_path", properties))
        elif type == "SERVICE":
            Classe = getattr(self.node_dataset, "Service")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("image_path", properties),
                self.is_in_line("name", properties),
                self.is_in_line("start_type", properties),
                self.is_in_line("service_type", properties),
            )
        elif type == "USER_SESSION":
            Classe = getattr(self.node_dataset, "User_session")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("privileges", properties),
                self.is_in_line("logon_id", properties),
                self.is_in_line("requesting_logon_id", properties),
                self.is_in_line("user", properties),
            )
        elif type == "MMAPED_FILE":  # type == "MAP_ANONYMOUS"
            Classe = getattr(self.node_dataset, "Mmaped_file")
            return Classe(
                id_,
                start,
                end,
                self.is_in_line("privileges", properties),
                self.is_in_line("logon_id", properties),
                self.is_in_line("requesting_logon_id", properties),
                self.is_in_line("user", properties),
            )
        elif type == "PROCESS_MEMORY":
            Classe = getattr(self.node_dataset, "Process_memory")
            return Classe(id_, start, end)
        elif type == "PATH":
            Classe = getattr(self.node_dataset, "Path")
            return Classe(id_, start, end)
        elif type == "SOCKET":
            Classe = getattr(self.node_dataset, "Socket")
            return Classe(id_, start, end)
        elif type == "ADDRESS":
            Classe = getattr(self.node_dataset, "Address")
            return Classe(id_, start, end)
        elif type == "LINK":
            Classe = getattr(self.node_dataset, "Link")
            return Classe(id_, start, end)
        elif type == "SHM":
            Classe = getattr(self.node_dataset, "Shm")
            return Classe(id_, start, end)
        elif type == "BLOCK":
            Classe = getattr(self.node_dataset, "Block")
            return Classe(id_, start, end)
        elif type == "ARGV":
            Classe = getattr(self.node_dataset, "Argv")
            return Classe(id_, start, end)
        elif type == "XATTR":
            Classe = getattr(self.node_dataset, "Xattr")
            return Classe(id_, start, end)
        elif type == "IATTR":
            Classe = getattr(self.node_dataset, "Iattr")
            return Classe(id_, start, end)
        elif type == "PIPE":
            Classe = getattr(self.node_dataset, "Pipe")
            return Classe(id_, start, end)
        else:
            Classe = getattr(self.node_dataset, "Node")
            return Classe(id_, start, end)

    def info(self):
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "labels": self.labels,
            "start": self.data[0]["timestamp"],
            "end": self.data[-1]["timestamp"],
            "logs": len(self.data),
        }


def extract_data(c, k, file, duration, base, metadata, node_dataset, cfg):
    print("File: ", file)
    graphs, cmds, paths = {}, {}, {}
    graph = []
    start = None
    end = None
    # start = "2019-09-22T14:00:00.000-04:00"
    # end = period(start, duration)
    try:
        if file.endswith(".gz"):
            f = gzip.open(file, "r")
        else:
            f = open(file, "r")

        for line in tqdm(f):
            ob = json.loads(line)
            if start is None:  # Set the start of the first graph
                start = ob["timestamp"]
                # print("Start: ", start)
                if metadata:
                    start = round_duration(start, duration)
                # print("Start1: ", start)
                end = period(start, duration)
                graph.append(ob)
                # print(start)

            elif start <= ob["timestamp"] < end:  # Add event to the graph
                graph.append(ob)

            elif len(graph) > 0:  # Create the current graph, and start an other one
                # print(start)
                ob_graph = Graph_builder(
                    graph, duration, metadata, node_dataset, cfg
                )  # Create the graph
                save_pkl(
                    ob_graph.get_g(),
                    base + f"graph_data/optc_{c}/{c}_{duration}_{k}_{start}.pkl",
                )
                if metadata:
                    save_pkl(
                        ob_graph.get_command_lines(),
                        base
                        + f"feature_data/optc_{c}/cmds/{c}_{duration}_{k}_{start}.pkl",
                    )
                    save_pkl(
                        ob_graph.get_path(),
                        base
                        + f"feature_data/optc_{c}/paths/{c}_{duration}_{k}_{start}.pkl",
                    )
                    cmds[start] = (
                        base
                        + f"feature_data/optc_{c}/cmds/{c}_{duration}_{k}_{start}.pkl"
                    )
                    paths[start] = (
                        base
                        + f"feature_data/optc_{c}/paths/{c}_{duration}_{k}_{start}.pkl"
                    )
                graphs[start] = (
                    base + f"graph_data/optc_{c}/{c}_{duration}_{k}_{start}.pkl"
                )
                del ob_graph

                # Start a new graph
                graph = []
                graph.append(ob)
                start = ob["timestamp"]
                if metadata:
                    start = round_duration(start, duration)
                end = period(start, duration)

        if len(graph):  # Save the last graph
            # print("NB event: ", len(graph))
            ob_graph = Graph_builder(graph, duration, metadata, node_dataset, cfg)
            save_pkl(
                ob_graph.get_g(),
                base + f"graph_data/optc_{c}/{c}_{duration}_{k}_{start}.pkl",
            )
            if metadata:
                save_pkl(
                    ob_graph.get_command_lines(),
                    base + f"feature_data/optc_{c}/cmds/{c}_{duration}_{k}_{start}.pkl",
                )
                save_pkl(
                    ob_graph.get_path(),
                    base
                    + f"feature_data/optc_{c}/paths/{c}_{duration}_{k}_{start}.pkl",
                )
                cmds[start] = (
                    base + f"feature_data/optc_{c}/cmds/{c}_{duration}_{k}_{start}.pkl"
                )
                paths[start] = (
                    base + f"feature_data/optc_{c}/paths/{c}_{duration}_{k}_{start}.pkl"
                )
            graphs[start] = base + f"graph_data/optc_{c}/{c}_{duration}_{k}_{start}.pkl"
            del ob_graph
        return graphs, cmds, paths

    except Exception as e:
        print(f"Error {e}")
        traceback.print_exc()
        exit(1)


def main( clients, duration, logs, dataset):
    print("Start task graph")
    utils_dataset = importlib.import_module(f"data.{dataset}_utils")
    node_dataset = importlib.import_module(f"data.{dataset}_graph_classes")
    cfg = open_config(dataset)
    metadata = True

    graphs = {c: {} for c in clients}
    cmds = {c: {} for c in clients}
    paths = {c: {} for c in clients}
    for c in clients:
        print("Client : ", c, logs)
        data = utils_dataset.logs_from_folder(logs, c)
        print(data)
        for k, d in enumerate(data):
            graphs_, cmds_, paths_ = extract_data(
                c, k, d, duration, BASE, metadata, node_dataset, cfg
            )
            graphs[c].update(graphs_)
            if metadata:
                cmds[c].update(cmds_)
                paths[c].update(paths_)

    save_pkl(graphs, f"{BASE}/graph_data/graphs.pkl")
    if metadata:
        save_pkl(cmds, f"{BASE}/feature_data/cmds.pkl")
        save_pkl(paths, f"{BASE}/feature_data/paths.pkl")
    print("End task graph")
    return
