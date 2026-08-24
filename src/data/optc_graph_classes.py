import networkx as nx
import numpy as np
import os

from others.w2v import eval_for_encoding

from utils.utils import encoding_parent_son, encoding_sid, get_duration, open_config


cfg = open_config("optc")


class Event:
    def __init__(self, time, action, src, dest, is_red):
        self.time = time
        self.action = action
        self.src = src
        self.dest = dest
        self.is_red = is_red
        self.nb_action = 1

    def _manage_red(self):
        """Perform the work of manage red."""
        self.src.set_red(self.is_red)
        self.dest.set_red(self.is_red)

    def _manage_time(self):
        """Perform the work of manage time."""
        self._add_alive_node_time(self.src)
        self._add_alive_node_time(self.dest)

    def _manage_time_(self, action):
        # self._add_alive_node_time_(self.src, 0)
        """Perform the work of manage time."""
        self._add_alive_node_time_(self.dest, action)

    def add_nb_action(self):
        """Perform the work of add nb action."""
        self.nb_action += 1
        return self.nb_action

    def _add_alive_node_time_(self, node, action=None):
        """Perform the work of add alive node time."""
        if node is None:
            return
        if action in cfg["MODEL"]["LST_ACTION_START"]:
            node.set_start(self.time)
        elif action in cfg["MODEL"]["LST_ACTION_END"]:
            node.set_end(self.time)
        return

    def _add_alive_node_time(self, node):
        """Perform the work of add alive node time."""
        if node is None:
            return
        if self.time < node.get_start():
            node.set_start(self.time)
        elif self.time > node.get_end():
            node.set_end(self.time)
        return

    def info(self):
        """Perform the work of info."""
        return f"{self.time}, {self.action}, src: {self.src.info()}, dest: {self.dest.info()}"


class Node:
    def __init__(self, id_, start, end, red=0):
        self.id = id_
        self.start = start
        self.end = end
        self.enc_node = None
        self.enc_local_node = None
        self.enc_local_event = None
        self.red = red

    def __repr__(self):
        # Retourne une représentation textuelle de l'objet
        return f"{type(self).__name__}: {self.id})"

    def get_start(self):
        """Retrieve get start."""
        return self.start

    def get_end(self):
        """Retrieve get end."""
        return self.end

    def set_red(self, red):
        """Store set red."""
        if self.red == 1:
            return
        else:
            self.red = red

    def lifetimenode(self):
        """Perform the work of lifetimenode."""
        min_life = self.lifetime()
        max_life = self.lifetime()
        sum_life = self.lifetime()
        return [min_life, max_life, sum_life]

    def lifetime(self):
        """Perform the work of lifetime."""
        life = get_duration(self.start, self.end)
        return life

    def set_start(self, start):
        """Store set start."""
        self.start = start

    def set_end(self, end):
        """Store set end."""
        self.end = end

    def set_enc_node(self, g):
        """Store set enc node."""
        if self.enc_node is None:
            self.enc_node = self.encoding(g)
            return self.enc_node
        else:
            return self.enc_node

    def node_time_for_node_type(self, g):
        """Perform the work of node time for node type."""
        node_time_out_freq = {}
        node_time_in_freq = {}

        for type_ in cfg["MODEL"]["NODES_TYPES_ALL"]:
            node_time_out_freq[type_] = []
            node_time_in_freq[type_] = []

        edges = g.out_edges(self, data=True)
        for edge in edges:
            n = edge[1]
            node_type = type(n).__name__
            if node_type in node_time_out_freq:
                node_time_out_freq[node_type].append(n.lifetime())
            else:
                node_time_out_freq[node_type] = [n.lifetime()]

        edges = g.in_edges(self, data=True)
        for edge in edges:
            n = edge[0]
            node_type = type(n).__name__
            if node_type in node_time_out_freq:
                node_time_in_freq[node_type].append(n.lifetime())
            else:
                node_time_in_freq[node_type] = [n.lifetime()]

        a = []
        for _, freq in node_time_out_freq.items():
            if len(freq) < 1:
                a.append(0)
            else:
                a.append(np.mean(freq))

        for _, freq in node_time_in_freq.items():
            if len(freq) < 1:
                a.append(0)
            else:
                a.append(np.mean(freq))

        return a

    def set_enc_local_node(self, g):
        """Store set enc local node."""
        if self.enc_local_node is None:
            lst = self.encoding_local_node(g)
            time = self.lifetimenode()
            lst.extend(time)
            neigh_time = self.node_time_for_node_type(g)
            lst.extend(neigh_time)
            self.enc_local_node = lst
            return lst
        else:
            return self.enc_local_node

    def set_enc_local_event(self, g, time):
        """Store set enc local event."""
        if self.enc_local_event is None:
            self.enc_local_event = self.encoding_local_event(g, time)
            return self.enc_local_event
        else:
            return self.enc_local_event

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def encoding_local_event(self, g, time):
        """Perform the work of encoding local event."""
        b = self.last_and_next(g.in_edges(self, data=True), time)
        c = self.last_and_next(g.out_edges(self, data=True), time)

        return b + c

    def last_and_next(self, edges, time):
        """Perform the work of last and next."""
        t0 = str(0)
        edge0 = None
        t1 = str(0)
        edge1 = None
        for edge in edges:
            if edge[2]["time"] > t0 and edge[2]["time"] < time:
                edge0 = edge
            elif edge[2]["time"] < t1 and edge[2]["time"] > time:
                edge1 = edge
            else:
                continue

        last_action = [0 for type_ in cfg["MODEL"]["LST_ACTION"]]
        next_action = [0 for type_ in cfg["MODEL"]["LST_ACTION"]]
        if edge0 is not None and edge1 is not None:
            last_action[edge0[2]["action"]] = 1
            next_action[edge1[2]["action"]] = 1

            last_time = time - edge0[2]["time"]
            next_time = edge1[2]["time"] - time

        else:
            last_time = 0
            next_time = 0

        return last_action + next_action + [last_time, next_time]

    def encoding_local_node(self, g):
        """Perform the work of encoding local node."""
        degree = nx.degree(g, self)
        in_degree = len(g.in_edges(self))
        out_degree = len(g.out_edges(self))

        edge_types_out_freq = {}
        edge_types_in_freq = {}
        node_types_out_freq = {}
        node_types_in_freq = {}

        for type_ in cfg["MODEL"]["LST_ACTION"]:
            edge_types_out_freq[type_] = 0
            edge_types_in_freq[type_] = 0

        for type_ in cfg["MODEL"]["NODES_TYPES_ALL"]:
            node_types_out_freq[type_] = 0
            node_types_in_freq[type_] = 0

        edges = g.out_edges(self, data=True)
        for edge in edges:
            edge_type = edge[2]["action"]
            if edge_type in edge_types_out_freq:
                edge_types_out_freq[edge_type] += 1
            else:
                edge_types_out_freq[edge_type] = 1

            node_type = type(edge[1]).__name__
            if node_type in node_types_out_freq:
                node_types_out_freq[node_type] += 1
            else:
                node_types_out_freq[node_type] = 1

        edges = g.in_edges(self, data=True)
        for edge in edges:
            edge_type = edge[2]["action"]
            if edge_type in edge_types_in_freq:
                edge_types_in_freq[edge_type] += 1
            else:
                edge_types_in_freq[edge_type] = 1

            node_type = type(edge[0]).__name__
            if node_type in node_types_in_freq:
                node_types_in_freq[node_type] += 1
            else:
                node_types_in_freq[node_type] = 1

        a = [freq for type, freq in edge_types_out_freq.items()]
        b = [freq for type, freq in edge_types_in_freq.items()]
        c = [freq for type, freq in node_types_out_freq.items()]
        d = [freq for type, freq in node_types_in_freq.items()]
        # e = NODES_TYPES.index(type(self).__name__)

        return (
            a
            + b
            + c
            + d
            + [degree, degree, in_degree, in_degree, out_degree, out_degree]
        )

    def info(self):
        """Perform the work of info."""
        return f"Node alives from {self.start} to {self.end}"


class Process(Node):
    def __init__(
        self,
        id_,
        start,
        end,
        image_path,
        parent_image_path,
        command_line,
        user,
        sid,
        pid,
        ppid,
        principal,
        red=0,
    ):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path
        self.parent_image_path = parent_image_path
        self.command_line = command_line
        self.user = user
        self.sid = sid
        self.pid = pid
        self.ppid = ppid
        self.principal = principal

    def encoding_principal(self):
        """Perform the work of encoding principal."""
        if "AUTHORITY" in self.principal:
            return 1
        elif "SYSTEMIA" in self.principal:
            return 2
        elif "Window Manager" in self.principal:
            return 3
        elif "Font Driver Host" in self.principal:
            return 4
        elif "" in self.principal:
            return 5
        else:
            return 6

    def set_pid(self, pid):
        """Store set pid."""
        if pid != 0 and self.pid == 0:
            self.pid = pid
        return pid

    def set_ppid(self, ppid):
        """Store set ppid."""
        if ppid != 0 and self.ppid == 0:
            self.ppid = ppid
        return ppid

    def set_command_line(self, cmd_line):
        """Store set command line."""
        if cmd_line != 0 and self.command_line == 0:
            self.command_line = cmd_line
        return cmd_line

    def set_image_path(self, image_path):
        """Store set image path."""
        if image_path != 0 and self.image_path == 0:
            self.image_path = image_path
        return image_path

    def set_parent_image_path(self, p_image_path):
        """Store set parent image path."""
        if p_image_path != 0 and self.parent_image_path == 0:
            self.parent_image_path = p_image_path
        return p_image_path

    def encoding(self, model):
        """Perform the work of encoding."""
        p_and_son = [0, 0, 0, 0]
        p_and_son = encoding_parent_son(
            p_and_son, self.parent_image_path, self.image_path
        )
        sid_enc = encoding_sid([self.sid])

        return (
            eval_for_encoding(model[1], self.image_path, False, cfg).tolist()
            + eval_for_encoding(model[1], self.parent_image_path, False, cfg).tolist()
            + eval_for_encoding(model[0], self.command_line, True, cfg).tolist()
            + sid_enc
            + p_and_son
        )

    def info(self):
        """Perform the work of info."""
        return f"Process {self.id} alives from {self.start} to {self.end}, with PID {self.pid} and PPID {self.ppid}. User is {self.user}. {self.red}"


class File(Node):
    def __init__(
        self, id_, start, end, image_path, file_path, new_path, info_class, size, red=0
    ):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = self._uniformized(image_path)
        self.file_path = self._uniformized(file_path)
        self.new_path = self._uniformized(new_path)
        self.info_class = info_class
        self.size = int(size)
        self.extension = self._parse_path(file_path)

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def _parse_path(self, file_path):
        """Perform the work of parse path."""
        if file_path == 0:
            return file_path
        else:
            lst_elt_path = os.path.splitext(file_path)
            if len(lst_elt_path) == 2:
                return lst_elt_path[1]
            else:
                return lst_elt_path[0]

    def _uniformized(self, path):
        """Perform the work of uniformized."""
        if path == 0:
            return path
        return path.replace("\Device\HarddiskVolume1", "C:")

    def info(self):
        """Perform the work of info."""
        return f"File {self.id} alives from {self.start} to {self.end}, with size {self.size}, extension {self.extension}, info classe {self.info_class}."


class Flow(Node):
    def __init__(
        self,
        id_,
        start,
        end,
        image_path,
        src_ip,
        src_port,
        dest_ip,
        dest_port,
        is_inbound,
        size,
        l4protocol,
        red=0,
    ):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path
        self.src_ip = src_ip
        self.src_port = src_port
        self.dest_ip = dest_ip
        self.dest_port = dest_port
        self.is_inbound = is_inbound
        self.size = int(size)
        self.l4protocol = l4protocol

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Flow {self.id} alives from {self.start} to {self.end}, with size {self.size}, from {self.src_ip}:{self.src_port} to {self.dest_ip}:{self.dest_port}."


class Module(Node):
    def __init__(self, id_, start, end, image_path, module_path, red=0):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path
        self.module_path = module_path

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Module {self.id} alives from {self.start} to {self.end}. Module path is {self.module_path}."


class Thread(Node):
    def __init__(self, id_, start, end, image_path, pid, tid, is_src_targ, red=0):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path
        self.pid = pid
        self.tid = tid
        self.is_src_targ = is_src_targ

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Thread {self.id} alives from {self.start} to {self.end}, with tid {self.tid} and PID {self.pid}."


class Registry(Node):
    def __init__(self, id_, start, end, image_path, key, type, red=0):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path
        self.key = key
        self.type = type

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Registry {self.id} alives from {self.start} to {self.end}, with key {self.key} and type {self.type}."


class Task(Node):
    def __init__(
        self, id_, start, end, image_path, task_process_uuid, path, task_name, red=0
    ):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path
        self.task_process_uuid = task_process_uuid
        self.path = path
        self.task_name = task_name

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Task {self.id} alives from {self.start} to {self.end}, with task {self.task_name} {self.task_process_uuid}."


class Shell(Node):
    def __init__(self, id_, start, end, image_path, red=0):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Shell {self.id} alives from {self.start} to {self.end}."


class Host(Node):
    def __init__(self, id_, start, end, image_path, red=0):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Host {self.id} alives from {self.start} to {self.end}."


class Service(Node):
    def __init__(
        self, id_, start, end, image_path, name, start_type, service_type, red=0
    ):
        super().__init__(id_, start, end, red)
        super().info()
        self.image_path = image_path
        self.name = name
        self.start_type = start_type
        self.service_type = service_type

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Service {self.id} alives from {self.start} to {self.end}, with type {self.service_type} and named {self.name}."


class User_session(Node):
    def __init__(
        self, id_, start, end, privileges, logon_id, requesting_logon_id, user, red=0
    ):
        super().__init__(id_, start, end, red)
        super().info()
        self.privileges = privileges
        self.logon_id = logon_id
        self.requesting_logon_id = requesting_logon_id
        self.user = user

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"User_session {self.id} alives from {self.start} to {self.end}, with priviledge {self.privileges}, user {self.user} and logon id {self.logon_id}."


class Mmaped_file(Node):
    def __init__(self, id_, start, end, red=0):
        super().__init__(id_, start, end, red)
        super().info()

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Nmap_file {self.id} alives from {self.start} to {self.end}."


class Path(Node):
    def __init__(self, id_, start, end, red=0):
        super().__init__(id_, start, end, red)
        super().info()

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Path {self.id} alives from {self.start} to {self.end}."


class Socket(Node):
    def __init__(self, id_, start, end, red=0):
        super().__init__(id_, start, end, red)
        super().info()

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Nmap_file {self.id} alives from {self.start} to {self.end}."


class Address(Node):
    def __init__(self, id_, start, end, red=0):
        super().__init__(id_, start, end, red)
        super().info()

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Nmap_file {self.id} alives from {self.start} to {self.end}."


class Link(Node):
    def __init__(self, id_, start, end, red=0):
        super().__init__(id_, start, end, red)
        super().info()

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Nmap_file {self.id} alives from {self.start} to {self.end}."


class Process_memory(Node):
    def __init__(self, id_, start, end, red=0):
        super().__init__(id_, start, end, red)
        super().info()

    def encoding(self, model):
        """Perform the work of encoding."""
        return [1]

    def info(self):
        """Perform the work of info."""
        return f"Nmap_file {self.id} alives from {self.start} to {self.end}."
