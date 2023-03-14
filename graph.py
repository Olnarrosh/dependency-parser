class Edge:
    def __init__(self, origin, target, cost, label):
        self.origin = origin
        self.target = target
        self.cost = cost
        self.label = label

    def __repr__(self):
        return f"Edge({self.origin}, {self.target}, {self.cost}, '{self.label}')"

    def __str__(self):
        return f"{self.origin} -({self.label} : {self.cost})-> {self.target}"

class Graph:
    # length parameter includes root
    # edges are given as (origin node, target node, cost, label)
    def __init__(self, length: int, edges: list[(int, int, float, str)]) -> None:
        self.length = length
        self.edges = [Edge(*e) for e in edges]

    def get_edge(self, origin: int, target: int, edges: list[Edge] = None) -> Edge:
        for edge in edges or self.edges:
            if (origin == None or edge.origin == origin) and (target == None or edge.target == target):
                return edge
        return None

    def cle(self):
        """Performs (recursively) the Chu-Liu-Edmonds algorithm on the graph.
        Returns the set of edges forming the minimum spanning tree.
        """
        # find minimum incoming edge per node
        edges_by_target = [list(filter(lambda e: e.target == i, self.edges)) for i in range(1, self.length)]
        min_edges = [min(edges_by_target[n], key=lambda x: x.cost) if edges_by_target[n] else None for n in range(self.length - 1)]
        # identify any cycle in those edges
        cycle = []
        for n in range(1, self.length):
            path = [min_edges[n - 1]]
            while path[-1] and path[-1] not in path[:-1]:
                origin = path[-1].origin
                if origin == n:
                    cycle = path
                    break
                path.append(min_edges[origin - 1])
            else:
                continue
            break
        if not cycle:
            return min_edges
        # contract the cycle
        cycle_nodes = [e.target for e in path]
        new_node = self.length
        contracted_edges = []
        old_edges = {}
        for edge in self.edges:
            if edge.origin in cycle_nodes and edge.target not in cycle_nodes:
                if not any(e.origin == new_node and e.target == edge.target and e.cost < edge.cost for e in contracted_edges):
                    cycle_edge = Edge(new_node, edge.target, edge.cost, edge.label)
                    contracted_edges.append(cycle_edge)
                    old_edges[cycle_edge] = edge
            elif edge.origin not in cycle_nodes and edge.target in cycle_nodes:
                cost = edge.cost - self.get_edge(min_edges[edge.target - 1].origin, edge.target).cost
                if not any(e.origin == edge.origin and e.target == new_node and e.cost < cost for e in contracted_edges):
                    cycle_edge = Edge(edge.origin, new_node, cost, edge.label)
                    contracted_edges.append(cycle_edge)
                    old_edges[cycle_edge] = edge
            elif edge.origin not in cycle_nodes and edge.target not in cycle_nodes:
                contracted_edges.append(edge)
                old_edges[edge] = edge
        # find minimal edges for next depth of the algorithm
        new_by_origin = [{edge for edge in contracted_edges if edge.origin == i} for i in range(self.length + 1)]
        new_by_target = [{edge for edge in contracted_edges if edge.target == i} for i in range(1, self.length + 1)]
        new_edges_min = []
        for origin in range(self.length + 1):
            for target in range(self.length):
                local_edges = new_by_origin[origin] & new_by_target[target]
                if local_edges:
                    new_edges_min.append(min(local_edges, key=lambda e: e.cost))
        # if necessary, further reduce the resulting graph
        g = Graph(self.length + 1, [])
        g.edges = new_edges_min
        subtree = list(filter(None, g.cle()))
        # resolve cycle
        restored_edges = [old_edges[e] for e in subtree]
        for n in cycle_nodes:
            restored_edges.append(min_edges[n - 1])
        for edge in subtree:
            if edge.target == new_node and self.get_edge(min_edges[(old_node := old_edges[edge].target) - 1].origin, old_node) in restored_edges:
                restored_edges.remove(self.get_edge(min_edges[old_node - 1].origin, old_node))
                break
        return restored_edges
