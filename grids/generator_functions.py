import networkx as nx
import numpy as np


def generate_4x4_grids(costs, seed=0):
    size = 4

    if costs == "uniform":
        G = nx.grid_graph(dim=(size, size))
        for e in G.edges():
            G.edges[e]["cost"] = lambda x: 1 + x / 100
        return G

    elif costs == "random":
        np.random.seed(seed=seed)
        rG = nx.grid_graph(dim=(size, size))
        for e in rG.edges():
            rG.edges[e]["cost"] = lambda x: np.random.random() + x / 100

        return rG

    elif costs == "braess":
        G = nx.grid_graph(dim=(size, size))
        edge_list = [edge for edge in G.edges]

        dG = nx.DiGraph(incoming_graph_data=edge_list)

        dG.add_edge((0, 1), (1, 0), cost=lambda x: 0)
        dG.add_edge((0, 2), (1, 1), cost=lambda x: 0)
        dG.add_edge((0, 3), (1, 2), cost=lambda x: 0)
        dG.add_edge((1, 1), (2, 0), cost=lambda x: 0)
        dG.add_edge((1, 2), (2, 1), cost=lambda x: 0)
        dG.add_edge((1, 3), (2, 2), cost=lambda x: 0)
        dG.add_edge((2, 1), (3, 0), cost=lambda x: 0)
        dG.add_edge((2, 2), (3, 1), cost=lambda x: 0)
        dG.add_edge((2, 3), (3, 2), cost=lambda x: 0)

        for node in dG.nodes():
            edges = dG.edges(node)
            for edge in edges:
                if edge[1][0] == edge[0][0] + 1:
                    # set cost to variable
                    dG.edges[edge]["cost"] = lambda x: x / 100
                elif edge[1][1] == edge[0][1] + 1:
                    # set cost to fixed
                    dG.edges[edge]["cost"] = lambda x: 1
        return dG
