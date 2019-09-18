import networkx as nx
import matplotlib.pyplot as plt

def example_graph():
    g = nx.Graph()
    g.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), ('D', 'E'), ('D', 'F'), ('D', 'G'), ('E', 'F'), ('G', 'F')])
    return g

def jaccard_wt(graph, node):
    # set of neighbors for our picked node
    A = set(graph.neighbors(node))
    neighbors = A
    jaccard_scores = []
    #condition for finding the nodes that are not neighbors of node
    for n in graph.nodes():
        if (n != node) and (n not in neighbors) :
            # set of neighbors for the nodes that pass the above condition
            B = set(graph.neighbors(n))
            neighbors2 = B
            node_score = 0
            A_Degrees = 0
            B_Degrees = 0
            for i in neighbors:
                A_Degrees = A_Degrees + len(list(graph.neighbors(i)))
            for j in neighbors2:
                B_Degrees = B_Degrees + len(list(graph.neighbors(j)))
            for common in list(neighbors & neighbors2):
                node_score = node_score + ( 1 / len(list(graph.neighbors(common))))

            jaccard_scores.append(((node, n), node_score / ((1 / A_Degrees) + (1 / B_Degrees))))

    jaccard_scores = sorted(jaccard_scores, key=lambda x: -x[1])
    return jaccard_scores

if __name__ == '__main__':
    g = example_graph()
    nx.draw_networkx(g, with_labels=True)
    plt.show()
    print(jaccard_wt(g, 'B'))