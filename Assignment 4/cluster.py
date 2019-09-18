from collections import Counter, defaultdict, deque
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import pickle
import copy
import math
import urllib.request
from TwitterAPI import TwitterAPI

def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Done for you.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('names.txt', delimiter=':')

def girvan_newman(G, depth=0):
    """ Recursive implementation of the girvan_newman algorithm.
    See http://www-rohan.sdsu.edu/~gawron/python_for_ss/course_core/book_draft/Social_Networks/Networkx.html
    
    Args:
    G.....a networkx graph

    Returns:
    A list of all discovered communities,
    a list of lists of nodes. """

    if G.order() == 1:
        return [G.nodes()]
    
    def find_best_edge(G0):
        eb = nx.edge_betweenness_centrality(G0)
        # eb is dict of (edge, score) pairs, where higher is better
        # Return the edge with the highest score.
        return sorted(eb.items(), key=lambda x: x[1], reverse=True)[0][0]
    # Each component is a separate community. We cluster each of these.
    components = [c for c in nx.connected_component_subgraphs(G)]
    indent = '   ' * depth  # for printing
    while len(components) <= 3:
        edge_to_remove = find_best_edge(G)
        #print(indent + 'removing ' + str(edge_to_remove))
        G.remove_edge(*edge_to_remove)
        components = [c for c in nx.connected_component_subgraphs(G)]
    result = [c for c in components]
    #print(indent + 'components=' + str(result))
#     for c in components:
#         result.extend(girvan_newman(c, depth + 1))
    return result

def draw_network(graph, filename):
    """
    Draw the network to a file. Only label the candidate nodes; the friend
    nodes should have no labels (to reduce clutter).

    Methods you'll need include networkx.draw_networkx, plt.figure, and plt.savefig.

    Your figure does not have to look exactly the same as mine, but try to
    make it look presentable.
    """
    nx.draw_networkx(graph, node_color='red', alpha =.6, width =.4, font_size = 14, node_size = 100, edge_color ='blue', with_labels=False)
    plt.axis("off")
    plt.savefig(filename)
    plt.figure(figsize=(18,12))

def main():
    graph = read_graph()
    graph_1= graph.copy() 
    b = open('cluster.txt','w')
    print('Graph has %d nodes and %d dges' %
          (graph.order(), graph.number_of_edges()))
    b.write('Graph has %d nodes and %d edges' % (graph.order(), graph.number_of_edges()))
    
    print("original graph")
    draw_network(graph,"cluster1.png")
    
    clusters = girvan_newman(graph_1,depth=3)
    print(len(clusters))
    b.write('\n\nNumber of communities discovered: %d' % len(clusters))
    b.write('\n\nAverage number of users per community: %s' % (graph.order()/ len(clusters)))
    print('Cluster 1 has %d nodes\nCluster 2 has %d nodes\nCluster 3 has %d nodes\nCluster 4 has %d nodes ' %
          (clusters[0].order(), clusters[1].order(), clusters[2].order(), clusters[3].order()))
    b.write('\n\nCluster 1 has %d nodes\nCluster 2 has %d nodes\nCluster 3 has %d nodes\nCluster 4 has %d nodes' % (clusters[0].order(), clusters[1].order(), clusters[2].order(), clusters[3].order()))
    
    print('Cluster 2 nodes:')
    print(clusters[1].nodes())
    #b.write('Cluster 2 nodes:')
    #b.write('clusters[1].nodes()')
    
    print('Cluster 3 nodes:')
    print(clusters[2].nodes())
    #b.write('Cluster 3 nodes:')
    #b.write('clusters[2].nodes()')
    
    print('Cluster 4 nodes:')
    print(clusters[3].nodes())
    #b.write('Cluster 4 nodes:')
    #b.write('clusters[3].nodes()')
    
    #print("second figure")
    
    draw_network(graph_1,"cluster2.png")
    b.close()

if __name__ == '__main__':
    main()

