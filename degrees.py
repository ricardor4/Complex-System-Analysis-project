import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


def propagate_simple(G, p):
    to_infect = set([])
    # Find infected nodes
    for v in G.nodes():
        if G.nodes[v]['infected'] == 'I':
            # infect all
            for w in nx.neighbors(G, v):
                if G.nodes[w]['infected'] == 'S':
                    if random.random() < p:
                        to_infect.add(w)
                        break
    # Infect marked nodes
    for v in to_infect:
        G.nodes[v]['infected'] = 'I'


def recover(G, beta):
    to_recover = set([])
    # Find infected nodes
    for v in G.nodes():
        if G.nodes[v]['infected'] == 'I':
            if random.random() < beta:
                G.nodes[v]['infected'] = 'R'


def update(G, p, q):
    propagate_simple(G, p)
    recover(G, q)


degrees_to_test = np.arange(1, 1001, 25)
results = list()
for deg in degrees_to_test:
    # create graph
    p = deg / 1000
    G = nx.generators.random_graphs.fast_gnp_random_graph(1000, p)
    nx.set_node_attributes(G, 'S', 'infected')
    # set initial
    initial_index = random.randint(0, len(G) - 1)
    G.nodes[random.choice(list(G.nodes))]['infected'] = 'I'
    # parameters

    # prob of contagion in contact
    p = 0.2
    # prob of recover
    beta = 1 / 10

    t = 100  # iterations to simulate
    Infectius = t * [None]
    for i in range(t):
        dict_atributes = nx.get_node_attributes(G, 'infected')
        # Susceptibles[i] = sum(x == 'S' for x in dict_atributes.values())
        Infectius[i] = sum(x == 'I' for x in dict_atributes.values())
        # Recovered[i] = sum(x == 'R' for x in dict_atributes.values())
        update(G, p, beta)
    results.append(max(Infectius))

print(results)
plt.plot(degrees_to_test, results)
plt.xlabel('Average degree of the graph')
plt.ylabel('Max of simultaneous cases (persons)')
plt.title("Average degree of the nodes and height of the disease curve")
ymax = max(results)
xpos = results.index(ymax)
xmax = degrees_to_test[xpos]

text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
plt.annotate(text, xy=(xmax, ymax))

plt.savefig(("degree" + str(len(degrees_to_test))))
plt.show()

