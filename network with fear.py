import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd

"""
G is of the MultiGraph ClasS
NODES ARE IN STATES S, ID (INFECTED WITH DISEASE), IF (infected with fear) IFD (infected with both) RF (Recovered from
fear->susceptible, RD recovered disease and fear, R recovered disease susceptible to fear)
"""

"""
G = nx.MultiGraph()
G.add_edge(1, 2, weight=4.7 )
G.add_edges_from([(3,4),(4,5)], color='red')
G.add_edges_from([(1,2,{'color':'blue'}), (2,3,{'weight':8})])
G[1][2][0]['weight'] = 4.7
print(G.edges(nbunch=1, data='weight'))
print(G.edges[1,2,0]['weight'])

type(G.edges(nbunch=1, data='weight'))

L = [x for x in G.edges(nbunch=1, data='weight')]
"""


def check_connection(G,u, v, busco='fisica' ):
    """
    G multigraph
    u,v nodos
    :type tipo: string 'fisica' o 'social'
    """
    for dic in G.get_edge_data(u,v).values():
        if dic['tipo'] == busco:
            return True

    return False


def propagate_disease(G, p):
    to_infect = set([])
    # Find infected nodes
    for v in G.nodes():
        if G.nodes[v]['infected disease'] == 'I':
            # infect all
            for w in nx.neighbors(G, v):
                if G.nodes[w]['infected disease'] == 'S' and check_connection(G, v, w):
                    if G.nodes[w]['infected fear'] == 'I' and random.random() < (p / 10): #TODO PARAMETRO P/10
                        to_infect.add(w)
                        break
                    elif random.random() < p:
                        to_infect.add(w)
                        break
    # Infect marked nodes
    for v in to_infect:
        G.nodes[v]['infected disease'] = 'I'


def propagate_fear(G, p):
    to_infect = set([])
    # Find infected nodes
    for v in G.nodes():
        if (G.nodes[v]['infected fear'] == 'I') or (G.nodes[v]['infected disease'] == 'I'):
            # me asustan los infectados y los asustados
            for w in nx.neighbors(G, v):
                if G.nodes[w]['infected fear'] == 'S' and check_connection(G, v, w, busco='social'):
                    if random.random() < p:
                        to_infect.add(w)
                        break
    # Infect marked nodes
    for v in to_infect:
        G.nodes[v]['infected fear'] = 'I'


def recover_disease(G, beta):
    for v in G.nodes():
        if G.nodes[v]['infected disease'] == 'I':
            if random.random() < beta:
                G.nodes[v]['infected disease'] = 'R'


def recover_fear(G, beta):
    for v in G.nodes():
        if G.nodes[v]['infected fear'] == 'I':
            if random.random() < beta:
                G.nodes[v]['infected fear'] = 'R'


def update(G, infect_disease, infect_fear, p_recover_disease, p_recover_fear):
    propagate_disease(G, infect_disease)
    propagate_fear(G, infect_fear)
    recover_disease(G, p_recover_disease)
    recover_fear(G, p_recover_fear)


H = nx.generators.karate_club_graph()
#nx.set_node_attributes(H, 'S', 'infected disease')
#nx.set_node_attributes(H, 'S', 'infected fear')
#nx.set_edge_attributes(H, 'fisica', name='tipo')

I = nx.generators.complete_graph(34)
#nx.set_node_attributes(I, 'S', 'infected disease')
#nx.set_node_attributes(I, 'S', 'infected fear')
#nx.set_edge_attributes(I, 'social', name='tipo')

G = nx.MultiGraph()
G.add_nodes_from(H)
G.add_edges_from(H.edges,tipo='fisica')

G.add_nodes_from(I)
G.add_edges_from(I.edges, tipo='social')

nx.set_node_attributes(G, 'S', 'infected disease')
nx.set_node_attributes(G, 'S', 'infected fear')

#probs
infect_disease = 0.1
infect_fear = 0.2
p_recover_disease = 0.05
p_recover_fear = 0.05

#iteration
number_of_simulations = 30

results_fear = list()
results_disease = list()

for j in range(number_of_simulations):
    #we set all nodes as susceptible to both fear and disease
    nx.set_node_attributes(G, 'S', 'infected disease')
    nx.set_node_attributes(G, 'S', 'infected fear')
    # initial condition
    # we start selecting one random node for being infected with disease and another random node for being infected with fear

    G.nodes[random.choice(list(G.nodes))]['infected disease'] = 'I'
    G.nodes[random.choice(list(G.nodes))]['infected fear'] = 'I'
    # simulate
    t = 100  # iterations to simulate
    Infectious_D = t * [None]
    Infectious_F = t * [None]
    for i in range(t):
        dict_attributes_D = nx.get_node_attributes(G, 'infected disease')
        dict_attributes_F = nx.get_node_attributes(G, 'infected fear')
        # Susceptibles[i] = sum(x == 'S' for x in dict_atributes.values())
        Infectious_D[i] = sum(x == 'I' for x in dict_attributes_D.values())
        Infectious_F[i] = sum(x == 'I' for x in dict_attributes_F.values())

        # Recovered[i] = sum(x == 'R' for x in dict_atributes.values())
        update(G, infect_disease, infect_fear, p_recover_disease, p_recover_fear)
    results_fear.append(Infectious_F)
    results_disease.append(Infectious_D)

df_fear = pd.DataFrame(results_fear)
df_disease = pd.DataFrame(results_disease)

mean_of_fear = df_fear.mean()
mean_of_disease = df_disease.mean()



plt.plot(range(t), mean_of_disease, label='Infected')
plt.plot(range(t), mean_of_fear, label='Scared')
plt.xlabel('days')
plt.ylabel("persons")
plt.legend()
plt.title("Fear spreads faster\n(mean of 30 simulations) ")
plt.savefig("Fear spreads faster")
plt.show()





