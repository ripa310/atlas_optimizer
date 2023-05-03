#%%
import networkx as nx
import pulp

# create the networkx graph
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
G = G.to_directed()


# define the subset of nodes V'
V_prime = {1, 3}

# define the weight function c
c = {1: 1, 2: 2, 3: 3, 4: 4}

# create the PuLP problem
prob = pulp.LpProblem("ILP", pulp.LpMinimize)

# create the variables
x = pulp.LpVariable.dicts("x", G.nodes(), cat="Binary")
y = pulp.LpVariable.dicts("y", G.edges(), cat="Binary")



# add the objective function
prob += pulp.lpSum(c[v] * x[v] for v in G.nodes())

# add the constraints
for v in V_prime:
    prob += x[v] == 1

for v in V_prime:
    prob += pulp.lpSum(y[(u , v)] for u in G.neighbors(v)) >= 1

for u, v in G.edges():
    prob += x[v] >= y[(u, v)]

for v in G.nodes():
    prob += pulp.lpSum(y[(u, v)] for u in G.neighbors(v) ) >= 2 * x[v]


# solve the problem
prob.solve()

# print the solution
print("Status:", pulp.LpStatus[prob.status])
print("Objective value:", pulp.value(prob.objective))
for v in G.nodes():
    print(f"x[{v}] = {pulp.value(x[v])}")
for u, v in G.edges():
    print(f"y[{u}, {v}] = {pulp.value(y[(u, v)])}")