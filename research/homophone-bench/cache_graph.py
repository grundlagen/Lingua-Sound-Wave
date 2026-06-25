"""Build the unedited v7 graph once and pickle it so routing/analysis is instant.
Imports chain_game.build_graph VERBATIM (no engine edit); min_sound=0 = full graph.
"""
import pickle, time
from chain_game import build_graph
t0 = time.time()
edges, sem_neigh = build_graph(min_sound=0.0)
# defaultdicts -> plain dicts for a clean, dependency-free pickle
edges = {k: v for k, v in edges.items()}
sem_neigh = {k: list(v) for k, v in sem_neigh.items()}
with open("graph-v7u.pkl", "wb") as f:
    pickle.dump({"edges": edges, "sem_neigh": sem_neigh}, f, protocol=4)
print(f"cached {len(edges)} nodes -> graph-v7u.pkl ({time.time()-t0:.0f}s)")
