"""Cache full-scope v7 node embeddings (same MiniLM the weave used) for on-demand
semantic NN in the translation engine."""
import pickle, json, time
import numpy as np
from sentence_transformers import SentenceTransformer
t0=time.time()
g=pickle.load(open("graph-v7u.pkl","rb"))
nodes=sorted(g["edges"].keys())
words=[n.split(":",1)[1] for n in nodes]
m=SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
v=np.asarray(m.encode(words,batch_size=256,normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)
np.save("node-vecs.npy", v)
json.dump(nodes, open("node-ids.json","w"))
print(f"cached {len(nodes)} vecs {v.shape} -> node-vecs.npy ({time.time()-t0:.0f}s)")
