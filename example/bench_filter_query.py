import numpy as np
from pgvector.psycopg import register_vector
import psycopg
import pickle
import time
import subprocess

CARDINALITY = 10000
TOPK = 10
ALPHA = 0.05
# METHOD = "seqpath"
# METHOD = "prefilter"
# METHOD = "postfilter"
# METHOD = "bitmap_v3_alpha"+str(ALPHA)
METHOD = "pushdown_v3_alpha"+str(ALPHA)

def convert_embedding_to_query_v1(vector, cardinality, k):
    str_vec="'["
    for i in range(len(vector)):
        str_vec += str(vector[i])
        if i < len(vector)-1:
            str_vec+=","
    str_vec+="]'"
    query = "SELECT id FROM sift1m WHERE id < "+str(cardinality)+" ORDER BY embedding <-> "+str_vec+" LIMIT "+str(k)

    return query

filename="/home/jiaruiluo/vector_search/dataset/sift/sift_query.bin"
save_path = "/home/jiaruiluo/postgres/pgvector/benchres/sift1m_m64_"+METHOD+"_card"+str(CARDINALITY)+"_topk"+str(TOPK)+".pkl"
file = open(filename,'rb')
embeddings = np.fromfile(file, dtype=np.float32)
shape = embeddings[:2].view(np.uint32)
embeddings = embeddings[2:].reshape(shape)
rows, dimensions = shape
print(shape)

queries = []
for i in range(rows):
    queries.append(convert_embedding_to_query_v1(embeddings[i], CARDINALITY, TOPK))
    
conn = psycopg.connect(dbname='jiaruiluo',host='localhost',user='jiaruiluo', autocommit=True)
register_vector(conn)
cur = conn.cursor()

cur.execute("set hnsw.iterative_scan='relaxed_order'")
cur.execute("set hnsw.ef_search = 40")
cur.execute("SET ivfflat.probes = 10")

results = []
latency = []

for i in range(rows):
    start_time = time.time()
    query = queries[i]
    cur.execute(query)
    result = cur.fetchall()
    end_time = time.time()
    results.append(result)
    search_time = end_time - start_time
    latency.append(search_time)

print("Exec end")
data_to_save = (len(results), results, latency)

with open(save_path, 'wb') as f:
    pickle.dump(data_to_save, f)
    
print("Save file")