import numpy as np
from pgvector.psycopg import register_vector
import psycopg
import pickle
import time
import subprocess

def convert_embedding_to_query_v1(vector):
    str_vec="'["
    for i in range(len(vector)):
        str_vec += str(vector[i])
        if i < len(vector)-1:
            str_vec+=","
    str_vec+="]'"
    query = "SELECT id FROM sift1m WHERE id < 50000 ORDER BY embedding <-> "+str_vec+" LIMIT 10"

    return query

def convert_embedding_to_query_OR(vector):
    str_vec="'["
    for i in range(len(vector)):
        str_vec += str(vector[i])
        if i < len(vector)-1:
            str_vec+=","
    str_vec+="]'"
    query = "SELECT id FROM sift1m WHERE id < 100000 OR id > 900000 ORDER BY embedding <-> "+str_vec+" LIMIT 10"

    return query

def convert_embedding_to_query_plain(vector):
    str_vec="'["
    for i in range(len(vector)):
        str_vec += str(vector[i])
        if i < len(vector)-1:
            str_vec+=","
    str_vec+="]'"
    query = "SELECT id FROM sift1m ORDER BY embedding <-> "+str_vec+" LIMIT 10"

    return query

filename="/home/jiaruiluo/vector_search/dataset/sift/sift_query.bin"
file = open(filename,'rb')
embeddings = np.fromfile(file, dtype=np.float32)
shape = embeddings[:2].view(np.uint32)
embeddings = embeddings[2:].reshape(shape)
rows, dimensions = shape
print(shape)

queries = []
for i in range(rows):
    queries.append(convert_embedding_to_query_v1(embeddings[i]))
    
conn = psycopg.connect(dbname='jiaruiluo',host='localhost',user='jiaruiluo', autocommit=True)
register_vector(conn)
cur = conn.cursor()
cur.execute("SELECT pg_backend_pid();")
pg_pid = cur.fetchone()[0]
print(f"PostgreSQL backend PID: {pg_pid}")

gdb_command = f"sudo gdbserver :1234 --attach {pg_pid}"
gdb_process = subprocess.Popen(
    gdb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True
)
# 等待 gdbserver 输出“Listening on port 1234”
print("Starting gdbserver and attaching to process...")
while True:
    output = gdb_process.stderr.readline().decode("utf-8")
    print(output)
    if "Listening on port" in output:
        print("gdbserver is ready.")
        break
    elif gdb_process.poll() is not None:
        print("gdbserver failed to start.")
        break

input("Press Enter to continue...")

query = queries[0]
print(query)

cur.execute("set hnsw.iterative_scan='relaxed_order'")
cur.execute("SET ivfflat.probes = 10")

cur.execute(query)
result = cur.fetchall()

print(result)
print(len(result))
# print(len(set(result)))