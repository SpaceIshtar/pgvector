import numpy as np
from pgvector.psycopg import register_vector
import psycopg
import time
import subprocess

# generate random data
# filename="/data/local/embedding_dataset/laion-5b/laion1B-nolang/laion1B-nolang/laion10M/laion_base.10M.bin"
# filename = "/home/jiaruiluo/vector_search/dataset/sift/sift_base.bin"
filename = "/home/jiaruiluo/vector_search/dataset/gist/gist_base.bin"
file = open(filename,'rb')
embeddings = np.fromfile(file, dtype=np.float32)
shape = embeddings[:2].view(np.uint32)
embeddings = embeddings[2:].reshape(shape)
rows, dimensions = shape
print(shape, flush=True)
# print(embeddings[0])

# enable extension
conn = psycopg.connect(dbname='jiaruiluo',host='localhost',user='jiaruiluo', autocommit=True)
register_vector(conn)
cur = conn.cursor()
cur.execute("SELECT pg_backend_pid();")
pg_pid = cur.fetchone()[0]
print(f"PostgreSQL backend PID: {pg_pid}")

# pg_pid = 536129
# gdb_command = f"sudo gdbserver :1234 --attach {pg_pid}"
# gdb_process = subprocess.Popen(
#     gdb_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True
# )
# # 等待 gdbserver 输出“Listening on port 1234”
# print("Starting gdbserver and attaching to process...")
# while True:
#     output = gdb_process.stderr.readline().decode("utf-8")
#     print(output)
#     if "Listening on port" in output:
#         print("gdbserver is ready.")
#         break
#     elif gdb_process.poll() is not None:
#         print("gdbserver failed to start.")
#         break

# input("Press Enter to continue...")

conn.execute('CREATE EXTENSION IF NOT EXISTS vector')

# create table
conn.execute('DROP TABLE IF EXISTS gist1m')
conn.execute(f'CREATE TABLE gist1m (id bigserial, embedding vector({dimensions}))')

# create any indexes *after* loading initial data (skipping for this example)
start = time.time()

# load data
print(f'Loading {len(embeddings)} rows', flush=True)
# cur.execute("SELECT pg_stat_reset_shared('io')")

with cur.copy('COPY gist1m (embedding) FROM STDIN WITH (FORMAT BINARY)') as copy:
    # use set_types for binary copy
    # https://www.psycopg.org/psycopg3/docs/basic/copy.html#binary-copy
    copy.set_types(['vector'])
    cur_time = time.time()
    for i, embedding in enumerate(embeddings):
        # show progress
        if i % 100000 == 0:
            # flush data
            new_time = time.time()
            if i > 0:
                print("Inserted batch, time cost: "+str(new_time-cur_time), flush=True)
                cur_time = new_time

        # if i == 100000:
        #     break

        copy.write_row([embedding])
        while conn.pgconn.flush() == 1:
                pass
            
create_index = True
if create_index:
    print('Creating index')
    conn.execute("SET maintenance_work_mem = '3GB'")
    conn.execute('SET max_parallel_maintenance_workers = 0')
    # conn.execute('CREATE INDEX ON gist1m USING ivfflat (embedding vector_l2_ops) WITH (lists = 1000);')
    conn.execute('CREATE INDEX ON gist1m using hnsw (embedding vector_l2_ops) WITH (m = 64, ef_construction = 200);')
            
end = time.time()
print(end-start)
print('\nSuccess!')