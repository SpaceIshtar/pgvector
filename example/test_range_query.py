import numpy as np
from pgvector.psycopg import register_vector
import psycopg
import pickle
import time
import subprocess

def convert_embedding_to_query_v1(vector, search_range):
    str_vec="'["
    for i in range(len(vector)):
        str_vec += str(vector[i])
        if i < len(vector)-1:
            str_vec+=","
    str_vec+="]'"
    query = "SELECT id FROM sift1m WHERE ANN_DWithin(embedding, " + str_vec + ", " + str(search_range) + ")" 

    return query

def convert_embedding_to_query_v2(vector, search_range):
    str_vec="'["
    for i in range(len(vector)):
        str_vec += str(vector[i])
        if i < len(vector)-1:
            str_vec+=","
    str_vec+="]'"
    query = "SELECT id FROM sift1m WHERE embedding <-> "

    query += str_vec + " < "+str(search_range)
    return query

def read_range_search_groundtruth(filename):
    """
    Reads the ground truth data saved in the specified binary format.

    Args:
        filename (str): The path to the binary file.

    Returns:
        tuple: A tuple containing:
            - nq (int): The number of queries.
            - threshold (float): The distance threshold used for the search.
            - limits (np.ndarray): Array of start and end indices for each query's results.
            - neighbor_ids (np.ndarray): Flattened array of neighbor IDs.
            - neighbor_distances (np.ndarray): Flattened array of neighbor distances.
    """
    with open(filename, "rb") as f:
        nq = np.fromfile(f, dtype=np.uint32, count=1)[0]
        threshold = np.fromfile(f, dtype=np.float32, count=1)[0]
        limits = np.fromfile(f, dtype=np.uint32, count=nq + 1)
        neighbor_ids = np.fromfile(f, dtype=np.uint32, count = limits[-1])
        # Calculate the expected number of distances based on the total number of IDs
        num_distances = len(neighbor_ids)
        neighbor_distances = np.fromfile(f, dtype=np.float32, count=num_distances)
    return nq, threshold, limits, neighbor_ids, neighbor_distances

def read_float_bin(filename):
    a = np.fromfile(filename, dtype=np.float32)
    [nd,dim] = a[:2].view(np.uint32)
    return nd, dim, a[2:].reshape(nd, dim)

def test_queries(queries):
    results = []
    conn = psycopg.connect(dbname='jiaruiluo',host='localhost',user='jiaruiluo', autocommit=True)
    register_vector(conn)
    cur = conn.cursor()
    # cur.execute("SELECT pg_backend_pid();")
    # pg_pid = cur.fetchone()[0]
    # print(f"PostgreSQL backend PID: {pg_pid}")

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
    
    cur.execute("set hnsw.iterative_scan='relaxed_order'")
    start_time = time.time()
    for i in range(len(queries)):
        cur.execute(queries[i])
        result = cur.fetchall()
        results.append(result)
        if i % 1000 == 0:
            print(i)
    end_time = time.time()
    cur.close()
    conn.close()
    return end_time - start_time, results

def evaluate_recall(results, limits, neighbor_ids):
    total_recall = 0.0
    list_results = []
    nq = len(results)
    if (nq != len(limits) - 1):
        print("nq: %d, len(limits): %d"%(nq, len(limits)))
        exit(-1)
    
    for i in range(nq):
        hit = 0
        total = 0
        start_idx = limits[i]
        end_idx = limits[i+1]
        recall = 0
        for j in range(start_idx, end_idx):
            gt_id = neighbor_ids[j]
            total += 1
            for t in results[i]:
                if t[0] == gt_id + 1:
                    hit += 1
                    break
        if total == 0:
            recall = 1
        else:
            recall = hit/total

        total_recall += recall
    return total_recall/nq

if __name__ == '__main__':
    gt_path = "/home/jiaruiluo/vector_search/dataset/sift/sift_range_300_gt.np.bin"  # Replace with your actual path
    query_path = "/home/jiaruiluo/vector_search/dataset/sift/sift_query.bin"
    nq, threshold, limits, neighbor_ids, neighbor_distances = read_range_search_groundtruth(gt_path)

    nq2, dim, queries = read_float_bin(query_path)
    if (nq != nq2):
        print("nq = %d, nq2 = %d"%(nq, nq2))
        exit(-1)
    
    index_scan_queries = []
    seq_scan_queries = []
    for i in range(nq):
        index_scan_queries.append(convert_embedding_to_query_v1(queries[i], threshold))
        seq_scan_queries.append(convert_embedding_to_query_v2(queries[i], threshold))
        
    print("Generated Queires")
        
    seq_scan_time, seq_scan_results = test_queries(seq_scan_queries)
    print("performed seq scan")
    index_scan_time, index_scan_results = test_queries(index_scan_queries)
    filename = "./results/index_scan_result_range300.pkl"
    with open(filename, 'wb') as file:
        # 使用 pickle.dump() 函数将 list 保存到文件中
        pickle.dump(index_scan_results, file)
    print("performed index scan")
    seq_scan_recall = evaluate_recall(seq_scan_results, limits, neighbor_ids)
    seq_scan_qps = nq/seq_scan_time
    index_scan_recall = evaluate_recall(index_scan_results, limits, neighbor_ids)
    index_scan_qps = nq/index_scan_time
    print("Seq Scan QPS:"+str(seq_scan_qps)+", recall: "+str(seq_scan_recall))
    print("Index Scan QPS: "+str(index_scan_qps)+", recall: "+str(index_scan_recall))