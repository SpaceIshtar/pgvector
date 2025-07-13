import pickle
import numpy as np

CARDINALITY = 10000
TOPK = 10
ALPHA = 0.05
# METHODS = ["bitmap_v2", "postfilter", "pushdown_v2"]
# METHODS = ["postfilter", "bitmap_v2"]
METHODS = ["postfilter", "bitmap_v3_alpha"+str(ALPHA), "pushdown_v3_alpha"+str(ALPHA)]
gt_path = "/home/jiaruiluo/postgres/pgvector/benchres/sift1m_prefilter_card"+str(CARDINALITY)+"_topk"+str(TOPK)+".pkl"
methods_path = ["/home/jiaruiluo/postgres/pgvector/benchres/sift1m_m64_"+method+"_card"+str(CARDINALITY)+"_topk"+str(TOPK)+".pkl" for method in METHODS]

def load_result(pkl_file):
    f = open(pkl_file, 'rb')
    result = pickle.load(f)
    f.close()
    return result

def evaluate_recall(gt_res, target_res):
    length = len(target_res)
    hit = 0
    less_than_topk = 0
    for i in range(length):
        res = []
        if (len(target_res[i]) < TOPK):
            less_than_topk += 1
        for j in range(len(target_res[i])):
            res.append(target_res[i][j][0])
        for j in range(TOPK):
            if gt_res[i][j][0] in res:
                hit += 1
    print(less_than_topk)
    return hit/(length*TOPK)

### Calculate the latency of prefilter
prefilter_res = load_result(gt_path)
prefilter_latency = np.mean(prefilter_res[2])
print("Prefilter: ")
print("Prefilter latency: "+str(prefilter_latency))
print(" ")

### Calculate the latency and recall of methods
for i in range(len(METHODS)):
    method = METHODS[i]
    path = methods_path[i]
    result = load_result(path)
    latency = np.mean(result[2])
    recall = evaluate_recall(prefilter_res[1], result[1])
    print(method+": ")
    print(method+" latency: "+str(latency))
    print(method+" recall: "+str(recall))
    print("")