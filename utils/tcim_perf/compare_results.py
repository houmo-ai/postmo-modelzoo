#!/usr/bin/env python3

import os
import numpy as np

def cosine_distance(data1, data2):
    """余弦距离
    :param data1:
    :param data2:
    :return:
    """
    v1_d = data1.flatten().astype("float64")
    v2_d = data2.flatten().astype("float64")
    assert len(v1_d) == len(v2_d), "v1 dim must be == v2 dim"
    v1_d[v1_d == np.inf] = np.finfo(np.float16).max
    v2_d[v2_d == np.inf] = np.finfo(np.float16).max
    v1_d[v1_d == -np.inf] = np.finfo(np.float16).min
    v2_d[v2_d == -np.inf] = np.finfo(np.float16).min
    v1_norm = v1_d / np.linalg.norm(v1_d)
    v2_norm = v2_d / np.linalg.norm(v2_d)
    return np.dot(v1_norm, v2_norm)

thread_num=1
loop_num=10
input_name=()
output_name=('')
output_num=len(output_name)

print(f"thread_num={thread_num} loop_num={loop_num}")

from prettytable import PrettyTable

header = ["Id", "thread", "run", "output", "match", "similarity"]
table = PrettyTable(header)

idx = 0
for oid in range(output_num):
    for tid in range(0, thread_num):
        for lid in range(loop_num):
            file_name = f"thread_{tid}_run_{lid}_{output_name[oid]}_output.bin"
            if not os.path.exists(file_name):
                print(f"{file_name} not exist")
                break
            data = np.fromfile(f"thread_{tid}_run_{lid}_{output_name[oid]}_output.bin", dtype=np.float32)
            ref = np.fromfile(f"{output_name[oid]}_output.bin", dtype=np.float32)
            if len(data) == len(ref):
                cosine_dist = cosine_distance(data, ref)
                # print("data=", data)
                # print("ref=", ref)
                is_match = (data == ref).all()
                # print("[compare] output [{}] match={}, similarity={:.6f}"
                #       .format(output_name[oid], is_match, cosine_dist))
                result = [idx, tid, lid, output_name[oid], is_match, cosine_dist]
                table.add_row(result)
            else:
                print("[compare] output [{}] len not equal {} vs {}"
                      .format(output_name[oid], len(data), len(ref)))
            idx += 1
print(f"\n{table}")
