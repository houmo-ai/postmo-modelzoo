#!/usr/bin/env python3

import os
import numpy as np
from prettytable import PrettyTable
from hmassist.utils.dist_metrics import cosine_distance


thread_num = 1
loop_num = 10
input_name = []
output_name = ["495"]
output_num = len(output_name)
dtype = np.uint8

print(f"thread_num={thread_num} loop_num={loop_num}")

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
            data = np.fromfile(f"thread_{tid}_run_{lid}_{output_name[oid]}_output.bin", dtype=dtype)
            ref = np.fromfile(f"{output_name[oid]}_output.bin", dtype=dtype)
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
