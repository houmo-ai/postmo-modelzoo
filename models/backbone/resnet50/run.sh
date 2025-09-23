#!/bin/bash

# 执行 100 次你的命令
for i in {1..100}
do
    echo "执行第 $i 次"
    # hmatc perf -c config.yml -wn 10 -sn 100 -tn 4
    hmatc perf -m xh2_conv.hmm -wn 10 -sn 100 -tn 2
done
