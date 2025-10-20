#!/usr/bin/env bash

echo "test start......"


#total_num=10 # the total number of instances to be generated
#iter_start_init=0 # set the begining of the seed, each instance corresponding to one different seed
#cpu_cores=5 # the cpu cores used to run the python script
#node_num=100

total_num=${1:-10}       # 第1个参数: 总实例数 (默认: 10)
cpu_cores=${2:-5}       # 第2个参数: CPU核心数 (默认: 5)
node_num=${3:-100}      # 第3个参数: 节点数 (默认: 100)
iter_start_init=${4:-0} # 第4个参数: 初始seed (默认: 0)
outputpath=${5:-"data_vrp100"}   # 第5个参数: 输出日志路径 (默认: "log")
demanddistribution=${6:-"uniform"}   # 第5个参数: 输出日志路径 (默认: "log")


batch=$((total_num / cpu_cores))


# -----------------------------------------------------------------
# 2. 处理 "capacities" 数组参数
# -----------------------------------------------------------------
# 检查是否提供了第 6 个或更多的参数 (因为 $5 现在是 output_path)
if [ "$#" -ge 7 ]; then
    # 如果是，使用第 6 个及之后的所有参数作为 capacities 数组
    capacities=("${@:7}")
else
    # 否则，使用默认值
    capacities=(50)
fi


mkdir -p log
# 对每个 capacity 进行循环
for capacity in "${capacities[@]}"; do
  # 对每个 capacity 执行的任务都需要一个内部循环
  for ((j=0; j<cpu_cores; j++)); do
    # 计算iter_start和iter_end的值
    echo "Starting processing for capacity: $capacity"
    iter_start=$((iter_start_init + j * batch))
    iter_end=$((iter_start_init + (j + 1) * batch))  # n_instance
    nohup python -u cvrp100_solved_by_hgs.py --capacity $capacity --problem_size $node_num --iters_start $iter_start --output_path $outputpath \
    --demand_distribution $demanddistribution --iters_end $iter_end > log/log_vrp100_C${capacity}_begin_${iter_start}_end_${iter_end}.txt 2>&1 &
  done

  wait
done
