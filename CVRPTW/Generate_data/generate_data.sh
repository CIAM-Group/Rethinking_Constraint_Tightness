#!/usr/bin/env bash




# differnt TW tightness degrees
scales=(0.2 0.5 1.0 1.5 2.0 2.5 3.0)

# number of instances
num=10000

# number of cpu cores used
cpu_num=50

speeds=(1.0)
for speed in "${speeds[@]}"; do
for scale in "${scales[@]}"; do
  # 计算iter_start和iter_end的值
  echo -e " \n Starting processing for scale: ${scale} -----------------------------------------------------------------"

  python generate_data.py --problem=VRPTW --problem_size=100 --num_samples $num --speed $speed --scale $scale
  wait  
  python HGS_baseline.py --problem=VRPTW --datasets=./data/VRPTW/vrptw100_uniform_0.2_n${num}_speed_${speed}_scale_${scale}.pkl -n=$num --cpus=$cpu_num -max_iteration=2000

  wait
  echo -e "End processing for scale: ${scale} --------------------------------------------------------------------- \n "
done
done



# python HGS_baseline.py --problem=VRPTW --datasets=./data/VRPTW/vrptw100_uniform_0.2_n10000_speed_1.0_scale_10.0.pkl -n=10000 --cpus=1 -max_iteration=2000