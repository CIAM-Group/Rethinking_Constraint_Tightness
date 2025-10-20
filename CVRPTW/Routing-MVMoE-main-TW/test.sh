#!/usr/bin/env bash

scales=(0.2 0.5 1.0 1.5 2.0 2.5 3.0) # 0.2 0.5 1.0 1.5 2.0 2.5 3.0 ## 0.2 0.4 0.5 0.6 0.8 1.0 1.2 1.4 1.5 1.6 1.8 2.0 2.2 2.4 2.5 2.6 2.8 3.0
for scale in "${scales[@]}"; do
  python test.py --test_set_path ./data/VRPTW/vrptw100_uniform_0.2_n10000_speed_1.0_scale_${scale}.pkl \
        --test_set_opt_sol_path ./data/VRPTW/hgs_vrptw100_uniform_0.2_n10000_speed_1.0_scale_${scale}.pkl
  wait
done
