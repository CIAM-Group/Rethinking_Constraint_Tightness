

This folder contains the code of generating CVRP instances with `customized capacity value` and `different demand distribution`.

### Dependency
```bash
hygese=0.0.0.8
numpy=1.24.2
```

you can adjust the time budget for hgs to solve one instance in the file `cvrp100_solved_by_hgs.py`, 
in  `ap = hgs.AlgorithmParameters(timeLimit = 10)` 


### How to run
An example to generate a dataset consisting of 10 CVRP100 instances with C=50 and `uniform` distributed demand, 
using 5 cpu cores, with the beginning of the seed=0:


Generate instances:
```bash
bash generate.sh 10 5 100 0 'data_vrp100_demand_uniform' 'uniform' 50 
# (instances_number, cpu_cores, problem_size, seed_beginning, output_path, demand_distribution, capacity)
```
Concat these instances to form the dataset:
```bash
python concat_dataset.py --output_path 'output' --capacity 50 --source_instance_path 'data_vrp100_demand_uniform' --demand_distribution 'uniform' 
```
---
An example to generate a dataset consisting of 10 CVRP100 instances with C=50 and `longtailed` distributed demand, 
using 5 cpu cores, with the beginning of the seed=10:


Generate instances:
```bash
bash generate.sh 10 5 100 10 'data_vrp100_demand_longtailed ' 'longtailed' 50
# (instances_number, cpu_cores, problem_size, seed_beginning, output_path, demand_distribution, capacity)
```
Concat these instances to form the dataset:
```bash
python concat_dataset.py --output_path 'output' --capacity 50 --source_instance_path 'data_vrp100_demand_longtailed' --demand_distribution 'longtailed' 
```
