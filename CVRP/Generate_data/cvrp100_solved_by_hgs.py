import numpy as np
import hygese as hgs
import random
import time
import os
import sys
import argparse

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

seed_everything(1234)

# CVRP200:  50 intance spend average time: 37.49226  s
# CVRP100:  50 intance spend average time: 11.22338  s
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

def add_common_args(parser):
    # -- Network --
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument("--problem_size", type=int, default=100)
    parser.add_argument("--iters_start", type=int, default=10)
    parser.add_argument("--iters_end", type=int, default=10)
    parser.add_argument("--demand_distribution", type=str, default='uniform', choices=['uniform', 'longtailed'])
    parser.add_argument("--output_path", type=str, default='data_vrp')


def generate_nazari_vrp_data(dataset_size, vrp_size,CPACITY=None,demand_distribution='uniform'):
    if CPACITY==None:

        CAPACITIES = { 10: 20., 20: 30., 50: 40., 100: 50., 200: 80., 500: 100.,
            1000: 250., 2000: 250., 3000: 250., 4000: 250.,
            5000: 500.,  6000: 500., 7000: 500., 8000: 500., 9000: 500.,
            10000: 1000., 20000: 1000., 30000: 1000., 40000: 1000.,
            50000: 2000., 60000: 2000., 70000: 2000., 80000: 2000., 90000: 2000.,
            100000: 2000.,
        }
        C = CAPACITIES[vrp_size]

    else:
        C = CPACITY


    if demand_distribution=='uniform':
        return list(zip(
            np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
            np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
            np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
            np.full(dataset_size, C).tolist()  # Capacity, same for whole dataset
        ))
    elif demand_distribution=='longtailed':
        numbers_to_generate = np.arange(1, 10)
        a = 1.5
        probabilities = 1 / (numbers_to_generate ** a)
        probabilities /= np.sum(probabilities)

        return list(zip(
            np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
            np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations

            np.random.choice(numbers_to_generate, size=(dataset_size, vrp_size), p=probabilities).tolist(),
            # np.clip(np.round(np.random.normal(5, 1, size=(dataset_size, vrp_size))), 1, 9).astype(int).tolist(),
            #

            np.full(dataset_size, C).tolist()  # Capacity, same for whole dataset
        ))
    else:
        assert False, 'no such implementation'

def generate_vrp_data(dataset_size, vrp_size, seed):

    np.random.seed(seed)
    return generate_nazari_vrp_data(dataset_size, vrp_size)


def save_dataset(one_row_data_all, savepath):


    np.savetxt(savepath, one_row_data_all,delimiter=',',fmt='%s')

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_op')
    add_common_args(parser)
    args = parser.parse_args()


    capacity_init = args.capacity
    iters_start =  args.iters_start
    iters_end = args.iters_end
    vrp_size = args.problem_size
    demand_distribution = args.demand_distribution
    output_path = args.output_path
    dataset_size = 1

    for iters in range(iters_start, iters_end): # iter多少次就有多少个instance
        print('iter',iters)
        np.random.seed(iters)
        # # generate the training set
        # CPACITY = np.random.randint(low_capacity,high_capacity)

        CPACITY = capacity_init

        dataset = generate_nazari_vrp_data(dataset_size, vrp_size, CPACITY,demand_distribution)
        # all_times = []
        all_data = []
        path = os.path.abspath(".").replace('\\', '/')

        file_name = f'vrp{vrp_size}_test_hgs_n{dataset_size}__C{CPACITY}_dmand{demand_distribution}_No{iters}.txt'
        root_path = f'/{output_path}/C{CPACITY}/'


        path_solution = path + root_path + file_name
        print(path_solution)
        start_index = 0
        solutions_data = []
        isExists = os.path.exists(path_solution)
        if isExists:
            print('isExists:', iters)
            continue
        print('iters:',iters)

        for kkk in range(len(dataset)): # len(dataset) = 1

            print('No ',kkk,'/',dataset_size)

            data_instance = dataset[kkk]
            # print(data_instance)
            depot, Customer, demand, capacity = data_instance

            depot = np.array(depot)

            Customer = np.array(Customer)

            demand = np.array([0] + demand)

            xy = np.concatenate((depot.reshape(1,-1),Customer),axis=0) * 10000



            ap = hgs.AlgorithmParameters(timeLimit = 10)  # seconds

            hgs_solver = hgs.Solver(parameters=ap, verbose=False)

            data = dict()
            data['x_coordinates'] = xy[:,0]
            data['y_coordinates'] = xy[:,1]
            data['demands'] = demand
            data['vehicle_capacity'] = capacity
            data['depot'] = 0

            result = hgs_solver.solve_cvrp(data)


            solution_routes = []

            for ff in range(len(result.routes)):
                solution_routes = solution_routes + [0] + result.routes[ff]

            all_data.append(['depot',*depot.tolist(),'customer', *Customer.ravel().tolist(), 'demand',*demand,
                             'capacity',capacity,'solution',*solution_routes,'cost',result.cost / 10000,'time',result.time])
            # print('time',result.time)

        path = os.path.abspath(".").replace('\\', '/')
        all_data = np.array(all_data, dtype=object)

        isExists = os.path.exists(path + root_path)
        if not isExists:
            os.makedirs(path + root_path)
        save_dataset(all_data,path + root_path + file_name)



