from tqdm import tqdm
import numpy as np
import os
import argparse

# import matplotlib.pyplot as plt

def OneRowSolution_to_TwoRow(ori_solution):
    solution = ori_solution

    node = []
    flag = []
    for i in range(1, len(solution)):
        if solution[i] != 0:
            node.append(solution[i])
        if solution[i] != 0 and solution[i - 1] == 0:
            flag.append(1)
        if solution[i] != 0 and solution[i - 1] != 0:
            flag.append(0)
    node_flag = node + flag
    return node_flag


def vrp_form_solution(solution_temp):
    # solution_temp： 经过转换的solution，shape (200)
    vrp_solution = []
    V = int(len(solution_temp) / 2)
    for i in range(V):
        if solution_temp[V + i] == 0:
            vrp_solution.append(solution_temp[i])
        if solution_temp[V + i] == 1:
            vrp_solution.append(0)
            vrp_solution.append(solution_temp[i])
    # 这个时候的solution，第一个点是 0, shape (100 + depot_num)
    return vrp_solution


def load_dataset(current_file_path):
    raw_lines = []
    # print(current_file_path)

    for line in open(current_file_path, "r").readlines():
        raw_lines.append(line[:-2])
        # print(raw_lines)
        # assert False

    depot_all = []
    loc_all = []
    demand_all = []
    capacity_all = []
    cost_all = []
    solution_all = []
    duration_all = []
    for i in range(len(raw_lines)):
        instance = raw_lines[i]

        line = instance.split(", ")
        if line[0] == '[\'depot\'':
            depot_index = int(line.index('[\'depot\''))
            customer_index = int(line.index('\'customer\''))
            demand_index = int(line.index('\'demand\''))
            capacity_index = int(line.index('\'capacity\''))
            solution_index = int(line.index('\'solution\''))

            cost_index = int(line.index('\'cost\''))
            time_index = int(line.index('\'time\''))
        else:
            line = instance.split(",")
            # print(line[0])
            depot_index = int(line.index('depot'))
            customer_index = int(line.index('customer'))
            demand_index = int(line.index('demand'))
            capacity_index = int(line.index('capacity'))
            solution_index = int(line.index('solution'))

            cost_index = int(line.index('cost'))
            time_index = int(line.index('time'))
        # print()
        # print(depot_index,customer_index,demand_index,capacity_index,cost_index)

        depot = [float(line[depot_index + 1]), float(line[depot_index + 2])]
        customer = [[float(line[idx]), float(line[idx + 1])] for idx in range(customer_index + 1, demand_index, 2)]

        # print(loc)
        # 包括 depot 的 location，在第一个

        capacity = int(float(line[capacity_index + 1]))
        demand = [int(line[idx]) for idx in range(demand_index + 1, capacity_index)]
        # 包括depot的demand，其为 0，在第一个
        # print(capacity)
        # print(demand)
        cost = float(line[cost_index + 1])
        # print(cost)

        time_cost = float(line[time_index + 1])

        solution = [int(line[idx]) for idx in range(solution_index + 1, cost_index)]

        converted_solution = OneRowSolution_to_TwoRow(solution)

        ########## Draw ##############
        # loc = depot+customer
        #
        #
        # drawPic_VRP(np.array([depot]+customer), np.array(solution), name='xx', optimal_tour_=None)

        # assert False

        depot_all.append(depot)
        loc_all.append([kk for item in customer for kk in item])
        demand_all.append(demand)
        capacity_all.append(capacity)
        cost_all.append(cost)
        solution_all.append(converted_solution)
        duration_all.append(time_cost)


    return depot_all, loc_all, demand_all, capacity_all, cost_all, solution_all, duration_all


def save_dataset(one_row_data_all, savepath):
    np.savetxt(savepath, one_row_data_all, delimiter=',', fmt='%s')

    return


def get_data_file_path(path):

    filelist = []

    for dir in os.listdir(path):
        filename = path + dir
        filelist.append(filename)
    filelist = np.array(filelist)
    return filelist


def load_raw_data(data_path):
    # print('load raw dataset begin!')

    raw_lines = []

    for line in open(data_path, "r").readlines():
        raw_lines.append(line[1:-2])
        # print(raw_lines)
        # assert False
    raw_lines = np.array(raw_lines)
    # print(f'load raw dataset done!', )  # 读1024个 TSP100 instance 用时 4s
    return np.array(raw_lines)


def catdata(depot_all, loc_all, demand_all, capacity_all, cost_all, solution_all, ):
    one_row_data_all = []
    for i in range(len(depot_all)):
        curr_depot = depot_all[i]

        curr_loc = loc_all[i]

        curr_demand = demand_all[i]
        curr_cap = capacity_all[i]
        curr_cost = cost_all[i]
        curr_node_flag = solution_all[i]

        one_row_data = ['depot'] + curr_depot + ['customer'] + curr_loc + ['capacity'] + [curr_cap] + ['demand'] + \
                       curr_demand + ['cost'] + [curr_cost] + ['node_flag'] + curr_node_flag
        one_row_data_all.append(one_row_data)

    one_row_data_all = np.array(one_row_data_all)
    # print(one_row_data_all.shape)
    #
    # save_dataset(one_row_data_all, path + f'/vrp2000_train_hgs_n{len(one_row_data_all)}.txt')
    return one_row_data_all


# def drawPic_VRP(arr_, tour_, name='xx',optimal_tour_=None):
#     # arr = arr_[0].clone().cpu().numpy()
#     # tour =  tour_[0].clone().cpu().numpy()
#     arr = arr_
#     tour =  tour_
#
#     if optimal_tour_ is not None:
#         optimal_tour = optimal_tour_.clone().cpu().numpy()
#     arr_max = np.max(arr)
#     arr_min = np.min(arr)
#     arr = (arr - arr_min) / (arr_max - arr_min)
#
#     print(arr.shape)
#     print('tour.shape',tour.shape)
#     fig, ax = plt.subplots(figsize=(20, 20))
#     # print(arr)
#     plt.scatter(arr[:, 0], arr[:, 1], color='black', linewidth=1)
#     plt.xlim(0,1)
#     plt.ylim(0, 1)
#
#     # 连接起点和终点
#
#     start = [arr[tour[0], 0], arr[tour[-1], 0]]
#     end = [arr[tour[0], 1], arr[tour[-1], 1]]
#     plt.plot(start, end, color='red', linewidth=2, )# linestyle="dashed"
#
#     # 连接各个点
#     for i in range(len(tour) - 1):
#         # worst greedy tour
#         tour = np.array(tour, dtype=int)
#         # print(tour)
#         start = [arr[tour[i], 0], arr[tour[i + 1], 0]]
#         end = [arr[tour[i], 1], arr[tour[i + 1], 1]]
#         plt.plot(start,end,color='red',linewidth=2)# ,linestyle ="dashed"
#
#         if optimal_tour_ is not None:
#             tour_optimal = np.array(optimal_tour, dtype=int)
#             # print('tour_optimal',tour_optimal)
#             start_optimal = [arr[tour_optimal[i], 0], arr[tour_optimal[i + 1], 0]]
#             end_optimal = [arr[tour_optimal[i], 1], arr[tour_optimal[i + 1], 1]]
#             plt.plot(start_optimal, end_optimal, color='green', linewidth=1)
#     plt.show()

def add_common_args(parser):
    # -- Network --
    parser.add_argument("--output_path", type=str, default=None, help="Embeddings size")
    parser.add_argument("--capacity", type=int, default=100, help="Embeddings size")
    parser.add_argument("--source_instance_path", type=str, default=10, help="Embeddings size")
    parser.add_argument("--demand_distribution", type=str, default='uniform', choices=['uniform', 'longtailed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    b = os.path.abspath('.').replace('\\', '/')
    add_common_args(parser)
    args = parser.parse_args()
    times = []
    cost_ = []

    source_instance_path = args.source_instance_path
    data_save_path = b + f'/{args.output_path}/'
    demand_distribution = args.demand_distribution
    batch = 1


    Capacities = [args.capacity]

    for Capacity in Capacities:

        begin = batch * Capacity
        end = batch * (Capacity + 1)
        one_row_data_all_in_all = []

        for current_capacity in range(begin, end):

            data_path_root = b + f'/{source_instance_path}/C{current_capacity}/'

            isExisit = os.path.exists(data_path_root)

            if isExisit:

                data_file_path = get_data_file_path(data_path_root)
                print(data_file_path)
                data_num = len(data_file_path)
                # print(data_path_root)
                ii=0
                for data_file in data_file_path:  # len(data_file_path)
                    ii+=1
                    if (ii + 1) % 100 == 0:
                        print(ii, '/', data_num, data_path_root)


                    print('current_file_path', data_file)
                    isExists = os.path.exists(data_file)
                    if isExists:
                        depot_all, loc_all, demand_all, capacity_all, cost_all, solution_all, duration_all \
                            = load_dataset(data_file)

                        one_row_data_all = catdata(depot_all, loc_all, demand_all, capacity_all, cost_all, solution_all)
                        if one_row_data_all_in_all == []:
                            one_row_data_all_in_all = one_row_data_all
                        else:
                            one_row_data_all_in_all = np.concatenate((one_row_data_all_in_all, one_row_data_all), axis=0)
                        times.append(duration_all)
                        cost_.append(cost_all)

        isExists = os.path.exists(data_save_path)
        if not isExists:
            os.makedirs(data_save_path)

        # assert False
        if len(one_row_data_all_in_all) == 0:
            pass
        else:
            save_dataset(one_row_data_all_in_all,
                         data_save_path + f'/vrp100_hgs_n{len(one_row_data_all_in_all)}_C{Capacity}_demand_{demand_distribution}.txt')
