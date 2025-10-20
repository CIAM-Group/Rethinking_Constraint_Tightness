##########################################################################################
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")  # for utils
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
sys.path.insert(0, "../../..")  # for utils
import logging
import numpy as np

from CVRP.LEHD_main.utils.utils import create_logger, copy_all_src
from CVRP.LEHD_main.CVRP.VRPTester import VRPTester as Tester

##########################################################################################
# parameters

# testing problem size
problem_size = 100

# decode method: use RRC or not (greedy)
Use_RRC = False

# RRC budget
RRC_budget = 10

########### model ###############
model_load_path = 'result/20250118_221129_train'
model_load_epoch = 40

if not Use_RRC:
    RRC_budget = 0

mode = 'test'
test_paras = {
   # problem_size: [filename, episode, batch]
    100: [ 'vrp100_test_lkh.txt',10000,5000],
}


##########################################################################################
# parameters
b = os.path.abspath("..").replace('\\', '/')

env_params = {
    'mode': mode,
    'data_path': b + f"/data/{test_paras[problem_size][0]}",
    'sub_path': False,
    'RRC_budget': RRC_budget
}


model_params = {
    'mode': mode,
    'embedding_dim': 192,
    'sqrt_embedding_dim': 192**(1/2),
    'decoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 12,
    'ff_hidden_dim': 512,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_episodes': test_paras[problem_size][1],   # 65
    'test_batch_size': test_paras[problem_size][2],
}

logger_params = {
    'log_file': {
        'desc': f'test__vrp{problem_size}',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main_test(epoch,path,data_path=None,use_RRC=None,cuda_device_num=None):
    if DEBUG_MODE:
        _set_debug_mode()
    create_logger(**logger_params)
    _print_config()
    tester_params['model_load']={
        'path': path,
        'epoch': epoch,
    }
    if use_RRC is not None:
        env_params['RRC_budget']=0
    if cuda_device_num is not None:
        tester_params['cuda_device_num'] = cuda_device_num
    if data_path is not None:
        env_params['data_path'] = b + '/data/testset/demand_uniform/' + data_path
    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student,gap
def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()


    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    score_optimal, score_student, gap = tester.run()
    return score_optimal, score_student,gap

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":

    capacities = {
        10: 'vrp100_C10_test_hgs_time_0h.txt',
        50: 'vrp100_C50_test_hgs_time_0h.txt',
        100: 'vrp100_C100_test_hgs_time_0h.txt',
        200: 'vrp100_C200_test_hgs_time_0h.txt',
        300: 'vrp100_C300_test_hgs_time_0h.txt',
        400: 'vrp100_C400_test_hgs_time_0h.txt',
        500: 'vrp100_C500_test_hgs_time_0h.txt',
    }

    path = f'./{model_load_path}'

    Cs = [10, 50, 100, 200, 300, 400, 500]  # 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000

    for C in Cs:
        data_path = capacities[C]


        for i in [model_load_epoch]:
            score_optimal, score_student,gap = main_test(i,path,data_path = data_path)

