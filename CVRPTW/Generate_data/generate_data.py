import os, random, math, time
import argparse
import pprint as pp
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate datasets")
    parser.add_argument('--problem', type=str, default="ALL", choices=["ALL", "CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                                                       "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW",
                                                                       "OVRPBL", "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"])
    parser.add_argument('--problem_size', type=int, default=50)
    parser.add_argument('--pomo_size', type=int, default=50, help="the number of start node, should <= problem size")
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--dir', type=str, default="./data")
    parser.add_argument('--no_cuda', action='store_false')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--window', type=float, default=0.2)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--scale_tw', type=float, default=1.0)

    args = parser.parse_args()
    pp.pprint(vars(args))
    env_params = {"problem_size": args.problem_size, "pomo_size": args.pomo_size}
    seed_everything(args.seed)

    # set log & gpu
    if not args.no_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        args.device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    print(">> SEED: {}, USE_CUDA: {}, CUDA_DEVICE_NUM: {}".format(args.seed, not args.no_cuda, args.gpu_id))

    envs = get_env(args.problem)
    for env in envs:
        env = env(**env_params)
        dataset_path = os.path.join(args.dir, env.problem, "{}{}_uniform_{}_n{}_speed_{}_scale_{}.pkl".format(env.problem.lower(),
                                                                                                 args.problem_size, args.window, args.num_samples, args.speed, args.scale_tw ))
        env.generate_dataset(args.num_samples, args.problem_size, dataset_path, args.window, args.speed, args.scale_tw)
        # sanity check
        env.load_dataset(dataset_path, num_samples=args.num_samples, disable_print=False)
