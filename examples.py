'''Examples of how to run job scheduling simulations with different schedulers
'''
import os.path as osp
from pprint import pprint

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import gymnasium as gym
import pathlib
import json
from cfg_loader import load
from spark_sched_sim.schedulers import *
from spark_sched_sim.wrappers import *
from spark_sched_sim import metrics
action_str_list=[]

ENV_KWARGS = {
    'num_executors': 10,
    'job_arrival_cap': 50,
    'job_arrival_rate': 4.e-5,
    'moving_delay': 2000.,
    'warmup_delay': 1000.,
    'dataset': 'tpch',
    'render_mode': 'human'
}
# ENV_KWARGS={
#   'num_executors': 50,
#   'job_arrival_cap':200,
#   'job_arrival_rate': 4.e-5,
#   'moving_delay': 2000.,
#   'warmup_delay': 1000.,
#   'dataset': 'tpch',
#   'mean_time_limit': 2.e+7
# }
def main():
    # save final rendering to artifacts dir
    pathlib.Path('artifacts').mkdir(parents=True, exist_ok=True) 

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(
        '--sched',
        choices=['fair', 'decima','fifo','sjf','random'],
        dest='sched',
        help='which scheduler to run',
        required=True,
    )

    args = parser.parse_args()

    sched_map = {
        'fair': fair_example,
        # 'sjf':sjf_example,
        # 'fifo':fifo_example,
        'random':random_example,
        'decima': decima_example
    }

    sched_map[args.sched]()
def random_example():
    # Fair scheduler
    scheduler = RandomScheduler(ENV_KWARGS['num_executors'])
                                   
    
    print(f'Example: random Scheduler')
    print('Env settings:')
    pprint(ENV_KWARGS)

    print('Running episode...')
    avg_job_duration = run_episode(ENV_KWARGS, scheduler)

    print(f'Done! Average job duration: {avg_job_duration:.1f}s', flush=True)
    print()


def fair_example():
    # Fair scheduler
    scheduler = RoundRobinScheduler(ENV_KWARGS['num_executors'],
                                    dynamic_partition=True)
    
    print(f'Example: Fair Scheduler')
    print('Env settings:')
    pprint(ENV_KWARGS)

    print('Running episode...')
    avg_job_duration = run_episode(ENV_KWARGS, scheduler)

    print(f'Done! Average job duration: {avg_job_duration:.1f}s', flush=True)
    print()



def decima_example():
    cfg = load(filename=osp.join('config', 'decima_tpch.yaml'))

    # agent_cfg = cfg['agent'] \
    #     | {'num_executors': ENV_KWARGS['num_executors'],
    #        'state_dict_path': osp.join('dagcheckpoint','dagnnmodel79.pt')} #modity  path of model
    #'models', 'decima', 


    agent_cfg = cfg['agent'] \
        | {'num_executors': ENV_KWARGS['num_executors'],
           'state_dict_path': osp.join('dagformercheckpoint','dagformermodel499.pt')} #modity  path of model
    scheduler = make_scheduler(agent_cfg)

    # agent_cfg = cfg['agent'] \
    #     | {'num_executors': ENV_KWARGS['num_executors'],
    #        'state_dict_path': osp.join('checkpoint','model299.pt')} #modity  path of model
    scheduler = make_scheduler(agent_cfg)

    print(f'Example: Decima')
    print('Env settings:')
    pprint(ENV_KWARGS)

    print('Running episode...')
    avg_job_duration = run_episode(ENV_KWARGS, scheduler)

    print(f'Done! Average job duration: {avg_job_duration:.1f}s', flush=True)



def run_episode(env_kwargs, scheduler, seed=1234):
    env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', **env_kwargs)
    if isinstance(scheduler, NeuralScheduler):
        env = NeuralActWrapper(env)
        env = scheduler.obs_wrapper_cls(env)
    # with open('data.txt', 'a') as file:
    obs, _ = env.reset(seed=seed, options=None)
    terminated = truncated = False
    
    while not (terminated or truncated):
        # print("wait")
        try:
            if isinstance(scheduler, NeuralScheduler):
                action, *_ = scheduler(obs)
            else:
                action = scheduler(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            action_str = reward
            action_str_list.append(action_str)
            with open('reward.txt', 'w') as file:
                for element in action_str_list:
                    file.write(f"{element}\n")
        # except RuntimeError :
            
        #         # 忽略捕获到的 RuntimeError 错误 再任务处理结束之后可能会出现错误
        #          break
        except ValueError:
            print("error")
            # 忽略捕获到的 ValueError 错误
            break
    
    
    avg_job_duration = metrics.avg_job_duration(env) * 1e-3

    # cleanup rendering
    env.close()

    return avg_job_duration


if __name__ == '__main__':
    main()