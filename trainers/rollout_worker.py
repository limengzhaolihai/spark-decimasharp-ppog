import sys
from abc import ABC, abstractmethod
import os.path as osp
import random
import ast
import gymnasium as gym
from gymnasium.core import ObsType, ActType
import torch
import os
from spark_sched_sim.wrappers import *
from spark_sched_sim.schedulers import *
from .utils import Profiler, HiddenPrints
from spark_sched_sim.metrics import *



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
episode=0
rewards_folder = "rewards_folder"  

class RolloutBuffer:
    def __init__(self, async_rollouts=False):
        self.obsns: list[ObsType] = []
        self.wall_times: list[float] = []
        self.actions: list[ActType] = []
        self.lgprobs: list[float] = []
        self.rewards: list[float] = []
        self.resets = set() if async_rollouts else None

    def add(self, obs, wall_time, action, lgprob, reward):
        self.obsns += [obs]
        self.wall_times += [wall_time]
        self.actions += [action]
        self.rewards += [reward]
        self.lgprobs += [lgprob]

    def add_reset(self, step):
        assert self.resets is not None, "resets are for async rollouts only."
        self.resets.add(step)

    def __len__(self):
        return len(self.obsns)


class RolloutWorker(ABC):
    print("begin abc")
    def __init__(self):
        self.reset_count = 0
        print("begin abc2")
    
    def __call__(
        self,
        rank,
        conn,
        agent_cls,
        env_kwargs,
        agent_kwargs,
        stdout_dir,
        base_seed,
        seed_step,
        lock,
    ):
        print("bbb")
        self.rank = rank
        self.conn = conn
        self.base_seed = base_seed
        self.seed_step = seed_step
        self.reset_count = 0

        # log each of the processes to separate files
        # sys.stdout = open(osp.join(stdout_dir, f"{rank}.out"), "a")
        print("bbb2")
        self.agent = make_scheduler(agent_kwargs)
        self.agent.actor.eval()
        print("bb3")
        # might need to download dataset, and only one process should do this.
        # this can be achieved using a lock, such that the first process to
        # acquire it downloads the dataset, and any subsequent processes notices
        # that the dataset is already present once it acquires the lock.
        with lock:
            env = gym.make("spark_sched_sim:SparkSchedSimEnv-v0", **env_kwargs)

        env = StochasticTimeLimit(env, env_kwargs["mean_time_limit"])#在这块有个打印
        env = NeuralActWrapper(env)
        env = self.agent.obs_wrapper_cls(env)
        self.env = env

        # IMPORTANT! Each worker needs to produce unique rollouts, which are
        # determined by the rng seed
        torch.manual_seed(rank)
        random.seed(rank)
        print("begin run2")
        # torch multiprocessing is very slow without this
        torch.set_num_threads(1)

        self.run()

    def run(self):
        print("begin run")
        while data := self.conn.recv():
            # load updated model parameters
            self.agent.actor.load_state_dict(data["actor_sd"])
            try:
                
                print("run collect")
                rollout_buffer = self.collect_rollout()

                self.conn.send(
                    {"rollout_buffer": rollout_buffer, "stats": self.collect_stats()}
                )

            except AssertionError as msg:
                print(msg, "\naborting rollout.", flush=True)
                self.conn.send(None)

    @abstractmethod
    def collect_rollout(self) -> RolloutBuffer:
        pass

    @property
    def seed(self):
        return self.base_seed + self.seed_step * self.reset_count

    def collect_stats(self):
        return {
            "avg_job_duration": self.env.unwrapped.avg_job_duration,
            "avg_num_jobs": avg_num_jobs(self.env),
            "num_completed_jobs": self.env.unwrapped.num_completed_jobs,
            "num_job_arrivals": self.env.unwrapped.num_completed_jobs
            + self.env.unwrapped.num_active_jobs,
        }



def read_file(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    return content

def write_to_file(filename, data):
    with open(filename, 'w') as file:
        file.writelines(data)


class RolloutWorkerSync(RolloutWorker):
    """model updates are synchronized with environment resets"""
    print("begin sync")
    def collect_rollout(self):
        print("begin collect sync")
        rollout_buffer = RolloutBuffer()

        obs, _ = self.env.reset(seed=self.seed)
        # print(obs)
        self.reset_count += 1
        wall_time = 0
        terminated = truncated = False
        ENV_KWARGS={
        'num_executors': 50,
        'job_arrival_cap':200,
        'job_arrival_rate': 4.e-5,
        'moving_delay': 2000.,
        'warmup_delay': 1000.,
        'dataset': 'tpch',
        'mean_time_limit': 2.e+7
        }
        print("load")
        cfg = load(filename=osp.join("config", "decima_tpch.yaml"))

    #     agent_cfg = cfg["agent"] | {
    #     "num_executors": ENV_KWARGS["num_executors"],
    #     "state_dict_path": osp.join("models", "decima", "model.pt"),
    # }
        # agent_cfg = cfg['agent'] | {
        #     'num_executors': ENV_KWARGS['num_executors'],
        #     "state_dict_path": osp.join("dagcheckpoint", "dagnnmodel519.pt"),
        # }



        # agent_cfg = cfg['agent'] | {
        #     'num_executors': ENV_KWARGS['num_executors'],
        #     "state_dict_path": osp.join("resnetDagnncheckpoint", "resnetdagmodel.pt"),
        # }



        agent_cfg = cfg['agent'] | {
            'num_executors': ENV_KWARGS['num_executors'],
            "state_dict_path": osp.join("dagformercheckpoint", "dagformer.pt"),
        }
       

        
        scheduler1 = make_scheduler(agent_cfg)
        env = gym.make('spark_sched_sim:SparkSchedSimEnv-v0', **ENV_KWARGS)
        if isinstance(scheduler1, NeuralScheduler):
            env = NeuralActWrapper(env)
            env = scheduler1.obs_wrapper_cls(env)
            i=1
        global episode
        while not (terminated or truncated):
            # obs, _ = env.reset(seed=1234, options=None)
           
            # action1, lgprob1 = scheduler1(obs)#使用模型选择的动作
            # print("111")
            # print(action1)
            # print("yes")
            # loaded_list = []
            # loaded_list2=[]
            # loaded_list1=[]
            # with open('output.txt', 'r') as file:
            #     for line in file:
            #         loaded_list.append(line.strip())  # 去除换行符
            # # with open('obs.txt', 'r') as file:
            # #     for line in file:
            # #         loaded_list1.append(line.strip())  # 去除换行符
            # with open('reward.txt', 'r') as file:
            #     for line in file:
            #         loaded_list2.append(line.strip())  # 去除换行符
            # # 打印加载的列表
            # first_element = loaded_list[0]
            # # first_element1 = loaded_list1[0]
            # first_element2 = loaded_list2[0]
            # # print(len(loaded_list))
            # if(i<=50):
            rewards_folder="dagformerreward" # train decima use  rewards_folder  
            if not osp.exists(rewards_folder):  
                    os.makedirs(rewards_folder)  
            filename = osp.join(rewards_folder, f"rewards_episode_{episode+1}.txt")
            # filename = osp.join(rewards_folder, f"1.txt")
            # filename = f"rewards_episode_{episode+1}.txt"  # 设置文件名  
           
            action, lgprob = self.agent(obs)#lgprob选择动作的概率使用测试的调度器选择的动作       
            new_obs, reward, terminated, truncated, info = self.env.step(action)  
          
            with open(filename, 'a') as f:  # 打开文件以写入模式 
                f.write(f"{reward}\n")  # 将奖励写入文件，每个奖励占一行

            next_wall_time = info["wall_time"]
            rollout_buffer.add(obs, wall_time, list(action.values()), lgprob, reward)     
            # else:
                # print("diedai")
            # print("222")
            # print(action)


            # if type(obs) == dict:
            #     print("obs is a dictionary.") #知道是字典了
            # elif type(obs) == list:
            #     print("obs is a list.")
            # elif type(obs) == str:
            #     print("obs is a string.")
            # else:
            #     print("obs has a different type.")
            # first_element = eval(first_element)
            # action=first_element
            # first_element1 = ast.literal_eval(first_element1)
            # obs=first_element1
            # first_element2 = eval(first_element2)
            # reward=first_element2
            # print(reward)
            # print("111..........")

            #print("ytes")
            # reward1 = self.env.step1(action1)
            # reward2 = self.env.step1(action)
            # print(reward1)
            # print(reward2)

            # if((reward1-reward2)<10):
            #     new_obs, reward, terminated, truncated, info = self.env.step(action1)   
            #     next_wall_time = info["wall_time"]
            #     rollout_buffer.add(obs, wall_time, list(action1.values()), lgprob1, reward)     
            # i+=1
            # print("self")
            # else:
            # new_obs, reward, terminated, truncated, info = self.env.step(action1)
            # next_wall_time = info["wall_time"]
            # rollout_buffer.add(obs, wall_time, list(action1.values()), lgprob, reward)     
            # print("other")


            # if(reward1>reward):
            #     print(reward1)
            #     print(reward)
            # rollout_buffer.add(obs, wall_time, list(action.values()), lgprob, reward)     
            # print("add ok")
            obs = new_obs
            wall_time = next_wall_time
        
       
        print("0k")
        episode=episode+1
        rollout_buffer.wall_times += [wall_time]
        
        return rollout_buffer


class RolloutWorkerAsync(RolloutWorker):
    """model updates occur at regular intervals, regardless of when the
    environment resets
    """
    print("begin async")
    # RolloutWorker(ABC)
    print("end")

    def __init__(self, rollout_duration):
        super().__init__()
        self.rollout_duration = rollout_duration
        self.next_obs = None
        self.next_wall_time = 0.0
    
    def collect_rollout(self):
        print("begin collect async")
        rollout_buffer = RolloutBuffer(async_rollouts=True)

        if self.reset_count == 0:
            self.next_obs, _ = self.env.reset(seed=self.seed)
            self.reset_count += 1

        elapsed_time = 0
        step = 0
        rollout_duration=1
        while elapsed_time < rollout_duration:
            # print("async")
            obs, wall_time = self.next_obs, self.next_wall_time

            action, lgprob = self.agent(obs)#建立调度器
            
            self.next_obs, reward, terminated, truncated, info = self.env.step(action)

            self.next_wall_time = info["wall_time"]
            
            filename = 'data.txt'

            # 加载文件内容
            content = read_file(filename)

            # 解析数据
            num_trajectories = len(content) // 3
            # reward_list = []
            # action_list = []
            # obs_list = []

            for i in range(num_trajectories):
                reward_line = content[i * 3].strip().split(': ')[1]
                action_line = content[i * 3 + 1].strip().split(': ')[1]
                obs_line = content[i * 3 + 2].strip().split(': ')[1]

                reward_values = list(map(float, reward_line.split()))
                action_values = list(map(float, action_line.split()))
                obs_values = list(map(float, obs_line.split()))

                # reward_list.append(reward_values)
                # action_list.append(action_values)
                # obs_list.append(obs_values)

            # 删除前三行
            print("delete recivie")
            content = content[num_trajectories * 3:]

            # 将剩余数据写回文件
            write_to_file(filename, content)

            # 打印提取的数据
            print("has recivie")



            rollout_buffer.add(obs, elapsed_time, list(action.values()), lgprob, reward)
            print("add ok")
            # add the duration of the this step to the total
            elapsed_time += self.next_wall_time - wall_time

            if terminated or truncated:
                self.next_obs, _ = self.env.reset(seed=self.seed)
                self.reset_count += 1
                self.next_wall_time = 0
                rollout_buffer.add_reset(step)

            step += 1
            elapsed_time +=1

            rollout_buffer.wall_times += [elapsed_time]

            return rollout_buffer
