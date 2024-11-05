import numpy as np

from .heuristic import HeuristicScheduler

class FIFOScheduler(HeuristicScheduler):

    def __init__(self, num_executors):
        name = 'FIFO'
        super().__init__(name)
        self.num_executors = num_executors

    def schedule(self, obs):
        obs = self.preprocess_obs(obs)
        num_active_jobs = len(obs.exec_supplies)

        # No dynamic partitioning in FIFO; all executors are given to jobs in arrival order
        executor_cap = self.num_executors

        # Search through jobs in the order they arrived (FIFO principle)
        for j in range(num_active_jobs):
            # If the job has enough executors already, skip it
            if obs.exec_supplies[j] >= executor_cap:
                continue

            # Find the stage of the job to schedule
            selected_stage_idx = self.find_stage(obs, j)
            if selected_stage_idx == -1:
                continue

            # Assign as many executors as available, respecting the executor capacity
            num_exec = min(
                obs.num_committable_execs,
                executor_cap - obs.exec_supplies[j]
            )
            return {
                'stage_idx': selected_stage_idx,
                'num_exec': num_exec
            }

        # No stages available for scheduling
        return {
            'stage_idx': -1,
            'num_exec': obs.num_committable_execs
        }
