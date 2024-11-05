from typing import NamedTuple, List, Dict, Tuple, Any
import numpy as np
from ..scheduler import Scheduler

class HeuristicObs(NamedTuple):
    job_ptr: np.ndarray
    frontier_stages: set
    schedulable_stages: dict
    exec_supplies: np.ndarray
    num_committable_execs: int
    source_job_idx: int

class SJFHeuristicScheduler(Scheduler):
    '''Shortest Job First (SJF) heuristic scheduler'''

    @classmethod
    def preprocess_obs(cls, obs: dict) -> HeuristicObs:
        '''Processes the input observation to create a HeuristicObs instance.'''
        frontier_mask = np.ones(obs['dag_batch'].nodes.shape[0], dtype=bool)
        dst_nodes = obs['dag_batch'].edge_links[:, 1]
        frontier_mask[dst_nodes] = False
        frontier_stages = set(frontier_mask.nonzero()[0])

        job_ptr = np.array(obs['dag_ptr'])
        stage_mask = obs['dag_batch'].nodes[:, 2].astype(bool)
        schedulable_stages = dict(
            zip(stage_mask.nonzero()[0], np.arange(stage_mask.sum()))
        )
        exec_supplies = np.array(obs['exec_supplies'])
        num_committable_execs = obs['num_committable_execs']
        source_job_idx = obs['source_job_idx']

        return HeuristicObs(
            job_ptr,
            frontier_stages,
            schedulable_stages,
            exec_supplies,
            num_committable_execs,
            source_job_idx
        )

    @classmethod
    def find_stage(cls, obs: HeuristicObs, job_idx: int) -> int:
        '''Searches for a schedulable stage in a given job, prioritizing 
        frontier stages.
        '''
        stage_idx_start = obs.job_ptr[job_idx]
        stage_idx_end = obs.job_ptr[job_idx + 1]

        selected_stage_idx = -1
        for node in range(stage_idx_start, stage_idx_end):
            try:
                schedulable_index = obs.schedulable_stages[node]
            except KeyError:
                continue

            if node in obs.frontier_stages:
                return schedulable_index

            if selected_stage_idx == -1:
                selected_stage_idx = schedulable_index

        return selected_stage_idx

    def get_action(self, obs: HeuristicObs) -> Tuple[int, int]:
        '''Get the next action to schedule based on the SJF algorithm.'''
        job_ptr = obs.job_ptr
        exec_supplies = obs.exec_supplies
        num_committable_execs = obs.num_committable_execs
        source_job_idx = obs.source_job_idx

        # Dictionary to hold job execution times (or costs)
        job_execution_times = {job_idx: 0 for job_idx in range(len(job_ptr) - 1)}
        
        # Populate job execution times (replace with your own logic to get execution time)
        for job_idx in range(len(job_ptr) - 1):
            job_execution_times[job_idx] = self.calculate_execution_time(job_idx, obs)

        # Find the job with the minimum execution time that can be scheduled
        selected_job_idx = min(
            (job_idx for job_idx in job_execution_times if job_execution_times[job_idx] > 0),
            key=job_execution_times.get,
            default=None
        )

        if selected_job_idx is not None:
            next_stage = self.find_stage(obs, selected_job_idx)
            if next_stage != -1:
                use_exec = min(
                    exec_supplies[next_stage],
                    num_committable_execs
                )
                return next_stage, use_exec

        return None, num_committable_execs

    def calculate_execution_time(self, job_idx: int, obs: HeuristicObs) -> int:
        '''Calculate the execution time for a given job (placeholder).'''
        # Implement your logic to calculate the execution time for the job
        # For example, this could be based on the number of stages, resources needed, etc.
        return np.random.randint(1, 10)  # Example: replace with actual logic
#存在问题？