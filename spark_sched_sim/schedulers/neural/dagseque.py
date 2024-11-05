import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric.utils as pyg_utils
import torch_sparse

from spark_sched_sim.wrappers import DAGNNObsWrapper
from spark_sched_sim import graph_utils

from .neural import *

class DecimaScheduler(nn.Module):
    '''Original Decima architecture without ResNet-18 as node encoder.'''

    def __init__(self, num_executors, embed_dim, gnn_mlp_kwargs, policy_mlp_kwargs, state_dict_path=None, opt_cls=None, opt_kwargs=None, max_grad_norm=None, num_node_features=5, num_dag_features=3, **kwargs):
        name = 'Decima'
        if state_dict_path:
            name += f':{state_dict_path}'

        actor = ActorNetwork(num_executors, num_node_features, num_dag_features, embed_dim, gnn_mlp_kwargs, policy_mlp_kwargs)
        
        obs_wrapper_cls = DAGNNObsWrapper

        super().__init__(name, actor, obs_wrapper_cls, num_executors, state_dict_path, opt_cls, opt_kwargs, max_grad_norm)

class ActorNetwork(nn.Module):
    def __init__(self, num_executors, num_node_features, num_dag_features, embed_dim, gnn_mlp_kwargs, policy_mlp_kwargs):
        super().__init__()
        self.encoder = EncoderNetwork(num_node_features, embed_dim, gnn_mlp_kwargs)

        emb_dims = {
            'node': embed_dim,
            'dag': embed_dim,
            'glob': embed_dim
        }

        self.stage_policy_network = StagePolicyNetwork(num_node_features, emb_dims, policy_mlp_kwargs)
        self.exec_policy_network = ExecPolicyNetwork(num_executors, num_dag_features, emb_dims, policy_mlp_kwargs)
        
        self._reset_biases()

    def _reset_biases(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.zero_()

class EncoderNetwork(nn.Module):
    def __init__(self, num_node_features, embed_dim, mlp_kwargs):
        super().__init__()

        # 使用简单的线性层替代 ResNet-18 进行特征提取
        self.node_encoder = NodeEncoder(num_node_features, embed_dim)
        self.dag_encoder = DagEncoder(num_node_features, embed_dim)
        self.global_encoder = GlobalEncoder(embed_dim)

    def forward(self, dag_batch):
        '''
            Returns:
                a dict of representations at three different levels:
                node, dag, and global.
        '''
        h_node = self.node_encoder(dag_batch)
        h_dag = self.dag_encoder(h_node, dag_batch)

        try:
            # batch of obsns
            obs_ptr = dag_batch['obs_ptr']
            h_glob = self.global_encoder(h_dag, obs_ptr)
        except:
            # single obs
            h_glob = self.global_encoder(h_dag)

        h_dict = {
            'node': h_node,
            'dag': h_dag,
            'glob': h_glob
        }

        return h_dict

class NodeEncoder(nn.Module):
    def __init__(self, num_node_features, embed_dim):
        super().__init__()
        self.fc = nn.Linear(num_node_features, embed_dim)

    def forward(self, dag_batch):
        x = dag_batch.x
        h_init = self.fc(x)

        # 在没有消息传递时，直接返回特征
        return h_init

class DagEncoder(nn.Module):
    def __init__(self, num_node_features, embed_dim):
        super().__init__()
        input_dim = num_node_features + embed_dim
        # 使用简单的线性层代替复杂的 MLP
        self.linear = nn.Linear(input_dim, embed_dim)

    def forward(self, h_node, dag_batch):
        # 包含原始输入
        h_node = torch.cat([dag_batch.x, h_node], dim=1)
        h_dag = segment_csr(self.linear(h_node), dag_batch.ptr)
        return h_dag

class GlobalEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        # 使用简单的线性层代替复杂的 MLP
        self.linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, h_dag, obs_ptr=None):
        h_dag = self.linear(h_dag)

        if obs_ptr is not None:
            # 批量观察
            h_glob = segment_csr(h_dag, obs_ptr)
        else:
            # 单个观察
            h_glob = h_dag.sum(0).unsqueeze(0)

        return h_glob
