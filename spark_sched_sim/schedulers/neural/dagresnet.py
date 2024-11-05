import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric.utils as pyg_utils
import torch_sparse

from torchvision import models
from spark_sched_sim.wrappers import DAGNNObsWrapper
from spark_sched_sim import graph_utils

from .neural import *


class DecimaSchedulerResnet(NeuralScheduler):
    '''Original Decima architecture with ResNet-18 as node encoder.'''

    def __init__(
        self,
        num_executors,
        embed_dim,
        gnn_mlp_kwargs,
        policy_mlp_kwargs,
        state_dict_path=None,
        opt_cls=None,
        opt_kwargs=None,
        max_grad_norm=None,
        num_node_features=5,
        num_dag_features=3,
        **kwargs
    ):
        name = 'Decima'
        if state_dict_path:
            name += f':{state_dict_path}'

        actor = ActorNetwork(
            num_executors, 
            num_node_features, 
            num_dag_features, 
            embed_dim,
            gnn_mlp_kwargs,
            policy_mlp_kwargs)
        
        obs_wrapper_cls = DAGNNObsWrapper

        super().__init__(
            name,
            actor,
            obs_wrapper_cls,
            num_executors,
            state_dict_path,
            opt_cls,
            opt_kwargs,
            max_grad_norm
        )



class ActorNetwork(nn.Module):
    def __init__(
        self, 
        num_executors, 
        num_node_features, 
        num_dag_features, 
        embed_dim,
        gnn_mlp_kwargs,
        policy_mlp_kwargs
    ):
        super().__init__()
        self.encoder = EncoderNetwork(
            num_node_features, embed_dim, gnn_mlp_kwargs)

        emb_dims = {
            'node': embed_dim,
            'dag': embed_dim,
            'glob': embed_dim
        }

        self.stage_policy_network = StagePolicyNetwork(
            num_node_features, emb_dims, policy_mlp_kwargs)

        self.exec_policy_network = ExecPolicyNetwork(
            num_executors, num_dag_features, emb_dims, policy_mlp_kwargs)
        
        self._reset_biases()
        

    def _reset_biases(self):
        for name, param in self.named_parameters():
            if 'bias' in name:
                param.data.zero_()



class EncoderNetwork(nn.Module):
    def __init__(self, num_node_features, embed_dim, mlp_kwargs):
        super().__init__()

        # 使用 ResNet-18 替代 MLP 进行特征提取
        self.node_encoder = NodeEncoder(
            num_node_features, embed_dim)
        
        self.dag_encoder = DagEncoder(
            num_node_features, embed_dim)
        
        self.global_encoder = GlobalEncoder(
            embed_dim)


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
    def __init__(
        self, 
        num_node_features, 
        embed_dim,
        reverse_flow=True
    ):
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        # 使用 ResNet-18 替代 MLP 作为特征提取网络
        self.feature_extractor = models.resnet18(pretrained=True)
        
        # 修改第一个卷积层的输入通道数为 num_node_features
        self.feature_extractor.conv1 = nn.Conv2d(
            in_channels=num_node_features, 
            out_channels=64,  
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # 替换最后的全连接层
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, embed_dim)


    def forward(self, dag_batch):
        edge_masks = dag_batch['edge_masks']
        
        if edge_masks.shape[0] == 0:
            # 如果没有消息传递，直接通过 ResNet-18 进行特征提取
            return self._forward_no_mp(dag_batch.x)

        # 对输入的节点特征通过 ResNet-18 进行预处理
        # 首先调整输入的维度，以便适配 ResNet-18
        h_init = self.feature_extractor(self._prepare_input(dag_batch.x))

        # 初始化 h
        h = torch.zeros_like(h_init)

        num_nodes = h.shape[0]

        src_node_mask = ~pyg_utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes)
        
        h[src_node_mask] = h_init[src_node_mask]

        edge_masks_it = iter(reversed(edge_masks)) \
            if self.reverse_flow else iter(edge_masks)

        # 反向或正向的消息传递
        for edge_mask in edge_masks_it:
            edge_index_masked = dag_batch.edge_index[:, edge_mask]
            adj = graph_utils.make_adj(edge_index_masked, num_nodes)

            # 发送消息的节点
            src_mask = pyg_utils.index_to_mask(
                edge_index_masked[self.j], num_nodes)

            # 接收消息的节点
            dst_mask = pyg_utils.index_to_mask(
                edge_index_masked[self.i], num_nodes)

            msg = torch.zeros_like(h)
            msg[src_mask] = h[src_mask]
            agg = torch_sparse.matmul(
                adj if self.reverse_flow else adj.t(), msg)
            h[dst_mask] = h_init[dst_mask] + agg[dst_mask]

        return h
    

    def _prepare_input(self, x):
        '''准备输入数据，使其符合 ResNet-18 的要求'''
        # 假设输入为 [batch_size, num_node_features]
        batch_size, num_features = x.shape
        # 将输入调整为 [batch_size, num_node_features, 1, 1] 的形状
        x = x.view(batch_size, num_features, 1, 1)  # ResNet需要4D输入
        return x
    

    def _forward_no_mp(self, x):
        '''不进行消息传递时，直接通过ResNet-18处理'''
        # 调整输入形状并通过 ResNet 进行处理
        return self.feature_extractor(self._prepare_input(x))

    


class DagEncoder(nn.Module):
    def __init__(
        self, 
        num_node_features, 
        embed_dim
    ):
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
