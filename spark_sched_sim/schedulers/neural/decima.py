import torch
import torch.nn as nn
from torch_scatter import segment_csr
import torch_geometric.utils as pyg_utils
import torch_sparse
import torch_geometric.nn as gnn
from .neural import *
from spark_sched_sim.wrappers import DAGNNObsWrapper
from spark_sched_sim import graph_utils



class DecimaScheduler(NeuralScheduler):
    '''Original Decima architecture, which uses asynchronous message passing
    as in DAGNN.
    Paper: https://dl.acm.org/doi/abs/10.1145/3341302.3342080
    '''

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

        self.node_encoder = NodeEncoder(
            num_node_features, embed_dim, mlp_kwargs)
        
        self.dag_encoder = DagEncoder(
            num_node_features, embed_dim, mlp_kwargs)
        
        self.global_encoder = GlobalEncoder(
            embed_dim, mlp_kwargs)


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
        mlp_kwargs,
        reverse_flow=True
    ):
        super().__init__()
        self.reverse_flow = reverse_flow
        self.j, self.i = (1, 0) if reverse_flow else (0, 1)

        self.mlp_prep = make_mlp(
            num_node_features, output_dim=embed_dim, **mlp_kwargs)
        
        # Initialize TransformerConv
        self.transformer_conv =gnn.TransformerConv(
            in_channels=embed_dim, 
            out_channels=embed_dim, 
            heads=4,  # Number of attention heads
            concat=True  # Whether to concatenate the outputs
        )

        self.mlp_update = make_mlp(
            embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, dag_batch):
        edge_masks = dag_batch['edge_masks']
        
        if edge_masks.shape[0] == 0:
            return self._forward_no_mp(dag_batch.x)
        
        # Pre-process the node features into initial representations
        h_init = self.mlp_prep(dag_batch.x)

        # Will store all the nodes' representations
        h = torch.zeros_like(h_init)

        num_nodes = h.shape[0]

        src_node_mask = ~pyg_utils.index_to_mask(
            dag_batch.edge_index[self.i], num_nodes)
        
        h[src_node_mask] = self.mlp_update(h_init[src_node_mask])

        # Use TransformerConv for message passing
        h = self.transformer_conv(h_init, dag_batch.edge_index)

        # Update the representations with the transformed output
        h = h + self.mlp_update(h)

        return h

    def _forward_no_mp(self, x):
        return self.mlp_prep(x)

class DagEncoder(nn.Module):
    def __init__(
        self, 
        num_node_features, 
        embed_dim, 
        mlp_kwargs
    ):
        super().__init__()
        input_dim = num_node_features + embed_dim
        self.mlp = make_mlp(
            input_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, h_node, dag_batch):
        # include original input
        h_node = torch.cat([dag_batch.x, h_node], dim=1)
        h_dag = segment_csr(self.mlp(h_node), dag_batch.ptr)
        return h_dag
    


class GlobalEncoder(nn.Module):
    def __init__(self, embed_dim, mlp_kwargs):
        super().__init__()
        self.mlp = make_mlp(embed_dim, output_dim=embed_dim, **mlp_kwargs)

    def forward(self, h_dag, obs_ptr=None):
        h_dag = self.mlp(h_dag)

        if obs_ptr is not None:
            # batch of observations
            h_glob = segment_csr(h_dag, obs_ptr)
        else:
            # single observation
            h_glob = h_dag.sum(0).unsqueeze(0)

        return h_glob
    
    
    
   
