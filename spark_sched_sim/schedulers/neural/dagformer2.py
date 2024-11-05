from itertools import chain

import torch
import torch.nn as nn
from torch_scatter import segment_max_csr
import torch_geometric.nn as gnn

from .neural import *
from spark_sched_sim.wrappers import TransformerObsWrapper


class DAGformer2Scheduler(NeuralScheduler):
    '''Graph transformer for DAGs, which uses reachability-based attention
    (DAGRA) and node-depth-based positional encoding (DAGPE)
    Paper: https://arxiv.org/abs/2210.13148
    '''
    def __init__(
        self,
        num_executors,
        embed_dim,
        num_encoder_layers,
        num_attn_heads,
        policy_mlp_kwargs,
        state_dict_path=None,
        opt_cls=None,
        opt_kwargs=None,
        max_grad_norm=None,
        num_node_features=5,
        num_dag_features=3,
        **kwargs
    ):
        name = 'DAGformer'
        if state_dict_path:
            name += f':{state_dict_path}'

        actor = ActorNetwork(
            num_executors, num_node_features, num_dag_features, 
            embed_dim, num_attn_heads, num_encoder_layers, policy_mlp_kwargs)
        
        # replaces edges with transitive closure and adds node depth attribute
        obs_wrapper_cls = TransformerObsWrapper

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
        num_attn_heads,
        num_encoder_layers,
        policy_mlp_kwargs
    ):
        super().__init__()
        self.encoder = EncoderNetwork(
            num_node_features, embed_dim, num_attn_heads, num_encoder_layers)

        emb_dims = {
            'node': embed_dim,
            'dag': embed_dim,
            'glob': embed_dim
        }

        self.stage_policy_network = StagePolicyNetwork(
            num_node_features, emb_dims, policy_mlp_kwargs)

        self.exec_policy_network = ExecPolicyNetwork(
            num_executors, num_dag_features, emb_dims, policy_mlp_kwargs)


class EncoderNetwork(nn.Module):
    def __init__(self, num_node_features, embed_dim, num_heads, num_layers):
        super().__init__()

        # graph transformer with 'DAGPE' (dag positional encodings)
        self._init_node_encoder(
            num_node_features, embed_dim, num_heads, num_layers)

        # mean pooling over nodes for each dag
        self.dag_encoder = DagEncoder(embed_dim, embed_dim)

        # max pooling over dags for each observation
        self.global_encoder = gnn.Sequential('h_dag, obs_ptr', [
            (segment_max_csr, 'h_dag, obs_ptr -> h_glob'), 
            lambda pair: pair[0],
            nn.Linear(embed_dim, embed_dim)])


    def _init_node_encoder(self, num_node_features, embed_dim, num_heads, num_layers):
        head_dim = embed_dim // num_heads

        prep_layer = [
            # project input features
            (nn.Linear(num_node_features, embed_dim), 'x -> h0'),

            # 'DAGPE': positional encoding for dags using node depth
            (gnn.PositionalEncoding(embed_dim), 'depth -> dagpe')
        ]
            
        transformer_layers = [
            [
                # add PE
                (lambda x, y: x+y, 'h0, dagpe -> h0'),

                # MHA
                (nn.BatchNorm1d(embed_dim), 'h0 -> h0_norm'),
                (gnn.TransformerConv(
                    embed_dim, head_dim, heads=num_heads, root_weight=False),
                 'h0_norm, edge_index -> h1'),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(inplace=True),

                # Residual connection (replace GatingMechanism with simple residual)
                (ResidualConnection(embed_dim), 'h0, h1 -> h1'),

                # FFN
                (nn.BatchNorm1d(embed_dim), 'h1 -> h1_norm'),
                (gnn.MLP([embed_dim, 2 * embed_dim, embed_dim], 
                         norm=None, act_kwargs={'inplace': True}),
                 'h1_norm -> h2'),
                nn.ReLU(inplace=True),

                # Residual connection
                (ResidualConnection(embed_dim), 'h1, h2 -> h0')
            ]
            for _ in range(num_layers)
        ]
        transformer_layers = list(chain(*transformer_layers))

        self.node_encoder = gnn.Sequential(
            'x, edge_index, depth', prep_layer + transformer_layers)


    def forward(self, dag_batch):
        '''
            Returns:
                a dict of representations at three different levels:
                node, dag, and global.
        '''
        h_node = self.node_encoder(
            dag_batch.x, dag_batch.edge_index, dag_batch.node_depth)

        h_dag = self.dag_encoder(
            h_node, dag_batch)

        try:
            # batch of observations
            obs_ptr = dag_batch['obs_ptr']
        except:
            # single observation
            obs_ptr = torch.tensor(
                [0, dag_batch.num_graphs], device=h_dag.device, 
                dtype=torch.long)
            
        h_glob = self.global_encoder(h_dag, obs_ptr)

        h_dict = {
            'node': h_node, # shape: (num_nodes, embed_dim)
            'dag': h_dag,   # shape: (num_graphs, embed_dim)
            'glob': h_glob  # shape: (num_obsns, embed_dim)
        }

        return h_dict
    


class ResidualConnection(torch.nn.Module):
    def __init__(self, d_input):
        super(ResidualConnection, self).__init__()
        self.linear = torch.nn.Linear(d_input, d_input)

    def forward(self, x, y):
        # Apply a linear transformation to y
        y_transformed = self.linear(y)
        # Add x (input) and y_transformed (output of linear transformation)
        return x + y_transformed


class DagEncoder(nn.Module):
    def __init__(self, dim_node_emb, dim_emb):
        super().__init__()
        self.lin = nn.Linear(dim_node_emb, dim_emb)

    def forward(self, h_node, dag_batch):
        '''max pool over terminal node representations for each dag'''
        node_mask = dag_batch['stage_mask']
        
        # readout using only the schedulable nodes
        h_dag = gnn.global_max_pool(
            h_node[node_mask], 
            dag_batch.batch[node_mask], 
            size=dag_batch.num_graphs)

        return self.lin(h_dag)
