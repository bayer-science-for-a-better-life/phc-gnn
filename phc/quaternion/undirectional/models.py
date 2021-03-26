import torch
import torch.nn as nn
from typing import Union, Optional

from torch_geometric.data import Batch
from torch_geometric.typing import Adj, Size

from phc.quaternion.algebra import QTensor
from phc.quaternion.algebra import cat as qcat
from phc.quaternion.activations import get_functional_activation
from phc.quaternion.downstream import QuaternionDownstreamNet
from phc.quaternion.undirectional.messagepassing import QMessagePassing
from phc.quaternion.encoder import NaiveQuaternionEncoder, QuaternionEncoder
from phc.quaternion.norm import QuaternionNorm
from phc.quaternion.layers import quaternion_dropout, QLinear
from phc.quaternion.pooling import QuaternionGlobalSumPooling, QuaternionSoftAttentionPooling


from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

ATOM_FEAT_DIMS = get_atom_feature_dims()
BOND_FEAT_DIMS = get_bond_feature_dims()


class QuaternionSkipConnectAdd(nn.Module):
    """  Undirectional Message Passing Network that utilizes Skip-Connections through Addition """

    def __init__(self,
                 atom_input_dims: Union[int, list] = ATOM_FEAT_DIMS,
                 atom_encoded_dim: int = 196,
                 bond_input_dims: Union[int, list] = BOND_FEAT_DIMS,
                 naive_encoder: bool = False,
                 init: str = "orthogonal", same_dropout: bool = False,
                 mp_layers: list = [196, 196, 196], bias: bool = True, dropout_mpnn: list = [0.0, 0.0, 0.0],
                 norm_mp: Optional[str] = "naive-batch-norm", add_self_loops: bool = True,
                 msg_aggr: str = "add", node_aggr: str = "sum", mlp: bool = False,
                 pooling: str = "softattention", activation: str = "relu", real_trafo: str = "linear",
                 downstream_layers: list = [256, 128], target_dim: int = 1,
                 dropout_dn: Union[list, float] = [0.2, 0.1], norm_dn: Optional[str] = "naive-batch-norm",
                 msg_encoder: str = "identity",
                 **kwargs) -> None:
        super(QuaternionSkipConnectAdd, self).__init__()
        assert all(x == atom_encoded_dim == mp_layers[0] for x in mp_layers), "dimensionalities need to match for model"
        assert activation.lower() in ["relu", "lrelu", "elu", "selu", "swish"]
        assert len(dropout_mpnn) == len(mp_layers)
        assert pooling in ["globalsum", "softattention"], f"pooling variable '{pooling}' wrong."
        assert norm_mp in ["None", None, "naive-batch-norm", "q-batch-norm"]

        if msg_aggr == "sum":  # for pytorch_geometrics MessagePassing class.
            msg_aggr = "add"

        self.msg_encoder_str = msg_encoder
        # save input args as attributes
        self.atom_input_dims = atom_input_dims
        self.bond_input_dims = bond_input_dims

        # one quaternion number consists of four components, so divide the feature dims by 4
        atom_encoded_dim = atom_encoded_dim // 4
        mp_layers = [dim // 4 for dim in mp_layers]
        downstream_layers = [dim // 4 for dim in downstream_layers]

        self.atom_encoded_dim = atom_encoded_dim
        self.naive_encoder = naive_encoder
        self.init = init
        self.same_dropout = same_dropout
        self.mp_layers = mp_layers
        self.bias = bias
        self.dropout_mpnn = dropout_mpnn
        self.norm_mp = norm_mp
        self.add_self_loops = add_self_loops
        self.msg_aggr_type = msg_aggr
        self.node_aggr_type = node_aggr
        self.mlp_mp = mlp
        self.pooling_type = pooling
        self.activation_str = activation
        self.real_trafo_type = real_trafo
        self.downstream_layers = downstream_layers
        self.target_dim = target_dim
        self.dropout_dn = dropout_dn
        self.norm_dn_type = norm_dn

        # define other attributes needed for module
        self.input_dim = atom_encoded_dim
        self.f_act = get_functional_activation(self.activation_str)
        # Quaternion MP layers
        self.convs = [None] * len(mp_layers)
        # batch normalization layers
        self.norms = [None] * len(mp_layers)

        # atom-encoder
        if not naive_encoder:
            self.atomencoder = QuaternionEncoder(out_dim=atom_encoded_dim, input_dims=atom_input_dims,
                                                 combine="sum")
        else:
            self.atomencoder = NaiveQuaternionEncoder(out_dim=atom_encoded_dim, input_dims=atom_input_dims,
                                                      combine="sum")

        # bond-encoder
        if not naive_encoder:
            self.bondencoders = [QuaternionEncoder(out_dim=odim,
                                                   input_dims=bond_input_dims,
                                                   combine="sum") for odim in mp_layers]
        else:
            self.bondencoders = [NaiveQuaternionEncoder(out_dim=odim,
                                                        input_dims=bond_input_dims,
                                                        combine="sum") for odim in mp_layers]

        self.bondencoders = nn.ModuleList(self.bondencoders)

        # prepare Quaternion MP layers and Norm if applicable
        for i in range(len(mp_layers)):
            if i == 0:
                in_dim = self.input_dim
            else:
                in_dim = self.mp_layers[i - 1]
            out_dim = self.mp_layers[i]
            self.convs[i] = QMessagePassing(in_features=in_dim, out_features=out_dim, bias=bias,
                                            norm=norm_mp, activation=activation, init=init, aggr=msg_aggr,
                                            mlp=mlp, same_dim=True, add_self_loops=add_self_loops,
                                            msg_encoder=msg_encoder,
                                            **kwargs)

            if norm_mp:
                self.norms[i] = QuaternionNorm(num_features=out_dim, type=norm_mp)


        self.convs = nn.ModuleList(self.convs)
        if norm_mp:
            self.norms = nn.ModuleList(self.norms)

        if pooling == "globalsum":
            self.pooling = QuaternionGlobalSumPooling()
        else:
            self.pooling = QuaternionSoftAttentionPooling(embed_dim=self.mp_layers[-1],
                                                          init=self.init,
                                                          bias=self.bias,
                                                          real_trafo=self.real_trafo_type)


        # downstream network
        self.downstream = QuaternionDownstreamNet(in_features=self.mp_layers[-1],
                                                  hidden_layers=self.downstream_layers,
                                                  out_features=self.target_dim,
                                                  activation=self.activation_str,
                                                  bias=self.bias, norm=self.norm_dn_type, init=self.init,
                                                  dropout=self.dropout_dn, same_dropout=self.same_dropout,
                                                  real_trafo=self.real_trafo_type)
        self.reset_parameters()

    def reset_parameters(self):

        # atom encoder
        self.atomencoder.reset_parameters()

        # bond encoders
        for encoder in self.bondencoders:
            encoder.reset_parameters()

        # mp and norm layers
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            if self.norm_mp:
                norm.reset_parameters()

        # pooling
        self.pooling.reset_parameters()

        # downstream network
        self.downstream.reset_parameters()

    def get_number_of_params_(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


    def compute_hidden_layer_embedding(self, conv: QMessagePassing, norm: Optional[QuaternionNorm],
                                       q: Union[QTensor, list], edge_index: Adj, edge_attr: QTensor,
                                       dropout_mpnn: float, size: Size = None) -> QTensor:

        tmp = q
        # apply message passing
        q = conv(q=tmp[0], edge_index=edge_index, edge_attr=edge_attr, size=size)
        # apply normalization
        if type(norm) is not None:
            q = norm(q)
        # apply non-linearity
        q = self.f_act(q)
        # apply dropout with train-mode flag
        q = quaternion_dropout(q=q, p=dropout_mpnn,
                               training=self.training, same=self.same_dropout)
        # skip connect through addition
        q = q + tmp[1]
        del tmp
        return q

    def forward(self, data: Batch, size: Size = None) -> QTensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if isinstance(self.bond_input_dims, list):
            edge_attr = edge_attr.to(torch.long)
        atom_encoded = self.atomencoder(x)
        for i in range(len(self.mp_layers)):
            if i == 0:
                q = [atom_encoded.clone(), atom_encoded.clone()]
            else:  # skip connect
                q = [q, atom_encoded.clone()]
            hidden_edge_attr = self.bondencoders[i](edge_attr)
            q = self.compute_hidden_layer_embedding(conv=self.convs[i], norm=self.norms[i],
                                                    q=q, edge_index=edge_index, edge_attr=hidden_edge_attr,
                                                    dropout_mpnn=self.dropout_mpnn[i], size=size)

        # apply graph pooling
        out = self.pooling(x=q, batch=batch)
        # downstream network prediction
        out = self.downstream(out)
        return out

    def __repr__(self):
        return "{}(atom_input_dim={}, atom_encoded_dim={}, " \
               "bond_input_dims={}, naive_encoder={}, init='{}', " \
               "same_dropout={}, mp_layers={}, bias={}, dropout_mpnn={}," \
               "norm_mp='{}', add_self_loops={}," \
               "msg_aggr='{}', node_aggr={} mlp={}, " \
               "pooling='{}', activation='{}', real_trafo='{}'," \
               "downstream_layers={}, target_dim={}, dropout_dn={}, " \
               "norm_dn={})".format(self.__class__.__name__,
                                    self.atom_input_dims, self.atom_encoded_dim,
                                    self.bond_input_dims, self.naive_encoder, self.init,
                                    self.same_dropout, self.mp_layers, self.bias, self.dropout_mpnn,
                                    self.norm_mp, self.add_self_loops, self.msg_aggr_type, self.node_aggr_type,
                                    self.mlp_mp, self.pooling_type, self.activation_str, self.real_trafo_type,
                                    self.downstream_layers, self.target_dim, self.dropout_dn, self.norm_dn_type)



class QuaternionSkipConnectConcat(nn.Module):
    """  Undirectional Message Passing Network that utilizes Skip-Connections through Concatenation """

    def __init__(self,
                 atom_input_dims: Union[int, list] = ATOM_FEAT_DIMS,
                 atom_encoded_dim: int = 128,
                 bond_input_dims: Union[int, list] = BOND_FEAT_DIMS,
                 naive_encoder: bool = False,
                 init: str = "orthogonal", same_dropout: bool = False,
                 mp_layers: list = [128, 196, 256], bias: bool = True, dropout_mpnn: list = [0.0, 0.0, 0.0],
                 norm_mp: Optional[str] = "naive-batch-norm", add_self_loops: bool = True,
                 msg_aggr: str = "add", node_aggr: str = "sum", mlp: bool = False,
                 pooling: str = "softattention", activation: str = "relu", real_trafo: str = "linear",
                 downstream_layers: list = [256, 128], target_dim: int = 1,
                 dropout_dn: Union[list, float] = [0.2, 0.1], norm_dn: Optional[str] = "naive-batch-norm",
                 msg_encoder: str = "identity",
                 **kwargs) -> None:
        super(QuaternionSkipConnectConcat, self).__init__()

        assert activation.lower() in ["relu", "lrelu", "elu", "selu", "swish"]
        assert len(dropout_mpnn) == len(mp_layers)
        assert pooling in ["globalsum", "softattention"], f"pooling variable '{pooling}' wrong."
        assert norm_mp in ["None", None, "naive-batch-norm", "q-batch-norm"]

        if msg_aggr == "sum":  # for pytorch_geometrics MessagePassing class.
            msg_aggr = "add"

        self.msg_encoder_str = msg_encoder
        # save input args as attributes
        self.atom_input_dims = atom_input_dims
        self.bond_input_dims = bond_input_dims

        # one quaternion number consists of four components, so divide the feature dims by 4
        atom_encoded_dim = atom_encoded_dim // 4
        mp_layers = [dim // 4 for dim in mp_layers]
        downstream_layers = [dim // 4 for dim in downstream_layers]

        self.atom_encoded_dim = atom_encoded_dim
        self.naive_encoder = naive_encoder
        self.init = init
        self.same_dropout = same_dropout
        self.mp_layers = mp_layers
        self.bias = bias
        self.dropout_mpnn = dropout_mpnn
        self.norm_mp = norm_mp
        self.add_self_loops = add_self_loops
        self.msg_aggr_type = msg_aggr
        self.node_aggr_type = node_aggr
        self.mlp_mp = mlp
        self.pooling_type = pooling
        self.activation_str = activation
        self.real_trafo_type = real_trafo
        self.downstream_layers = downstream_layers
        self.target_dim = target_dim
        self.dropout_dn = dropout_dn
        self.norm_dn_type = norm_dn

        # define other attributes needed for module
        self.input_dim = atom_encoded_dim
        self.f_act = get_functional_activation(self.activation_str)
        # Quaternion MP layers
        self.convs = [None] * len(mp_layers)
        # batch normalization layers
        self.norms = [None] * len(mp_layers)

        dims = [atom_encoded_dim] + mp_layers
        # atom-encoder
        if not naive_encoder:
            self.atomencoder = QuaternionEncoder(out_dim=atom_encoded_dim, input_dims=atom_input_dims,
                                                 combine="sum")
        else:
            self.atomencoder = NaiveQuaternionEncoder(out_dim=atom_encoded_dim, input_dims=atom_input_dims,
                                                      combine="sum")

        # bond-encoder
        self.bondencoders = []
        if not naive_encoder:
            module = QuaternionEncoder
        else:
            module = NaiveQuaternionEncoder
        for i in range(len(mp_layers)):
            if i == 0:
                out_dim = self.input_dim
            else:
                out_dim = self.mp_layers[i-1] + self.input_dim

            self.bondencoders.append(
                module(input_dims=bond_input_dims, out_dim=out_dim, combine="sum")
            )

        self.bondencoders = nn.ModuleList(self.bondencoders)

        # prepare Quaternion MP layers and Norm if applicable
        for i in range(len(mp_layers)):
            if i == 0:
                in_dim = self.input_dim
            else:
                in_dim = self.mp_layers[i - 1] + self.input_dim
            out_dim = self.mp_layers[i]
            self.convs[i] = QMessagePassing(in_features=in_dim, out_features=out_dim, bias=bias,
                                            norm=norm_mp, activation=activation, init=init, aggr=msg_aggr,
                                            mlp=mlp, same_dim=False, add_self_loops=add_self_loops,
                                            msg_encoder=msg_encoder,
                                            **kwargs)

            if norm_mp:
                self.norms[i] = QuaternionNorm(num_features=out_dim, type=norm_mp)


        self.convs = nn.ModuleList(self.convs)
        if norm_mp:
            self.norms = nn.ModuleList(self.norms)

        if pooling == "globalsum":
            self.pooling = QuaternionGlobalSumPooling()
        else:
            self.pooling = QuaternionSoftAttentionPooling(embed_dim=self.mp_layers[-1] + self.input_dim,
                                                          init=self.init,
                                                          bias=self.bias,
                                                          real_trafo=self.real_trafo_type)


        # downstream network
        self.downstream = QuaternionDownstreamNet(in_features=self.mp_layers[-1] + self.input_dim,
                                                  hidden_layers=self.downstream_layers,
                                                  out_features=self.target_dim,
                                                  activation=self.activation_str,
                                                  bias=self.bias, norm=self.norm_dn_type, init=self.init,
                                                  dropout=self.dropout_dn, same_dropout=self.same_dropout,
                                                  real_trafo=self.real_trafo_type)
        self.reset_parameters()

    def reset_parameters(self):

        # atom encoder
        self.atomencoder.reset_parameters()

        # bond encoders
        for encoder in self.bondencoders:
            encoder.reset_parameters()

        # mp and norm layers
        for conv, norm in zip(self.convs, self.norms):
            conv.reset_parameters()
            if self.norm_mp:
                norm.reset_parameters()

        # pooling
        self.pooling.reset_parameters()

        # downstream network
        self.downstream.reset_parameters()

    def get_number_of_params_(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


    def compute_hidden_layer_embedding(self, conv: QMessagePassing, norm: Optional[QuaternionNorm],
                                       q: Union[QTensor, list], edge_index: Adj, edge_attr: QTensor,
                                       dropout_mpnn: float, size: Size = None) -> QTensor:

        tmp = q
        # apply message passing
        q = conv(q=tmp[0], edge_index=edge_index, edge_attr=edge_attr, size=size)
        # apply normalization
        if type(norm) is not None:
            q = norm(q)
        # apply non-linearity
        q = self.f_act(q)
        # apply dropout with train-mode flag
        q = quaternion_dropout(q=q, p=dropout_mpnn,
                               training=self.training, same=self.same_dropout)
        # skip connect through concatenation
        q = qcat([q, tmp[1]], dim=-1)
        del tmp
        return q

    def forward(self, data: Batch, size: Size = None) -> QTensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if isinstance(self.bond_input_dims, list):
            edge_attr = edge_attr.to(torch.long)
        atom_encoded = self.atomencoder(x)
        for i in range(len(self.mp_layers)):
            if i == 0:
                q = [atom_encoded.clone(), atom_encoded.clone()]
            else:  # skip connect
                q = [q, atom_encoded.clone()]
            hidden_edge_attr = self.bondencoders[i](edge_attr)
            q = self.compute_hidden_layer_embedding(conv=self.convs[i], norm=self.norms[i],
                                                    q=q, edge_index=edge_index, edge_attr=hidden_edge_attr,
                                                    dropout_mpnn=self.dropout_mpnn[i], size=size)

        # apply graph pooling
        out = self.pooling(x=q, batch=batch)
        # downstream network prediction
        out = self.downstream(out)
        return out

    def __repr__(self):
        return "{}(atom_input_dim={}, atom_encoded_dim={}, " \
               "bond_input_dims={}, naive_encoder={}, init='{}', " \
               "same_dropout={}, mp_layers={}, bias={}, dropout_mpnn={}," \
               "norm_mp='{}', add_self_loops={}," \
               "msg_aggr='{}', node_aggr={} mlp={}, " \
               "pooling='{}', activation='{}', real_trafo='{}'," \
               "downstream_layers={}, target_dim={}, dropout_dn={}, " \
               "norm_dn={})".format(self.__class__.__name__,
                                    self.atom_input_dims, self.atom_encoded_dim,
                                    self.bond_input_dims, self.naive_encoder, self.init,
                                    self.same_dropout, self.mp_layers, self.bias, self.dropout_mpnn,
                                    self.norm_mp, self.add_self_loops, self.msg_aggr_type, self.node_aggr_type,
                                    self.mlp_mp, self.pooling_type, self.activation_str, self.real_trafo_type,
                                    self.downstream_layers, self.target_dim, self.dropout_dn, self.norm_dn_type)


