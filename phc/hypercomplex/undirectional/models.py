import torch
import torch.nn as nn
from typing import Union, Optional

from torch_geometric.data import Batch
from torch_geometric.typing import Adj, Size

from phc.quaternion.activations import get_module_activation

from phc.hypercomplex.downstream import PHMDownstreamNet
from phc.hypercomplex.undirectional.messagepassing import PHMMessagePassing
from phc.hypercomplex.encoder import NaivePHMEncoder, PHMEncoder
from phc.hypercomplex.norm import PHMNorm
from phc.hypercomplex.layers import phm_dropout, PHMLinear
from phc.hypercomplex.pooling import PHMGlobalSumPooling, PHMSoftAttentionPooling
from phc.hypercomplex.utils import get_multiplication_matrices

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

ATOM_FEAT_DIMS = get_atom_feature_dims()
BOND_FEAT_DIMS = get_bond_feature_dims()


class PHMSkipConnectAdd(nn.Module):
    """  Undirectional Message Passing Network that utilizes Skip-Connections through Addition """

    def __init__(self,
                 phm_dim: int = 4, learn_phm: bool = True,
                 phm_rule: Union[None, nn.ParameterList] = None,
                 atom_input_dims: Union[int, list] = ATOM_FEAT_DIMS,
                 atom_encoded_dim: int = 196,
                 bond_input_dims: Union[int, list] = BOND_FEAT_DIMS,
                 naive_encoder: bool = False,
                 w_init: str = "phm", c_init: str = "standard",
                 same_dropout: bool = False,
                 mp_layers: list = [196, 196, 196], bias: bool = True, dropout_mpnn: list = [0.0, 0.0, 0.0],
                 norm_mp: Optional[str] = "naive-batch-norm", add_self_loops: bool = True,
                 msg_aggr: str = "add", node_aggr: str = "sum", mlp: bool = False,
                 pooling: str = "softattention", activation: str = "relu", real_trafo: str = "linear",
                 downstream_layers: list = [256, 128], target_dim: int = 1,
                 dropout_dn: Union[list, float] = [0.2, 0.1], norm_dn: Optional[str] = "naive-batch-norm",
                 msg_encoder: str = "identity",
                 sc_type: str = "first",
                 **kwargs) -> None:
        super(PHMSkipConnectAdd, self).__init__()
        assert all(x == atom_encoded_dim == mp_layers[0] for x in mp_layers), "dimensionalities need to match for model"
        assert activation.lower() in ["relu", "lrelu", "elu", "selu", "swish"]
        assert len(dropout_mpnn) == len(mp_layers)
        assert pooling in ["globalsum", "softattention"], f"pooling variable '{pooling}' wrong."
        assert norm_mp in [None, "naive-batch-norm", "None", "naive-naive-batch-norm"]
        assert norm_dn in [None, "naive-batch-norm", "None", "naive-naive-batch-norm"]
        assert w_init in ["phm", "glorot_uniform", "glorot_normal"], f"w_init variable '{w_init}' wrong."
        assert c_init in ["standard", "random"], f"c_init variable '{c_init}' wrong."


        if msg_aggr == "sum":  # for pytorch_geometrics MessagePassing class.
            msg_aggr = "add"
        self.msg_encoder_str = msg_encoder
        self.phm_rule = phm_rule
        if self.phm_rule is None:
            self.variable_phm = True
        else:
            self.variable_phm = False

        self.phm_dim = phm_dim
        self.learn_phm = learn_phm
        # save input args as attributes
        self.atom_input_dims = atom_input_dims
        self.bond_input_dims = bond_input_dims

        # for node and bond embedding that also uses encoders, here need integer division
        atom_encoded_dim = atom_encoded_dim // phm_dim
        mp_layers_div = [d // phm_dim for d in mp_layers]

        self.atom_encoded_dim = atom_encoded_dim
        self.naive_encoder = naive_encoder
        self.w_init = w_init
        self.c_init = c_init
        self.same_dropout = same_dropout
        self.mp_layers = mp_layers
        self.bias = bias
        self.dropout_mpnn = dropout_mpnn
        self.norm_mp = None if norm_mp=="None" else norm_mp
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
        self.norm_dn_type = None if norm_dn=="None" else norm_dn

        # define other attributes needed for module
        self.input_dim = atom_encoded_dim
        self.f_act = get_module_activation(self.activation_str)
        # PHM MP layers
        self.convs = nn.ModuleList([None] * len(mp_layers))
        # batch normalization layers
        self.norms = nn.ModuleList([None] * len(mp_layers))

        self.sc_type = sc_type

        # atom embedding
        if naive_encoder:
            self.atomencoder = NaivePHMEncoder(out_dim=atom_encoded_dim, input_dims=atom_input_dims, phm_dim=phm_dim,
                                               combine="sum")
        else:
            self.atomencoder = PHMEncoder(out_dim=atom_encoded_dim, input_dims=atom_input_dims, phm_dim=phm_dim,
                                          combine="sum")

        # bond/edge embeddings
        if naive_encoder:
            self.bondencoders = [NaivePHMEncoder(out_dim=odim, input_dims=bond_input_dims, phm_dim=phm_dim,
                                                 combine="sum") for odim in mp_layers_div]
        else:
            self.bondencoders = [PHMEncoder(out_dim=odim, input_dims=bond_input_dims, phm_dim=phm_dim,
                                            combine="sum") for odim in mp_layers_div]


        self.bondencoders = nn.ModuleList(self.bondencoders)

        # prepare Quaternion MP layers and Norm if applicable
        for i in range(len(mp_layers)):
            if i == 0:
                in_dim = self.input_dim
            else:
                in_dim = self.mp_layers[i - 1]
            out_dim = self.mp_layers[i]
            self.convs[i] = PHMMessagePassing(in_features=in_dim, out_features=out_dim, bias=bias,
                                              phm_dim=phm_dim, learn_phm=learn_phm, phm_rule=self.phm_rule,
                                              norm=self.norm_mp, activation=activation, w_init=w_init, c_init=c_init,
                                              aggr=msg_aggr, mlp=mlp,
                                              add_self_loops=add_self_loops,
                                              same_dim=True, msg_encoder=msg_encoder,
                                              **kwargs)

            if self.norm_mp:
                self.norms[i] = PHMNorm(num_features=out_dim, phm_dim=phm_dim, type=norm_mp)

        if pooling == "globalsum":
            self.pooling = PHMGlobalSumPooling(phm_dim=phm_dim)
        else:
            self.pooling = PHMSoftAttentionPooling(embed_dim=self.mp_layers[-1],
                                                   phm_dim=phm_dim,
                                                   learn_phm=learn_phm,
                                                   phm_rule=self.phm_rule,
                                                   w_init=self.w_init,
                                                   c_init=self.c_init,
                                                   bias=self.bias,
                                                   real_trafo=self.real_trafo_type)


        # downstream network
        self.downstream = PHMDownstreamNet(in_features=self.mp_layers[-1],
                                           hidden_layers=self.downstream_layers,
                                           out_features=self.target_dim,
                                           phm_rule=self.phm_rule,
                                           phm_dim=phm_dim,
                                           learn_phm=learn_phm,
                                           activation=self.activation_str,
                                           bias=self.bias, norm=self.norm_dn_type,
                                           w_init=self.w_init, c_init=self.c_init,
                                           dropout=self.dropout_dn, same_dropout=self.same_dropout,
                                           real_trafo=self.real_trafo_type)
        self.reset_parameters()

    def reset_parameters(self):

        if not self.variable_phm:
            phm_rule = get_multiplication_matrices(phm_dim=self.phm_dim)
            for i, init_data in enumerate(phm_rule):
                self.phm_rule[i].data = init_data

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


    def compute_hidden_layer_embedding(self, conv: PHMMessagePassing, norm: Optional[PHMNorm],
                                       x: Union[torch.Tensor, list], edge_index: Adj, edge_attr: torch.Tensor,
                                       dropout_mpnn: float, size: Size = None) -> torch.Tensor:

        tmp = x
        # apply message passing
        x = conv(x=tmp[0], edge_index=edge_index, edge_attr=edge_attr, size=size)
        # apply normalization
        if norm:
            x = norm(x)
        # apply non-linearity
        x = self.f_act(x)
        # apply dropout with train-mode flag
        x = phm_dropout(x=x, p=dropout_mpnn, phm_dim=self.phm_dim, training=self.training, same=self.same_dropout)
        # skip connect through addition
        x = x + tmp[1]
        del tmp
        return x

    def forward(self, data: Batch, size: Size = None) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if isinstance(self.bond_input_dims, list):
            edge_attr = edge_attr.to(torch.long)
        atom_encoded = self.atomencoder(x)

        atom_encoded = atom_encoded.reshape(atom_encoded.size(0), self.phm_dim * self.atom_encoded_dim)
        for i in range(len(self.mp_layers)):
            if i == 0:
                x = [atom_encoded.clone(), atom_encoded.clone()]
            else:  # skip connect
                if self.sc_type == "first":
                    x = [x, atom_encoded.clone()]
                elif self.sc_type == "last":
                    #  skip-connections previous
                    x = [x.clone(), x.clone()]  ## comment out again
                else:
                    raise ValueError

            hidden_edge_attr = self.bondencoders[i](edge_attr)
            hidden_edge_attr = hidden_edge_attr.reshape(hidden_edge_attr.size(0),
                                                        self.phm_dim * self.mp_layers[i])
            x = self.compute_hidden_layer_embedding(conv=self.convs[i], norm=self.norms[i],
                                                    x=x, edge_index=edge_index, edge_attr=hidden_edge_attr,
                                                    dropout_mpnn=self.dropout_mpnn[i], size=size)

        # apply graph pooling
        out = self.pooling(x=x, batch=batch)
        # downstream network prediction
        out = self.downstream(out)
        return out

    def __repr__(self):
        return "{}(phm_dim={}, learn_phm={}, phm_rule={}, " \
               "atom_input_dim={}, atom_encoded_dim={}, " \
               "bond_input_dims={}, naive_encoder={}, w_init='{}', c_init='{}', " \
               "same_dropout={}, mp_layers={}, bias={}, dropout_mpnn={}," \
               "norm_mp='{}', add_self_loops={}," \
               "msg_aggr='{}', node_aggr={} mlp={}, " \
               "pooling='{}', activation='{}', real_trafo='{}'," \
               "downstream_layers={}, target_dim={}, dropout_dn={}, " \
               "norm_dn={})".format(self.__class__.__name__,
                                    self.phm_dim, self.learn_phm, self.phm_rule,
                                    self.atom_input_dims, self.atom_encoded_dim,
                                    self.bond_input_dims, self.naive_encoder, self.w_init, self.c_init,
                                    self.same_dropout, self.mp_layers, self.bias, self.dropout_mpnn,
                                    self.norm_mp, self.add_self_loops, self.msg_aggr_type, self.node_aggr_type,
                                    self.mlp_mp, self.pooling_type, self.activation_str, self.real_trafo_type,
                                    self.downstream_layers, self.target_dim, self.dropout_dn, self.norm_dn_type)



class PHMSkipConnectConcat(nn.Module):
    """  Undirectional Message Passing Network that utilizes Skip-Connections through Concatenation """

    def __init__(self,
                 phm_dim: int = 4, learn_phm: bool = True, phm_rule: Union[None, nn.ParameterList] = None,
                 atom_input_dims: Union[int, list] = ATOM_FEAT_DIMS,
                 atom_encoded_dim: int = 128,
                 bond_input_dims: Union[int, list] = BOND_FEAT_DIMS,
                 naive_encoder: bool = False,
                 w_init: str = "phm", c_init: str = "standard",
                 same_dropout: bool = False,
                 mp_layers: list = [128, 196, 256], bias: bool = True, dropout_mpnn: list = [0.0, 0.0, 0.0],
                 norm_mp: Optional[str] = "naive-batch-norm", add_self_loops: bool = True,
                 msg_aggr: str = "add", node_aggr: str = "sum", mlp: bool = False,
                 pooling: str = "softattention", activation: str = "relu", real_trafo: str = "linear",
                 downstream_layers: list = [256, 128], target_dim: int = 1,
                 dropout_dn: Union[list, float] = [0.2, 0.1], norm_dn: Optional[str] = "naive-batch-norm",
                 msg_encoder: str = "identity",
                 **kwargs) -> None:
        super(PHMSkipConnectConcat, self).__init__()
        assert activation.lower() in ["relu", "lrelu", "elu", "selu", "swish"]
        assert len(dropout_mpnn) == len(mp_layers)
        assert pooling in ["globalsum", "softattention"], f"pooling variable '{pooling}' wrong."
        assert norm_mp in [None, "naive-batch-norm", "None", "naive-naive-batch-norm"]
        assert w_init in ["phm", "glorot_uniform", "glorot_normal"], f"w_init variable '{w_init}' wrong."
        assert c_init in ["standard", "random"], f"c_init variable '{c_init}' wrong."

        if msg_aggr == "sum":  # for pytorch_geometrics MessagePassing class.
            msg_aggr = "add"

        self.msg_encoder_str = msg_encoder

        self.phm_rule = phm_rule
        if self.phm_rule is None:
            self.variable_phm = True
        else:
            self.variable_phm = False

        self.phm_dim = phm_dim
        self.learn_phm = learn_phm
        # save input args as attributes
        self.atom_input_dims = atom_input_dims
        self.bond_input_dims = bond_input_dims

        # for node and bond embedding that also uses encoders, here need integer division
        atom_encoded_dim = atom_encoded_dim // phm_dim
        mp_layers_div = [d // phm_dim for d in mp_layers]

        self.atom_encoded_dim = atom_encoded_dim
        self.naive_encoder = naive_encoder
        self.w_init = w_init
        self.c_init = c_init
        self.same_dropout = same_dropout
        self.mp_layers = mp_layers
        self.bias = bias
        self.dropout_mpnn = dropout_mpnn
        self.norm_mp = None if norm_mp == "None" else norm_mp
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
        self.norm_dn_type = None if norm_dn == "None" else norm_dn

        # define other attributes needed for module
        self.input_dim = atom_encoded_dim
        self.f_act = get_module_activation(self.activation_str)
        # PHM MP layers
        self.convs = nn.ModuleList([None] * len(mp_layers))
        # batch normalization layers
        self.norms = nn.ModuleList([None] * len(mp_layers))

        # atom embedding
        if naive_encoder:
            self.atomencoder = NaivePHMEncoder(out_dim=atom_encoded_dim, input_dims=atom_input_dims,
                                               phm_dim=phm_dim,
                                               combine="sum")
        else:
            self.atomencoder = PHMEncoder(out_dim=atom_encoded_dim, input_dims=atom_input_dims, phm_dim=phm_dim,
                                          combine="sum")


        # bond/edge embeddings
        self.bondencoders = []
        if naive_encoder:
            module = NaivePHMEncoder
        else:
            module = PHMEncoder

        for i in range(len(mp_layers)):
            if i == 0:
                out_dim = self.input_dim
            else:
                out_dim = self.mp_layers[i - 1] // phm_dim + self.input_dim

            self.bondencoders.append(
                module(input_dims=bond_input_dims, out_dim=out_dim, phm_dim=phm_dim, combine="sum")
            )

        self.bondencoders = nn.ModuleList(self.bondencoders)

        # prepare Quaternion MP layers and Norm if applicable
        for i in range(len(mp_layers)):
            if i == 0:
                in_dim = self.input_dim
            else:
                in_dim = self.mp_layers[i - 1] + self.input_dim
            out_dim = self.mp_layers[i]
            self.convs[i] = PHMMessagePassing(in_features=in_dim, out_features=out_dim, bias=bias,
                                              phm_dim=phm_dim, learn_phm=learn_phm, phm_rule=self.phm_rule,
                                              norm=self.norm_mp, activation=activation,
                                              w_init=w_init, c_init=c_init,
                                              aggr=msg_aggr, mlp=mlp,
                                              add_self_loops=add_self_loops,
                                              same_dim=False, msg_encoder=msg_encoder,
                                              **kwargs)

            if self.norm_mp:
                self.norms[i] = PHMNorm(num_features=out_dim, phm_dim=phm_dim, type=norm_mp)


        if pooling == "globalsum":
            self.pooling = PHMGlobalSumPooling(phm_dim=phm_dim)
        else:
            self.pooling = PHMSoftAttentionPooling(embed_dim=self.mp_layers[-1] + self.input_dim,
                                                   phm_dim=phm_dim,
                                                   learn_phm=learn_phm,
                                                   phm_rule=self.phm_rule,
                                                   w_init=self.w_init, c_init=self.c_init,
                                                   bias=self.bias,
                                                   real_trafo=self.real_trafo_type)


        # downstream network
        self.downstream = PHMDownstreamNet(in_features=self.mp_layers[-1] + self.input_dim,
                                           hidden_layers=self.downstream_layers,
                                           out_features=self.target_dim,
                                           phm_dim=phm_dim,
                                           learn_phm=learn_phm,
                                           phm_rule=self.phm_rule,
                                           activation=self.activation_str,
                                           bias=self.bias, norm=self.norm_dn_type,
                                           w_init=self.w_init, c_init=self.c_init,
                                           dropout=self.dropout_dn, same_dropout=self.same_dropout,
                                           real_trafo=self.real_trafo_type)
        self.reset_parameters()

    def reset_parameters(self):

        if not self.variable_phm:
            phm_rule = get_multiplication_matrices(phm_dim=self.phm_dim)
            for i, init_data in enumerate(phm_rule):
                self.phm_rule[i].data = init_data

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


    def compute_hidden_layer_embedding(self, conv: PHMMessagePassing, norm: Optional[PHMNorm],
                                       x: Union[torch.Tensor, list], edge_index: Adj, edge_attr: torch.Tensor,
                                       dropout_mpnn: float, size: Size = None) -> torch.Tensor:

        tmp = x
        # apply message passing
        x = conv(x=tmp[0], edge_index=edge_index, edge_attr=edge_attr, size=size)
        # apply normalization
        if norm:
            x = norm(x)
        # apply non-linearity
        x = self.f_act(x)
        # apply dropout with train-mode flag
        x = phm_dropout(x=x, p=dropout_mpnn, phm_dim=self.phm_dim, training=self.training, same=self.same_dropout)
        # skip connect through concatenation
        x = torch.cat([x, tmp[1]], dim=-1)
        del tmp
        return x


    def forward(self, data: Batch, size: Size = None) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if isinstance(self.bond_input_dims, list):
            edge_attr = edge_attr.to(torch.long)
        atom_encoded = self.atomencoder(x)
        atom_encoded = atom_encoded.reshape(atom_encoded.size(0), self.phm_dim * self.atom_encoded_dim)
        for i in range(len(self.mp_layers)):
            if i == 0:
                x = [atom_encoded.clone(), atom_encoded.clone()]
            else:  # skip connect
                x = [x, atom_encoded.clone()]

            hidden_edge_attr = self.bondencoders[i](edge_attr)
            if i == 0:
                hidden_edge_attr = hidden_edge_attr.reshape(hidden_edge_attr.size(0), self.phm_dim * self.input_dim)
            else:
                hidden_edge_attr = hidden_edge_attr.reshape(hidden_edge_attr.size(0),
                                                            self.phm_dim * (self.mp_layers[i-1] + self.input_dim))

            x = self.compute_hidden_layer_embedding(conv=self.convs[i], norm=self.norms[i],
                                                    x=x, edge_index=edge_index, edge_attr=hidden_edge_attr,
                                                    dropout_mpnn=self.dropout_mpnn[i], size=size)

        # apply graph pooling
        out = self.pooling(x=x, batch=batch)
        # downstream network prediction
        out = self.downstream(out)
        return out

    def __repr__(self):
        return "{}(phm_dim={}, learn_phm={}, phm_rule={}, " \
               "atom_input_dim={}, atom_encoded_dim={}, " \
               "bond_input_dims={}, naive_encoder={}, w_init='{}', c_init='{}', "\
               "same_dropout={}, mp_layers={}, bias={}, dropout_mpnn={}," \
               "norm_mp='{}', add_self_loops={}," \
               "msg_aggr='{}', node_aggr={} mlp={}, " \
               "pooling='{}', activation='{}', real_trafo='{}'," \
               "downstream_layers={}, target_dim={}, dropout_dn={}, " \
               "norm_dn={})".format(self.__class__.__name__,
                                    self.phm_dim, self.learn_phm, self.phm_rule,
                                    self.atom_input_dims, self.atom_encoded_dim,
                                    self.bond_input_dims, self.naive_encoder, self.w_init, self.c_init,
                                    self.same_dropout, self.mp_layers, self.bias, self.dropout_mpnn,
                                    self.norm_mp, self.add_self_loops, self.msg_aggr_type, self.node_aggr_type,
                                    self.mlp_mp, self.pooling_type, self.activation_str, self.real_trafo_type,
                                    self.downstream_layers, self.target_dim, self.dropout_dn, self.norm_dn_type)


"""
model1 = PHMSkipConnectAdd()
model2 = PHMSkipConnectConcat()
model1.get_number_of_params_()
model2.get_number_of_params_()
"""