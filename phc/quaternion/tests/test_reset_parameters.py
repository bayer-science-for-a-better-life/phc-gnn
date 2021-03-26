import unittest
import torch
import itertools


from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
ATOM_FEATS, BOND_FEATS = get_atom_feature_dims(), get_bond_feature_dims()

from phc.quaternion.layers import QLinear
from phc.quaternion.encoder import IntegerEncoder, QuaternionEncoder

# quaternion undirectional models
from phc.quaternion.undirectional.models import QuaternionSkipConnectAdd as UQ_SC_ADD
from phc.quaternion.undirectional.models  import QuaternionSkipConnectConcat as UQ_SC_CAT


def weights_flag(name):
    """ Select only the weight matrices of QLinear and nn.Embedding """
    return "W_" in name or "weight" in name and "b_" not in name and "bias" not in name and "bn" not in name


def check_qlinear(in_features: int, out_features: int, init: str,
                  norm_tol: float = 5.0, nruns: int = 10) -> bool:
    weights = []
    module = QLinear(in_features, out_features, bias=False, init=init)
    for i in range(nruns):
        weights.append(module.W.stack(dim=0))
        module.reset_parameters()

    # check that each combination is different
    # get tuple combinations
    combinations = list(itertools.combinations([i for i in range(10)], 2))
    bools_list = []
    for c0, c1 in combinations:
        w0, w1 = weights[c0], weights[c1]
        is_not_equal = (w0 - w1).norm(p=2).item() > norm_tol
        bools_list.append(is_not_equal)

    return sum(bools_list) / len(bools_list) == 1.0


class TestResetModels(unittest.TestCase):

    def test_reset_qlinear(self):
        in_features = 32
        out_features = 64

        #  quaternion init
        quaternion_init_res = check_qlinear(in_features, out_features,
                                            init="quaternion", norm_tol=5.0, nruns=10)
        assert quaternion_init_res, "quaternion test init failed for QLinear"

        # orthogonal init
        orthogonal_init_res = check_qlinear(in_features, out_features,
                                            init="orthogonal", norm_tol=5.0, nruns=10)
        assert orthogonal_init_res, "orthogonal test init failed for QLinear"

        # glorot normal
        glorot_normal = check_qlinear(in_features, out_features,
                                      init="glorot-normal", norm_tol=5.0, nruns=10)

        assert glorot_normal, "glorot-normal test init failed for QLinear"

        # glorot uniform
        glorot_uniform = check_qlinear(in_features, out_features,
                                       init="glorot-uniform", norm_tol=5.0, nruns=10)
        assert glorot_uniform, "glorot-uniform test init gailed for QLinear"


    def test_reset_encoder(self):
        out_dim = 32
        input_dims = ATOM_FEATS
        integer_encoder = IntegerEncoder(out_dim=out_dim, input_dims=input_dims, combine="sum")
        nruns = 10
        weights = []
        norm_tol = 5.0
        for i in range(nruns):
            embeddings = torch.cat([emb.weight for emb in integer_encoder.embeddings], dim=0)
            weights.append(embeddings)
            integer_encoder.reset_parameters()

        combinations = list(itertools.combinations([i for i in range(10)], 2))
        bools_list = []

        # check that each combination is different
        # get tuple combinations
        for c0, c1 in combinations:
            w0, w1 = weights[c0], weights[c1]
            is_not_equal = (w0 - w1).norm(p=2).item() > norm_tol
            bools_list.append(is_not_equal)
        assert sum(bools_list) / len(bools_list) == 1.0, "IntegerEncoder init has an error"


    def test_reset_quaternionencoder(self):
        out_dim = 32
        nruns = 10
        norm_tol = 5.0
        qencoder = QuaternionEncoder(input_dims=ATOM_FEATS, out_dim=out_dim, combine="sum")

        weights = []
        for i in range(nruns):
            w_r = torch.cat([emb.weight for emb in qencoder.r.embeddings], dim=0)
            w_i = torch.cat([emb.weight for emb in qencoder.i.embeddings], dim=0)
            w_j = torch.cat([emb.weight for emb in qencoder.j.embeddings], dim=0)
            w_k = torch.cat([emb.weight for emb in qencoder.k.embeddings], dim=0)

            w = torch.cat([w_r, w_i, w_j, w_k], dim=0)
            weights.append(w)
            qencoder.reset_parameters()

        combinations = list(itertools.combinations([i for i in range(10)], 2))
        bools_list = []
        # check that each combination is different
        # get tuple combinations
        for c0, c1 in combinations:
            w0, w1 = weights[c0], weights[c1]
            is_not_equal = (w0 - w1).norm(p=2).item() > norm_tol
            bools_list.append(is_not_equal)

        assert sum(bools_list) / len(bools_list) == 1.0, "QuaternionEncoder init has an error"

    def test_reset_parameters_affine_weights_embedding_weights(self):
        nruns = 10
        params_model_3 = []
        params_model_4 = []


        model3 = UQ_SC_ADD(init="quaternion", naive_encoder=False).train()
        model4 = UQ_SC_CAT(init="quaternion", naive_encoder=False).train()

        # get weight matrices of quaternion.layers.QLinear and weight matrices of torch.nn.Embedding
        for i in range(nruns):
            model3.reset_parameters()
            model4.reset_parameters()

            params_model_3 += [[param.data.detach().clone() for name, param in
                                model3.named_parameters() if weights_flag(name)]]
            params_model_4 += [[param.data.detach().clone() for name, param in
                                model4.named_parameters() if weights_flag(name)]]

        # get tuple combinations
        combinations = list(itertools.combinations([i for i in range(nruns)], 2))
        for c0, c1 in combinations:
            # model 1
            bools_list = []
            for i in range(len(params_model_3[0])):
                is_equal = torch.allclose(params_model_3[c0][i], params_model_3[c1][i])
                bools_list.append(not is_equal)


        assert sum(bools_list) / len(bools_list) == 1.0

        nruns = 10
        params_model_3 = []
        params_model_4 = []


        model3 = UQ_SC_ADD(init="orthogonal", naive_encoder=False).train()
        model4 = UQ_SC_CAT(init="orthogonal", naive_encoder=False).train()

        # get weight matrices of quaternion.layers.QLinear and weight matrices of torch.nn.Embedding
        for i in range(nruns):
            model3.reset_parameters()
            model4.reset_parameters()

            params_model_3 += [[param.data.detach().clone() for name, param in
                                model3.named_parameters() if weights_flag(name)]]
            params_model_4 += [[param.data.detach().clone() for name, param in
                                model4.named_parameters() if weights_flag(name)]]

        # get tuple combinations
        combinations = list(itertools.combinations([i for i in range(nruns)], 2))
        for c0, c1 in combinations:
            for model in [params_model_3, params_model_4]:
                bools_list = []
                for i in range(len(model[0])):
                    is_equal = torch.allclose(model[c0][i], model[c1][i])
                    bools_list.append(not is_equal)

                assert sum(bools_list) / len(bools_list) == 1.0