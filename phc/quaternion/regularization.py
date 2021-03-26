import torch


def get_model_blocks(model, attr: str, **kwargs) -> list:
    r""" Returns the parameters stored in a list of an attribute from a nn.Module"""
    try:
        if hasattr(model.module, attr):
            module = model.module.__getattr__(attr)
            if len(list(module.parameters())) > 0:
                params = [dict(params=list(module.parameters()), **kwargs)]
            else:
                params = []
        else:
            params = []
    except:
        if hasattr(model, attr):
            module = model.__getattr__(attr)
            if len(list(module.parameters())) > 0:
                params = [dict(params=list(module.parameters()), **kwargs)]
            else:
                params = []
        else:
            params = []
    return params


def quaternion_weight_regularization(model, device, p: int = 1):
    assert p in [1, 2]
    reg_loss = 0.0

    directional = False
    undirectional = True
    q_embedding_weights_qlinear = []
    if hasattr(model, "embedding"):
        directional = True
        undirectional = False
        for name, param in model.embedding.qlinear.named_parameters():
            if "W_r" in name or "W_i" in name or "W_j" in name or "W_k" in name:
                q_embedding_weights_qlinear.append(param)

        q_embedding_weights_qlinear = torch.stack(q_embedding_weights_qlinear, dim=0).to(device)
        reg_loss += q_embedding_weights_qlinear.norm(p=p, dim=0).mean()

    # quaternion weights for message passing layers
    quaternion_weights = []
    for mp in model.convs:
        if directional:
            if mp.transform.__class__.__name__ == "QMLP":
                linear1_weight = torch.stack([mp.transform.qlinear1.W_r, mp.transform.qlinear1.W_i,
                                              mp.transform.qlinear1.W_j, mp.transform.qlinear1.W_k], dim=0).to(device)
                linear2_weight = torch.stack([mp.transform.qlinear2.W_r, mp.transform.qlinear2.W_i,
                                              mp.transform.qlinear2.W_j, mp.transform.qlinear2.W_k], dim=0).to(device)
                quaternion_weights.append(linear1_weight)
                quaternion_weights.append(linear2_weight)
            elif mp.transform.__class__.__name__ == "QLinear":
                linear_weight = torch.stack([mp.transform.W_r, mp.transform.W_i, mp.transform.W_j, mp.transform.W_k],
                                            dim=0).to(device)
                quaternion_weights.append(linear_weight)
        elif undirectional:
            if mp.transform.transform.__class__.__name__ == "QMLP":
                linear1_weight = torch.stack([mp.transform.transform.qlinear1.W_r, mp.transform.transform.qlinear1.W_i,
                                              mp.transform.transform.qlinear1.W_j, mp.transform.transform.qlinear1.W_k],
                                             dim=0).to(device)
                linear2_weight = torch.stack([mp.transform.transform.qlinear2.W_r, mp.transform.transform.qlinear2.W_i,
                                              mp.transform.transform.qlinear2.W_j, mp.transform.transform.qlinear2.W_k],
                                             dim=0).to(device)
                quaternion_weights.append(linear1_weight)
                quaternion_weights.append(linear2_weight)
            elif mp.transform.transform.__class__.__name__ == "QLinear":
                linear_weight = torch.stack([mp.transform.transform.W_r, mp.transform.transform.W_i,
                                             mp.transform.transform.W_j, mp.transform.transform.W_k],
                                            dim=0).to(device)
                quaternion_weights.append(linear_weight)
        else:
            raise AttributeError


    # pooling
    if model.pooling.__class__.__name__ == "QuaternionSoftAttentionPooling":
        stacked_weights = torch.stack([model.pooling.linear.W_r, model.pooling.linear.W_i,
                                       model.pooling.linear.W_k, model.pooling.linear.W_k],
                                      dim=0).to(device)

        quaternion_weights.append(stacked_weights)

    # downstream network
    for linear in model.downstream.affine:
        stacked_weights = torch.stack([linear.W_r, linear.W_i, linear.W_j, linear.W_k], dim=0).to(device)
        quaternion_weights.append(stacked_weights)


    # quaternion weight regularisation for Message passing, and optionally Pooling and Downstream Network
    for weight in quaternion_weights:
        reg_loss += weight.norm(p=p, dim=0).mean()

    return reg_loss

