import torch


def multiplication_rule_regularization(model, p: int = 1, device=None):
    reg = 0.0
    for name, module in model.named_modules():
        if hasattr(module, "phm_rule"):
            m = getattr(module, "phm_rule")
            if m is not None:
                phm_rule = m
                reg += phm_rule.norm(p=p).mean()
    return reg


def phm_weight_regularization(model, p: int = 2, device=None):
    reg = 0.0
    for name, module in model.named_modules():
        if hasattr(module, "W"):
            m = getattr(module, "W")
            if m is not None:
                weight = m
                reg += weight.norm(p=p, dim=0).mean()
    return reg