import logging
import os
import torch
import numpy as np
import argparse
from torch_geometric.utils import remove_isolated_nodes
import re


def set_logging(save_dir):
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "run.log")),
            logging.StreamHandler()
        ]
    )


def set_seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



class CustomRemoveIsolatedNodes(object):
    def __call__(self, data):
        num_nodes = data.num_nodes
        out = remove_isolated_nodes(data.edge_index, data.edge_attr, num_nodes)
        data.edge_index, data.edge_attr, mask = out

        for key, item in data:
            if torch.is_tensor(item) and item.size(0) == num_nodes and "edge" not in key:
                data[key] = item[mask]

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
