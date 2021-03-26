import logging
import argparse
import json
import pickle
from datetime import datetime
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import DataLoader
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.utils import degree

from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

import os

from benchmarks.utils import set_logging, set_seed_all, str2bool

from phc.quaternion.regularization import quaternion_weight_regularization, get_model_blocks

# quaternion undirectional models
from phc.quaternion.undirectional.models import QuaternionSkipConnectAdd as UQ_SC_ADD
from phc.quaternion.undirectional.models import QuaternionSkipConnectConcat as UQ_SC_CAT

# PHM undirectional models
from phc.hypercomplex.undirectional.models import PHMSkipConnectAdd as UPH_SC_ADD
from phc.hypercomplex.undirectional.models import PHMSkipConnectConcat as UPH_SC_CAT

from phc.hypercomplex.regularization import phm_weight_regularization, multiplication_rule_regularization
from phc.hypercomplex.utils import get_multiplication_matrices


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_parser():
    parser = argparse.ArgumentParser(description='Training script for the Hypercomplex MPNN on PCBA data')

    parser.add_argument('--device', action='store', dest='device', default=0, type=int,
                        help="Which gpu device to use. Defaults to 0")
    parser.add_argument('--nworkers', action='store', dest='nworkers', default=0, type=int,
                        help="How many workers for the dataloader should be used. Defaults to 0")
    parser.add_argument('--pin_memory', action='store', dest='pin_memory', default="True", type=str,
                        help="If `pin_memory` should be set to True for the pytorch dataloader. Defaults to True.")
    parser.add_argument('--batch_size', action='store', dest='batch_size', default=512, type=int,
                        help="Batch size for training. Defaults to 512.")
    parser.add_argument('--save_dir', action='store', dest='save_dir', default="pcba/`save_dir`", type=str,
                        help="Where the results should be saved. Defaults to pcba/`save_dir`")
    parser.add_argument('--n_runs', action='store', dest='n_runs', default=5, type=int,
                        help="How many runs should be done. Defaults to 5.")
    parser.add_argument('--seed', action='store', dest='seed', default=0, type=int,
                        help="Seed for reproducibility. Defaults to 0.")
    parser.add_argument('--pooling', dest="pooling", action='store', type=str, default="softattention",
                        help="Pooling strategy of node embeddings. Defaults to 'softattention'",
                        choices=["globalsum", "softattention"])
    parser.add_argument('--type', action='store', type=str, default="undirectional-phm-sc-add",
                        help="Which model to use. Defaults to 'undirectional-phm-sc-add'.",
                        choices=["undirectional-quaternion-sc-cat", "undirectional-quaternion-sc-add",
                                 "undirectional-phm-sc-cat", "undirectional-phm-sc-add"])
    parser.add_argument('--phm_dim', action='store', type=int, default=2,
                        help="Dimension of hypercomplex components. Defaults to 2.")
    parser.add_argument('--learn_phm', action='store', type=str, default="True",
                        help="If the multiplication rule should be learned per affine layer. Defaults to True")
    parser.add_argument('--unique_phm', action='store', type=str, default="False",
                        help="If the same multiplication rule should be (learned and) used through the whole network."
                             "Defaults to 'False'.")

    parser.add_argument('--w_init', action='store', type=str, default="phm",
                        help="Which init to use for the weight matrices. Defaults to 'phm'",
                        choices=["quaternion", "glorot-normal", "glorot-uniform", "orthogonal", "phm"])
    parser.add_argument('--c_init', action='store', type=str, default="standard",
                        help="Which init to use for the contribution matrices. Defaults to 'standard'",
                        choices=["standard", "random"])


    parser.add_argument('--input_embed_dim', action='store', type=int, default=512,
                        help="(Embedding) Dimension for the input features. Defaults to 512")
    parser.add_argument('--embed_combine', type=str, default="sum",
                        help="Combine type for Atom-Encoder embedding. Defaults to sum.")
    parser.add_argument('--full_encoder', type=str, default="True",
                        help="If the atom- and bondencoder should initialise all hypercomplex components. "
                             "Defaults to True.")
    parser.add_argument('--mp_units', help='Number of hidden units for each message passing layer.'
                                           'Defaults to "512,512,512,512,512,512,512"',
                        type=str, default="512,512,512,512,512,512,512")
    parser.add_argument('--mp_norm', help='Which normalization to use for the message passing layer.'
                                          'Defaults to naive-batch-norm.',
                        type=str, default="naive-batch-norm", choices=["None", "naive-batch-norm", "q-batch-norm",
                                                                       "naive-naive-batch-norm"])
    parser.add_argument('--mlp_mp', type=str, default="False",
                        help="If the message passing layers should contain a 2 -layer MLP for trafo. Otherwise only "
                             "one linear layer is used. Defaults to False.")
    parser.add_argument('--dropout_mpnn', help='Dropouts for message passing layers.'
                                               'Defaults to "0.1,0.1,0.1,0.1,0.1,0.1,0.1"',
                        type=str, default="0.1,0.1,0.1,0.1,0.1,0.1,0.1")
    parser.add_argument('--same_dropout', help='If the dropout mask should be the same for hypercomplex constituents'
                                               'Defaults to "False" ', type=str, default="False")
    parser.add_argument('--bias', help='Whether or not the affine transformation should include bias.'
                                       'Defaults to "True"', type=str, default="True")
    parser.add_argument('--d_units', help='Number of hidden units in the Downstream Network.'
                                          'Defaults to "768,256" ', type=str, default="768,256")
    parser.add_argument('--d_bn', help='Which batch-norm should be used for the downstream network'
                                       'Defaults to "naive-batch-norm"', type=str, default="naive-batch-norm",
                        choices=["None", "naive-batch-norm", "q-batch-norm",
                                 "naive-naive-batch-norm"])
    parser.add_argument('--dropout_dn', type=str, default="0.3,0.2",
                        help="Dropout rate for the downstream network. Defaults to '0.3,0.2' .")
    parser.add_argument('--activation', action='store', dest='activation', default="relu", type=str,
                        help="Which activation function to use Defauls to 'relu'",
                        choices=["relu", "lrelu", "elu", "selu", "swish"])
    parser.add_argument('--aggr_msg', help='Which scheme to use for aggregating the messages m_vw. Defaults to sum.',
                        type=str, default="sum", choices=["sum", "mean", "min", "max", "softmax", "pna"])
    parser.add_argument('--aggr_node', help='Which scheme to use for aggregating the node embeddings. Defaults to sum.',
                        type=str, default="sum", choices=["sum", "mean", "min", "max", "softmax", "pna"])
    parser.add_argument('--msg_scale', help='(Directional) message scaling.'
                                            'Only applicable if PNA is set in `aggr_msg`.'
                                            'Defaults to False',
                        type=str, default="False")
    parser.add_argument('--real_trafo',
                        help='Which scheme to use for transforming quaternion embeddings to euclidean.'
                             'Defaults to linear',
                        type=str, default="linear", choices=["linear", "sum"])
    parser.add_argument('--epochs', type=int, default=150,
                        help="Number of epochs to train. Defaults to 150.")
    parser.add_argument('--lr', action='store', type=float, default=0.0005,
                        help="Learning rate for AdamW Optimizer. Defaults to 0.0005")
    parser.add_argument('--patience', type=int, default=5,
                        help="Patience for learning rate scheduler. Defaults to 5.")
    parser.add_argument('--factor', type=float, default=0.75,
                        help="Multiplicative learning rate factor. Defaults to 0.75.")
    parser.add_argument('--weightdecay', action='store', type=float, default=0.0001,
                        help="weight decay for weight matrices in PHM-Layers. Defaults to 0.0001 ")
    parser.add_argument('--regularization', action='store', type=int, default=2,
                        help="Which regularization norm should be used. Choices are [1, 2]. Defaults to 2.")
    parser.add_argument('--weightdecay2', action='store', type=float, default=0.0,
                        help="L1-regularization for multiplication rule matrices in PHM-Layers. Defaults to 0.0 ")
    parser.add_argument('--grad_clipping', action='store', type=float, default=2.0,
                        help="Gradient clipping. Defaults to max-norm 2.0 - If no gradient clipping should be used,"
                             "insert 0.0")

    parser.add_argument('--log_weights', action='store', type=str, default="False",
                        help="If the weights should be logged on tensorboard. Defaults to False")
    parser.add_argument('--msg_encoder', action='store', type=str, default="identity",
                        choices=["identity", "relu", "lrelu", "elu", "swish"],
                        help="If the weights should be logged on tensorboard. Defaults to 'identity'.")

    parser.add_argument('--sc_type', action="store", type=str, default="first",
                        choices=["first", "last"],
                        help="How to apply skip-connections for the ADD model. Choices are:"
                             "['first', 'last']. Defaults to 'first'.")

    args = parser.parse_args()
    return args

PRE_TRAFO = False


def train(epoch, model, device, transform, loader, optimizer, evaluator, kwargs):
    model = model.train()
    total_loss = 0.0
    y_true_list = list()
    y_pred_list = list()
    for i, data in enumerate(loader):
        if not PRE_TRAFO:
            data = transform(data)
        data = data.to(device)
        data.y = data.y.to(torch.float)
        mask = ~torch.isnan(data.y)
        optimizer.zero_grad()
        logits = model(data)
        try:
            loss = F.binary_cross_entropy_with_logits(input=logits[mask], target=data.y[mask])
            reg_loss = torch.tensor(0.0, device=device)
            if kwargs["weight_decay"] > 0.0:
                if not hasattr(model, "phm_dim"):
                    reg_loss += kwargs["lr"] * kwargs["weight_decay"] * quaternion_weight_regularization(model,
                                                                                                         device=device,
                                                                                                         p=kwargs["p"]
                                                                                                         )
                else:
                    reg_loss += kwargs["lr"] * kwargs["weight_decay"] * phm_weight_regularization(model,
                                                                                                  device=device,
                                                                                                  p=kwargs["p"]
                                                                                                  )
            if kwargs["weight_decay2"] > 0.0:
                reg_loss += kwargs["lr"] * kwargs["weight_decay2"] * multiplication_rule_regularization(model,
                                                                                                        p=1)

            loss += reg_loss.squeeze()

            y_true_list.append(data.y)
            y_pred_list.append(logits)
            loss.backward()
            if kwargs["grad_clipping"] > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=kwargs["grad_clipping"], norm_type=2)

            optimizer.step()
            total_loss += loss.item() * data.num_graphs
        except IndexError:
            logging.info(f"index error at step {i}")
            continue
        if i % 200 == 0 and i != 0:
            logging.info(f"Run {kwargs['run']}/{kwargs['nruns']} Epoch: {epoch}, Step: {i}/{len(loader)}, "
                         f"Loss: {np.round(total_loss / i, 4)}.")

    y_true_list = torch.cat(y_true_list, dim=0).cpu()
    y_pred_list = torch.cat(y_pred_list, dim=0).cpu()
    evaluation_res = evaluator.eval({
        'y_pred': y_pred_list,
        'y_true': y_true_list,
    })[evaluator.eval_metric]
    total_loss /= len(loader.dataset)
    metric_key = evaluator.eval_metric
    metrics = {"loss": total_loss, metric_key: evaluation_res}
    return metrics


@torch.no_grad()
def test_validate(model, device, transform, loader, evaluator):
    """
    Validates/Tests the model.
    Calls customized `calculate_metrics` function but also uses OGB evaluator
    """
    model = model.eval()
    total_loss = 0.0
    y_true_list = [None] * len(loader)
    y_pred_list = [None] * len(loader)
    for i, data in enumerate(loader):
        data = data.to(device)
        if not PRE_TRAFO:
            data = transform(data)
        mask = ~torch.isnan(data.y)
        logits = model(data)
        data.y = data.y.to(torch.float)
        loss = F.binary_cross_entropy_with_logits(input=logits[mask], target=data.y[mask])
        total_loss += loss.item() * data.num_graphs
        y_true_list[i] = data.y
        y_pred_list[i] = logits

    y_true_list = torch.cat(y_true_list, dim=0).cpu()
    y_pred_list = torch.cat(y_pred_list, dim=0).cpu()
    evaluation_res = evaluator.eval({
        'y_pred': y_pred_list,
        'y_true': y_true_list,
    })[evaluator.eval_metric]

    total_loss /= len(loader.dataset)
    metric_key = evaluator.eval_metric
    metrics = {"loss": total_loss, metric_key: evaluation_res}
    return metrics


def do_run(i, model, args, transform, train_loader, valid_loader, test_loader, device, evaluator):

    logging.info(f"Run {i}/{args.n_runs}, seed: {args.seed + i - 1}")
    logging.info("Reset model parameters")
    set_seed_all(args.seed + i - 1)
    model.reset_parameters()
    model = model.to(device)

    # setting up parameter groups for optimization
    # quaternion/hypercomplex - in general
    params_mp = get_model_blocks(model, attr="convs", **dict(lr=args.lr, weight_decay=0.0))
    params_pooling = get_model_blocks(model, attr="pooling", **dict(lr=args.lr, weight_decay=0.0))
    params_downstream = get_model_blocks(model, attr="downstream", **dict(lr=args.lr, weight_decay=0.0))
    # quaternion/hypercomplex - undirectional
    params_norms = get_model_blocks(model, attr="norms", **dict(lr=args.lr, weight_decay=0.0))
    params_embedding_atom = get_model_blocks(model, attr="atomencoder", **dict(lr=args.lr, weight_decay=0.0))
    params_embedding_bonds = get_model_blocks(model, attr="bondencoders", **dict(lr=args.lr, weight_decay=0.0))

    params = params_mp + params_pooling + params_downstream + \
             params_norms + params_embedding_atom + params_embedding_bonds

    #  check if all params are captured
    total_params_splitted = sum([sum([p.numel() for p in pl["params"] if p.requires_grad]) for pl in params])
    total_params_model = sum([p.numel() for p in model.parameters() if p.requires_grad])
    assert total_params_model == total_params_splitted, f"splitted total params: {total_params_splitted}." \
                                                        f"However, total params of model are: {total_params_model}"
    #
    optimizer = torch.optim.Adam(params, weight_decay=0.0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor,
                                                           patience=args.patience, mode="max",
                                                           min_lr=1e-6, verbose=True)


    save_dir = os.path.join(args.save_dir,  f"run_{i}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # tensorboard logging
    logging.info(f"Creating new tensorboard logging for run {i}.")
    writer = SummaryWriter(log_dir=save_dir)

    lr_arr = []
    train_metrics_arr = []
    val_metrics_arr = []
    start_time = datetime.now()
    best_metric = float("-inf")
    model_save_dir = os.path.join(save_dir, "model.pt")
    model_last_save_dir = os.path.join(save_dir, "model_last.pt")
    metric_key = evaluator.eval_metric

    # At any point you can hit Ctrl + C to break out of training early.
    exit_script = "n"
    try:
        for epoch in range(1, args.epochs + 1):

            if str2bool(args.log_weights):
                # weights logging after one epoch
                # -> might need to move this into the train function to track after batch-updates
                for name, value in model.named_parameters():
                    if value.requires_grad:
                        writer.add_histogram(tag=name, values=value.data.cpu().numpy(),
                                             global_step=epoch)
                        # log only the parameter weights.
                        # if the grads should be logged too, need to move that code into the train_step function and retrieve
                        # gradients after loss.backward() is called. Otherwise the grads are None or 0.0

            lr = scheduler.optimizer.param_groups[0]['lr']
            lr_arr.append(lr)
            logging.info(f"Epoch: {epoch}/{args.epochs}, Learning Rate: {lr}.")
            train_metrics = train(epoch=epoch, model=model, device=device, transform=transform,
                                  loader=train_loader, optimizer=optimizer, evaluator=evaluator,
                                  kwargs={"run": i, "nruns": args.n_runs, "grad_clipping": args.grad_clipping,
                                          "lr": lr, "weight_decay": args.weightdecay,
                                          "weight_decay2": args.weightdecay2,
                                          "p": args.regularization})
            train_metrics_arr.append(train_metrics)
            logging.info(f"Training Metrics in Epoch: {epoch} \n Metrics: {train_metrics}.")

            validation_metrics = test_validate(model=model, device=device, transform=transform,
                                               loader=valid_loader, evaluator=evaluator)
            val_metrics_arr.append(validation_metrics)
            logging.info(f"Validation Metrics in Epoch: {epoch} \n Metrics: {validation_metrics}.")
            if validation_metrics[metric_key] > best_metric:
                logging.info(f"Saving model with validation '{metric_key}': {validation_metrics[metric_key]}.")
                best_metric = validation_metrics[metric_key]

                torch.save(model, model_save_dir)

            scheduler.step(validation_metrics[metric_key])

            # tensorboard logging
            writer.add_scalar(tag="lr", scalar_value=lr, global_step=epoch)
            writer.add_scalar(tag="train_loss", scalar_value=train_metrics["loss"], global_step=epoch)
            writer.add_scalar(tag=f"train_{metric_key}", scalar_value=train_metrics[metric_key], global_step=epoch)
            writer.add_scalar(tag="valid_loss", scalar_value=validation_metrics["loss"], global_step=epoch)
            writer.add_scalar(tag=f"valid_{metric_key}", scalar_value=validation_metrics[metric_key], global_step=epoch)
    except KeyboardInterrupt:
        logging.info("-" * 80)
        logging.info(f"training interupted in epoch {epoch}")
        logging.info(f"saving model at {model_last_save_dir}")
        exit_script = input("Should the entire run be exited after evaluation ? Type (Y/N) ")

    torch.save(model, model_last_save_dir)

    end_time = datetime.now()
    training_time = end_time - start_time
    logging.info(f"Training time: {training_time}")

    # after training, load the best model and test
    #if exit_script.lower() == "y":
    try:
        model = torch.load(model_save_dir)
    except FileNotFoundError:
        logging.info(f"File '{model_save_dir}' not found. Cannot load model. Will use current model.")

    model = model.to(device)
    logging.info(f"Testing model from best validation epoch")
    test_metrics = test_validate(model=model, loader=test_loader, device=device,
                                 transform=transform, evaluator=evaluator)
    logging.info(f"Test Metrics of model: \n Metrics: {test_metrics}.")

    # we also load the last model and test, just to see how it performs
    model = torch.load(model_last_save_dir)
    model = model.to(device)
    logging.info(f"Testing model from last epoch")
    test_metrics_last = test_validate(model=model, loader=test_loader, device=device,
                                      transform=transform, evaluator=evaluator)
    logging.info(f"Test Metrics of model: \n Metrics: {test_metrics_last}.")

    svz = {'train_metrics': train_metrics_arr, 'lr': lr_arr,
           'val_metrics': val_metrics_arr, 'test_metrics': test_metrics,
           'test_metrics_lastepoch': test_metrics_last}

    with open(os.path.join(save_dir, "arrays.pickle"), "wb") as fp:
        pickle.dump(svz, fp)

    svz2 = {"best_val": best_metric,
            "test_best_valEpoch": test_metrics[metric_key],
            "test_lastEpoch": test_metrics_last[metric_key]}

    with open(os.path.join(save_dir, "val_test.json"), 'w') as f:
        json.dump(svz2, f)

    # close tensorboard writer for run
    writer.close()

    if exit_script.lower() == "y":
        logging.info("exit script.")
        exit()

    return test_metrics[metric_key], test_metrics_last[metric_key], best_metric


def main():
    args = get_parser()
    # get some argparse arguments that are parsed a bool string
    naive_encoder = not str2bool(args.full_encoder)
    pin_memory = str2bool(args.pin_memory)
    use_bias = str2bool(args.bias)
    downstream_bn = str(args.d_bn)
    same_dropout = str2bool(args.same_dropout)
    mlp_mp = str2bool(args.mlp_mp)

    phm_dim = args.phm_dim
    learn_phm = str2bool(args.learn_phm)


    base_dir = "pcba/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if base_dir not in args.save_dir:
        args.save_dir = os.path.join(base_dir, args.save_dir)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    set_logging(save_dir=args.save_dir)
    logging.info(f"Creating log directory at {args.save_dir}.")
    with open(os.path.join(args.save_dir, "params.json"), 'w') as fp:
        json.dump(args.__dict__, fp)

    mp_layers = [int(item) for item in args.mp_units.split(',')]
    downstream_layers = [int(item) for item in args.d_units.split(',')]
    mp_dropout = [float(item) for item in args.dropout_mpnn.split(',')]
    dn_dropout = [float(item) for item in args.dropout_dn.split(',')]
    logging.info(f'Initialising model with {mp_layers} hidden units with dropout {mp_dropout} '
                 f'and downstream units: {downstream_layers} with dropout {dn_dropout}.')

    if args.pooling == "globalsum":
        logging.info("Using GlobalSum Pooling")
    else:
        logging.info("Using SoftAttention Pooling")


    logging.info(f"Using Adam optimizer with weight_decay ({args.weightdecay}) and regularization "
                 f"norm ({args.regularization})")
    logging.info(f"Weight init: {args.w_init} \n Contribution init: {args.c_init}")

    # data
    dname = "ogbg-molpcba"
    transform = RemoveIsolatedNodes()
    # pre-transform doesnt work somehow..
    dataset = PygGraphPropPredDataset(name=dname, root="dataset")#, pre_transform=transform, transform=None)
    evaluator = Evaluator(name=dname)
    split_idx = dataset.get_idx_split()
    train_data = dataset[split_idx["train"]]
    valid_data = dataset[split_idx["valid"]]
    test_data = dataset[split_idx["test"]]


    if PRE_TRAFO:
        # pre-transform in memory to overcome computations when training
        logging.info("Pre-transforming graphs, to overcome computation in batching.")
        train_data_list = []
        valid_data_list = []
        test_data_list = []
        for data in train_data:
            train_data_list.append(transform(data))
        for data in valid_data:
            valid_data_list.append(transform(data))
        for data in test_data:
            test_data_list.append(transform(data))

        logging.info("finised. Initiliasing dataloaders")

        train_loader = DataLoader(train_data_list, batch_size=args.batch_size, drop_last=False,
                                  shuffle=True, num_workers=args.nworkers, pin_memory=pin_memory)
        valid_loader = DataLoader(valid_data_list, batch_size=args.batch_size, drop_last=False,
                                  shuffle=False, num_workers=args.nworkers, pin_memory=pin_memory)
        test_loader = DataLoader(test_data_list, batch_size=args.batch_size, drop_last=False,
                                 shuffle=False, num_workers=args.nworkers, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(train_data, batch_size=args.batch_size, drop_last=False,
                                  shuffle=True, num_workers=args.nworkers, pin_memory=pin_memory)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, drop_last=False,
                                  shuffle=False, num_workers=args.nworkers, pin_memory=pin_memory)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, drop_last=False,
                                 shuffle=False, num_workers=args.nworkers, pin_memory=pin_memory)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    FULL_ATOM_FEATURE_DIMS = get_atom_feature_dims()
    FULL_BOND_FEATURE_DIMS = get_bond_feature_dims()

    # for hypercomplex model
    unique_phm = str2bool(args.unique_phm)
    if unique_phm:
        phm_rule = get_multiplication_matrices(phm_dim=args.phm_dim, type="phm")
        phm_rule = torch.nn.ParameterList(
            [torch.nn.Parameter(a, requires_grad=learn_phm) for a in phm_rule]
        )
    else:
        phm_rule = None

    if args.aggr_msg == "pna" or args.aggr_node == "pna":
        # if PNA is used
        # Compute in-degree histogram over training data.
        deg = torch.zeros(6, dtype=torch.long)
        for data in dataset[split_idx['train']]:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
    else:
        deg = None

    aggr_kwargs = {"aggregators": ['mean', 'min', 'max', 'std'],
                   "scalers": ['identity', 'amplification', 'attenuation'],
                   "deg": deg,
                   "post_layers": 1,
                   "msg_scalers": str2bool(args.msg_scale),  # this key is for directional messagepassing layers.

                   "initial_beta": 1.0,  # Softmax
                   "learn_beta": True
                   }

    if "quaternion" in args.type:
        if args.aggr_msg == "pna" or args.aggr_msg == "pna":
            logging.info("PNA not implemented for quaternion models.")
            raise NotImplementedError

    if args.type == "undirectional-quaternion-sc-add":
        logging.info("Using Quaternion Undirectional MPNN with Skip Connection through Addition")
        model = UQ_SC_ADD(atom_input_dims=FULL_ATOM_FEATURE_DIMS,
                          atom_encoded_dim=args.input_embed_dim,
                          bond_input_dims=FULL_BOND_FEATURE_DIMS, naive_encoder=naive_encoder,
                          mp_layers=mp_layers, dropout_mpnn=mp_dropout,
                          init=args.w_init, same_dropout=same_dropout,
                          norm_mp=args.mp_norm, add_self_loops=True, msg_aggr=args.aggr_msg, node_aggr=args.aggr_node,
                          mlp=mlp_mp, pooling=args.pooling, activation=args.activation,
                          real_trafo=args.real_trafo, downstream_layers=downstream_layers, target_dim=dataset.num_tasks,
                          dropout_dn=dn_dropout, norm_dn=downstream_bn,
                          msg_encoder=args.msg_encoder,
                          **aggr_kwargs)
    elif args.type == "undirectional-quaternion-sc-cat":
        logging.info("Using Quaternion Undirectional MPNN with Skip Connection through Concatenation")
        model = UQ_SC_CAT(atom_input_dims=FULL_ATOM_FEATURE_DIMS,
                          atom_encoded_dim=args.input_embed_dim,
                          bond_input_dims=FULL_BOND_FEATURE_DIMS, naive_encoder=naive_encoder,
                          mp_layers=mp_layers, dropout_mpnn=mp_dropout,
                          init=args.w_init, same_dropout=same_dropout,
                          norm_mp=args.mp_norm, add_self_loops=True, msg_aggr=args.aggr_msg, node_aggr=args.aggr_node,
                          mlp=mlp_mp, pooling=args.pooling, activation=args.activation,
                          real_trafo=args.real_trafo, downstream_layers=downstream_layers, target_dim=dataset.num_tasks,
                          dropout_dn=dn_dropout, norm_dn=downstream_bn,
                          msg_encoder=args.msg_encoder,
                          **aggr_kwargs)
    elif args.type == "undirectional-phm-sc-add":
        logging.info("Using PHM Undirectional MPNN with Skip Connection through Addition")
        model = UPH_SC_ADD(phm_dim=phm_dim, learn_phm=learn_phm, phm_rule=phm_rule,
                           atom_input_dims=FULL_ATOM_FEATURE_DIMS,
                           atom_encoded_dim=args.input_embed_dim,
                           bond_input_dims=FULL_BOND_FEATURE_DIMS, naive_encoder=naive_encoder,
                           mp_layers=mp_layers, dropout_mpnn=mp_dropout,
                           w_init=args.w_init, c_init=args.c_init,
                           same_dropout=same_dropout,
                           norm_mp=args.mp_norm, add_self_loops=True, msg_aggr=args.aggr_msg, node_aggr=args.aggr_node,
                           mlp=mlp_mp, pooling=args.pooling, activation=args.activation,
                           real_trafo=args.real_trafo, downstream_layers=downstream_layers,
                           target_dim=dataset.num_tasks,
                           dropout_dn=dn_dropout, norm_dn=downstream_bn,
                           msg_encoder=args.msg_encoder,
                           sc_type=args.sc_type,
                           **aggr_kwargs)

    elif args.type == "undirectional-phm-sc-cat":
        logging.info("Using PHM Undirectional MPNN with Skip Connection through Concatenation")
        model = UPH_SC_CAT(phm_dim=phm_dim, learn_phm=learn_phm, phm_rule=phm_rule,
                           atom_input_dims=FULL_ATOM_FEATURE_DIMS,
                           atom_encoded_dim=args.input_embed_dim,
                           bond_input_dims=FULL_BOND_FEATURE_DIMS, naive_encoder=naive_encoder,
                           mp_layers=mp_layers, dropout_mpnn=mp_dropout,
                           w_init=args.w_init, c_init=args.c_init,
                           same_dropout=same_dropout,
                           norm_mp=args.mp_norm, add_self_loops=True, msg_aggr=args.aggr_msg, node_aggr=args.aggr_node,
                           mlp=mlp_mp, pooling=args.pooling, activation=args.activation,
                           real_trafo=args.real_trafo, downstream_layers=downstream_layers,
                           target_dim=dataset.num_tasks,
                           dropout_dn=dn_dropout, norm_dn=downstream_bn,
                           msg_encoder=args.msg_encoder,
                           **aggr_kwargs)

    else:
        raise ModuleNotFoundError


    logging.info(f"Model consists of {model.get_number_of_params_()} trainable parameters")
    # do runs
    test_best_epoch_metrics_arr = []
    test_last_epoch_metrics_arr = []
    val_metrics_arr = []


    for i in range(1, args.n_runs + 1):
        ogb_bestEpoch_test_metrics, ogb_lastEpoch_test_metric, ogb_val_metrics = do_run(i, model, args,
                                                                                        transform,
                                                                                        train_loader, valid_loader,
                                                                                        test_loader, device, evaluator)

        test_best_epoch_metrics_arr.append(ogb_bestEpoch_test_metrics)
        test_last_epoch_metrics_arr.append(ogb_lastEpoch_test_metric)
        val_metrics_arr.append(ogb_val_metrics)


    logging.info(f"Performance of model across {args.n_runs} runs:")
    test_bestEpoch_perf = torch.tensor(test_best_epoch_metrics_arr)
    test_lastEpoch_perf = torch.tensor(test_last_epoch_metrics_arr)
    valid_perf = torch.tensor(val_metrics_arr)
    logging.info('===========================')
    logging.info(f'Final Test (best val-epoch) '
                 f'"{evaluator.eval_metric}": {test_bestEpoch_perf.mean():.4f} ± {test_bestEpoch_perf.std():.4f}')
    logging.info(f'Final Test (last-epoch) '
                 f'"{evaluator.eval_metric}": {test_lastEpoch_perf.mean():.4f} ± {test_lastEpoch_perf.std():.4f}')
    logging.info(f'Final (best) Valid "{evaluator.eval_metric}": {valid_perf.mean():.4f} ± {valid_perf.std():.4f}')


if __name__ == "__main__":
    torch.set_num_threads(6)
    main()