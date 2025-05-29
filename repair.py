import torch
from torch import nn
from torch.nn import MSELoss
import yaml
import numpy as np
import os
from tqdm import trange
import torch_geometric
from einops import rearrange
from src.data import dataset
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN_equiv
from src.utils.loss import select_criterion
from src.utils.optim import setup_optimization
from src.utils.helpers import overwrite_conf, count_parameters, set_seed, mask_input, mask_hidden

import wandb
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def main(args=None):

    # read config file
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))

    # only for sweeps
    torch.set_float32_matmul_precision('high')
    print(yaml.dump(conf, default_flow_style=False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if conf["wandb"]:
        wandb.init(config=conf, **conf["wandb_args"])

    set_seed(conf['train_args']['seed'])

    # =============================================================================================
    #   SETUP DATASET AND DATALOADER
    # =============================================================================================
    print("=============================================================================================")
    print("\tSETUP DATASET AND DATALOADER")
    print("=============================================================================================")
    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)
    train_set = dataset(conf['data'],
                        split='train',
                        debug=conf["debug"],
                        direction=conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)
    conf['scalegmn_args']["layer_layout"] = train_set.get_layer_layout()

    val_set = dataset(conf['data'],
                      split='val',
                      debug=conf["debug"],
                      direction=conf['scalegmn_args']['direction'],
                      equiv_on_hidden=equiv_on_hidden,
                      get_first_layer_mask=get_first_layer_mask)

    test_set = dataset(conf['data'],
                       split='test',
                       debug=conf["debug"],
                       direction=conf['scalegmn_args']['direction'],
                       equiv_on_hidden=equiv_on_hidden,
                       get_first_layer_mask=get_first_layer_mask)

    print(f'Len train set: {len(train_set)}')
    print(f'Len val set: {len(val_set)}')
    print(f'Len test set: {len(test_set)}')



    train_loader = torch_geometric.loader.DataLoader(
        dataset=train_set,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=True,
        sampler=None,
    )
    val_loader = torch_geometric.loader.DataLoader(
        dataset=val_set,
        batch_size=conf["batch_size"],
        shuffle=False,
    )
    test_loader = torch_geometric.loader.DataLoader(
        dataset=test_set,
        batch_size=conf["batch_size"],
        shuffle=True,
        num_workers=conf["num_workers"],
        pin_memory=True,
    )

    # =============================================================================================
    #   DEFINE MODEL
    # =============================================================================================
    print("=============================================================================================")
    print("\tDEFINE MODEL")
    print("=============================================================================================")
    net = ScaleGMN_equiv(conf['scalegmn_args'])

    print(net)
    cnt_p = count_parameters(net=net)
    if conf["wandb"]:
        wandb.log({'number of parameters': cnt_p}, step=0)

    for p in net.parameters():
        p.requires_grad = True

    net = net.to(device)

    # =============================================================================================
    #   DEFINE LOSS
    # =============================================================================================
    print("=============================================================================================")
    print("\tDEFINE LOSS")
    print("=============================================================================================")
    criterion = select_criterion(conf['train_args']['loss'], {})

    # =============================================================================================
    #   DEFINE OPTIMIZATION
    # =============================================================================================
    print("=============================================================================================")
    print("\tDEFINE OPTIMIZATION")
    print("=============================================================================================")
    conf_opt = conf['optimization']
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(model_params, optimizer_name=conf_opt['optimizer_name'],
                                              optimizer_args=conf_opt['optimizer_args'],
                                              scheduler_args=conf_opt['scheduler_args'])

    best_val_loss = float("inf")
    best_test_results, best_val_results = None, None
    test_loss = -1.0
    global_step = 0
    start_epoch = 0

    epoch_iter = trange(start_epoch, conf['train_args']['num_epochs'])
    net.train()
    optimizer.zero_grad()

    for epoch in epoch_iter:
        for i, (poisoned_batch, healthy_batch) in enumerate(train_loader):
            # Move graph‐structure tensors to device
            poisoned_batch = poisoned_batch.to(device)
            healthy_batch  = healthy_batch.to(device)

            # Manually move your raw weight/bias lists to device
            weights_p = [w.to(device) for w in poisoned_batch.weights]
            biases_p  = [b.to(device) for b in poisoned_batch.biases]
            weights_h = [w.to(device) for w in healthy_batch.weights]
            biases_h  = [b.to(device) for b in healthy_batch.biases]

            optimizer.zero_grad()

            # Predict and apply deltas to the poisoned model
            # delta_w, delta_b = net(poisoned_batch, weights_h, biases_h)
            # new_w, new_b     = residual_param_update(weights_p, biases_p, delta_w, delta_b)
            # print(f"weights_p: {[w.shape for w in weights_p]}, biases_p: {[b.shape for b in biases_p]}")

            delta_w, delta_b = net(poisoned_batch, weights_p, biases_p)
            # print(f"delta_w: {[w.shape for w in delta_w]}, delta_b: {[b.shape for b in delta_b]}")
            delta_w = [w.squeeze(-1).repeat(layer_size, 1, 1) for w, layer_size in zip(delta_w, poisoned_batch.layer_layout[0][:-1])]
            delta_b = [b.squeeze(-1).repeat(layer_size, 1) for b, layer_size in zip(delta_b, poisoned_batch.layer_layout[0][1:])]
            # print(f"delta_w: {[w.shape for w in delta_w]}, delta_b: {[b.shape for b in delta_b]}")

            new_w = [weights_p[j] * delta_w[j] for j in range(len(weights_p))]
            new_b = [biases_p[j] * delta_b[j] for j in range(len(biases_p))]

            # Compute MSE against the healthy weights
            loss = 0.0
            for nw, hw in zip(new_w, weights_h):
                loss += criterion(nw, hw)
            for nb, hb in zip(new_b, biases_h):
                loss += criterion(nb, hb)
            loss = loss / (len(new_w) + len(new_b))

            log = {"batch_loss": loss.item()}

            loss.backward()
            if conf['optimization']['clip_grad']:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    net.parameters(),
                    conf['optimization']['clip_grad_max_norm']
                )
                log = {"train/loss": loss.item(), "grad_norm": grad_norm}
            else:
                log = {"train/loss": loss.item()}

            optimizer.step()
            if scheduler[1] is not None and scheduler[1] != 'ReduceLROnPlateau':
                log["lr"] = scheduler[0].get_last_lr()[0]
                scheduler[0].step()

            if conf["wandb"]:
                log["global_step"] = global_step
                wandb.log(log, step=global_step)

            epoch_iter.set_description(
                f"[{epoch} {i+1}], train loss: {loss.item():.3f}"
            )
            global_step += 1

            if (global_step) % conf['train_args']['eval_every'] == 0:
                val_loss_dict = evaluate(net, val_loader, device)
                test_loss_dict = evaluate(net, test_loader, device)

                val_loss = val_loss_dict["avg_loss"]
                test_loss = test_loss_dict["avg_loss"]
                train_loss_dict = evaluate(net, train_loader, device, num_batches=100)

                if val_loss < best_val_loss:
                    best_val_loss   = val_loss
                    best_val_results = val_loss_dict
                    best_test_results = test_loss_dict

                if conf["wandb"]:
                    wandb.log({
                        "train/avg_loss": train_loss_dict["avg_loss"],
                        "val/best_loss": best_val_results["avg_loss"],
                        "test/best_loss": best_test_results["avg_loss"],
                        **{f"val/{k}": v for k, v in val_loss_dict.items()},
                        **{f"test/{k}": v for k, v in test_loss_dict.items()},
                        "epoch": epoch,
                        "global_step": global_step,
                    })


@torch.no_grad()
def evaluate(model, loader, device, num_batches=None):
    """
    Compute average parameter-MSE between the hypernetwork's output
    (poisoned + delta → “cleansed”) and the true healthy weights.
    """
    model.eval()
    losses = []
    for i, (poisoned_batch, healthy_batch) in enumerate(loader):
        if num_batches is not None and i >= num_batches:
            break

        # 1) Move graph‐structure tensors to device
        poisoned_batch = poisoned_batch.to(device)
        healthy_batch  = healthy_batch.to(device)

        # 2) Move raw weight/bias lists to device
        w_p = [w.to(device) for w in poisoned_batch.weights]
        b_p = [b.to(device) for b in poisoned_batch.biases]
        w_h = [w.to(device) for w in healthy_batch.weights]
        b_h = [b.to(device) for b in healthy_batch.biases]

        # 3) Forward through the hypernetwork
        delta_w, delta_b = model(poisoned_batch, w_p, b_p)
        # print(f"delta_w: {[w.shape for w in delta_w]}, delta_b: {[b.shape for b in delta_b]}")
        delta_w = [w.squeeze(-1).repeat(layer_size, 1, 1) for w, layer_size in zip(delta_w, poisoned_batch.layer_layout[0][:-1])]
        delta_b = [b.squeeze(-1).repeat(layer_size, 1) for b, layer_size in zip(delta_b, poisoned_batch.layer_layout[0][1:])]
        # print(f"delta_w: {[w.shape for w in delta_w]}, delta_b: {[b.shape for b in delta_b]}")

        new_w = [w_p[j] * delta_w[j] for j in range(len(w_p))]
        new_b = [b_p[j] * delta_b[j] for j in range(len(b_p))]

        # 4) Compute parameter-space MSE against the healthy weights
        batch_loss = 0.0
        for nw, hw in zip(new_w, w_h):
            batch_loss += nn.MSELoss()(nw, hw)
        for nb, hb in zip(new_b, b_h):
            batch_loss += nn.MSELoss()(nb, hb)
        batch_loss = batch_loss / (len(new_w) + len(new_b))
        log = {"batch_loss": batch_loss.item()}
        losses.append(batch_loss.cpu())

    avg_loss = torch.stack(losses).mean().item()
    log = {"avg_loss": avg_loss}
    wandb.log(log)
    model.train()
    return {"avg_loss": avg_loss}


def residual_param_update(weights, biases, delta_weights, delta_biases):
    """
    Apply the delta weights and biases to the original weights and biases.
    """
    # print(len(weights), len(biases), len(delta_weights), len(delta_biases))
    # print(weights[0].shape, biases[0].shape, delta_weights[0].shape, delta_biases[0].shape)
    new_weights = [weights[j] + delta_weights[j] for j in range(len(weights))]
    new_biases = [biases[j] + delta_biases[j] for j in range(len(weights))]
    return new_weights, new_biases

if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)