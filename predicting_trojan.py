import torch
import torch_geometric
import yaml
import torch.nn.functional as F
import os
from src.data import dataset
from tqdm import tqdm
from src.utils.setup_arg_parser import setup_arg_parser
from src.scalegmn.models import ScaleGMN
from src.utils.loss import select_criterion
from src.utils.optim import setup_optimization
from src.utils.helpers import overwrite_conf, count_parameters, assert_symms, set_seed, mask_input, mask_hidden, count_named_parameters
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score
import matplotlib.pyplot as plt
import wandb
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def main():

    # read config file
    conf = yaml.safe_load(open(args.conf))
    conf = overwrite_conf(conf, vars(args))

    assert_symms(conf)

    print(yaml.dump(conf, default_flow_style=False))
    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")
    if conf["wandb"]:
        wandb.init(config=conf, **conf["wandb_args"])

    set_seed(conf['train_args']['seed'])
    # =============================================================================================
    #   SETUP DATASET AND DATALOADER
    # =============================================================================================
    equiv_on_hidden = mask_hidden(conf)
    get_first_layer_mask = mask_input(conf)
    train_set = dataset(conf['data'],
                        split='train',
                        debug=conf["debug"],
                        direction=conf['scalegmn_args']['direction'],
                        equiv_on_hidden=equiv_on_hidden,
                        get_first_layer_mask=get_first_layer_mask)
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
        sampler=None
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
    conf['scalegmn_args']["layer_layout"] = train_set.get_layer_layout()
    # conf['scalegmn_args']['input_nn'] = 'conv'
    net = ScaleGMN(conf['scalegmn_args'])
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
    criterion = select_criterion(conf['train_args']['loss'], {})

    # =============================================================================================
    #   DEFINE OPTIMIZATION
    # =============================================================================================
    conf_opt = conf['optimization']
    model_params = [p for p in net.parameters() if p.requires_grad]
    optimizer, scheduler = setup_optimization(model_params, optimizer_name=conf_opt['optimizer_name'], optimizer_args=conf_opt['optimizer_args'], scheduler_args=conf_opt['scheduler_args'])
    # =============================================================================================
    # TRAINING LOOP
    # =============================================================================================
    step = 0
    best_val_auc = -float("inf")
    best_train_auc_TRAIN = -float("inf")
    best_test_results, best_val_results, best_train_results, best_train_results_TRAIN = None, None, None, None

    for epoch in range(conf['train_args']['num_epochs']):
        net.train()
        len_dataloader = len(train_loader)
        with tqdm(train_loader, desc="Training") as pbar:
            for i, batch in enumerate(pbar):
                step = epoch * len_dataloader + i
                batch = batch.to(device)
                gt_test_acc = batch.y.to(device)

                optimizer.zero_grad()
                inputs = batch.to(device)
                outputs = net(inputs)
                # pred_acc = F.sigmoid(net(inputs)).squeeze(-1)
                loss = criterion(outputs.squeeze(-1), gt_test_acc)
                loss.backward()
                pbar.set_description(f"Training | Loss = {loss.detach().cpu().item():.2f}")
                log = {}
                if conf['optimization']['clip_grad']:
                    log['grad_norm'] = torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                                    conf['optimization']['clip_grad_max_norm']).item()

                optimizer.step()

                if conf["wandb"]:
                    if step % 10 == 0:
                        probs = torch.sigmoid(outputs.squeeze(-1))    # convert to probabilities
                        log[f"train/{conf['train_args']['loss']}"] = loss.detach().cpu().item()
                        # log["train/rsq"] = r2_score(gt_test_acc.cpu().numpy(), probs.detach().cpu().numpy())
                        log["train/auc"] = roc_auc_score(gt_test_acc.detach().cpu().numpy(), probs.detach().cpu().numpy())

                    wandb.log(log, step=step)

                if scheduler[1] is not None and scheduler[1] != 'ReduceLROnPlateau':
                    scheduler[0].step()

        #############################################
        # VALIDATION
        #############################################
        if conf["validate"]:
            print(f"\nValidation after epoch {epoch}:")
            val_loss_dict = evaluate(net, val_loader, criterion, device)
            print(f"Epoch {epoch}, val L1 err: {val_loss_dict['avg_err']:.2f}, val loss: {val_loss_dict['avg_loss']:.2f}, val AUC: {val_loss_dict['auc']:.2f}")

            test_loss_dict = evaluate(net, test_loader, criterion, device)
            train_loss_dict = evaluate(net, train_loader, criterion, device)

            best_val_criteria = val_loss_dict['auc'] >= best_val_auc
            if best_val_criteria:
                best_val_auc = val_loss_dict['auc']
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict
                best_train_results = train_loss_dict

            best_train_criteria = train_loss_dict['auc'] >= best_train_auc_TRAIN
            if best_train_criteria:
                best_train_auc_TRAIN = train_loss_dict['auc']
                best_train_results_TRAIN = train_loss_dict

            if conf["wandb"]:
                plt.clf()
                plot = plt.scatter(val_loss_dict['actual'], val_loss_dict['pred'])
                plt.xlabel("Actual model accuracy")
                plt.ylabel("Predicted model accuracy")
                wandb.log({
                    "train/l1_err": train_loss_dict['avg_err'],
                    "train/loss": train_loss_dict['avg_loss'],
                    "train/auc": train_loss_dict['auc'],
                    "train/best_auc": best_train_results['auc'] if best_train_results is not None else None,
                    "train/best_auc_TRAIN_based": best_train_results_TRAIN['auc'] if best_train_results_TRAIN is not None else None,
                    "val/l1_err": val_loss_dict['avg_err'],
                    "val/loss": val_loss_dict['avg_loss'],
                    "val/scatter": wandb.Image(plot),
                    "val/auc": val_loss_dict['auc'],
                    "val/best_auc": best_val_results['auc'] if best_val_results is not None else None,
                    # test
                    "test/l1_err": test_loss_dict['avg_err'],
                    "test/loss": test_loss_dict['avg_loss'],
                    "test/auc": test_loss_dict['auc'],
                    "test/best_auc": best_test_results['auc'] if best_test_results is not None else None,
                    "epoch": epoch
                }, step=step)

            net.train()  # redundant


@torch.no_grad()
def evaluate(net, loader, loss_fn, device):
    net.eval()
    pred, actual = [], []
    err, losses = [], []
    for i, batch in enumerate(tqdm(loader, desc="Validating")):
        batch = batch.to(device)
        gt_test_acc: torch.Tensor = batch.y.to(device)
        inputs = batch.to(device)
        
        logits = net(inputs).squeeze(-1)  # logits output
        probs = torch.sigmoid(logits)    # convert to probabilities

        err.append(torch.abs(probs - gt_test_acc).mean().item())
        losses.append(loss_fn(logits, gt_test_acc).item())  # loss expects logits
        pred.append(probs.detach().cpu().numpy())
        actual.append(gt_test_acc.cpu().numpy())
        if i == 0:
            print(probs[0:10].detach().cpu().numpy())
            print(gt_test_acc[0:10].detach().cpu().numpy())

    avg_err, avg_loss = np.mean(err), np.mean(losses)
    actual, pred = np.concatenate(actual), np.concatenate(pred)
    try:
        auc = roc_auc_score(actual, pred)
    except ValueError:
        auc = float('nan')  # Happens if only one class is present in `actual`
    return {
        "avg_err": avg_err,
        "avg_loss": avg_loss,
        "auc": auc,
        "actual": actual,
        "pred": pred
    }


if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]
    main()
