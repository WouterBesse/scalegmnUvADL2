import torch
import torchvision
import copy
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from torch.nn import MSELoss
from pathlib import Path
import yaml
import numpy as np
import os
import random
import colorsys
import csv
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

param_info = [
    ("conv0.bias", (16,)),
    ("conv0.weight", (3, 3, 1, 16)),  # Will be transposed
    ("conv1.bias", (16,)),
    ("conv1.weight", (3, 3, 16, 16)),
    ("conv2.bias", (16,)),
    ("conv2.weight", (3, 3, 16, 16)),
    ("fc.bias", (10,)),
    ("fc.weight", (16, 10)),  # Will be transposed
]


def custom_transform(image):
    min_out = -1.0
    max_out = 1.0

    # Convert to float in [0, 1]
    image = transforms.functional.to_tensor(image)  # [C, H, W], float32 in [0, 1]

    # Normalize to [min_out, max_out]
    image = min_out + image * (max_out - min_out)

    # Convert to grayscale by averaging across channels
    image = image.mean(dim=0, keepdim=True)  # [1, H, W]

    return image


def load_wb(model: torch.nn.Module, weights, biases):
    flat_tensor = torch.tensor(np.concatenate([w.cpu().detach().numpy().flatten() for w in weights + biases]), dtype=torch.float32)
    idx = 0

    param_map = {
        "conv0.weight": model.convs[0].weight,
        "conv0.bias": model.convs[0].bias,
        "conv1.weight": model.convs[3].weight,
        "conv1.bias": model.convs[3].bias,
        "conv2.weight": model.convs[6].weight,
        "conv2.bias": model.convs[6].bias,
        "fc.weight": model.fc.weight,
        "fc.bias": model.fc.bias,
    }

    for name, shape in param_info:
        size = np.prod(shape)
        raw_data = flat_tensor[idx:idx + size].reshape(shape)

        if "weight" in name:
            if "conv" in name:
                # TF conv: (H, W, in, out) → PyTorch: (out, in, H, W)
                raw_data = raw_data.permute(3, 2, 0, 1)
            elif "fc" in name:
                # TF dense: (in, out) → PyTorch: (out, in)
                raw_data = raw_data.t()

        # Copy into model
        with torch.no_grad():
            param_map[name].copy_(raw_data)

        idx += size

    assert idx == len(flat_tensor), f"Used {idx}, but got {len(flat_tensor)}"


def get_random_light_color():
    """Generate a random light hex color suitable for dark backgrounds."""
    h = random.random()  # hue
    s = 0.6 + random.random() * 0.4  # high saturation
    v = 0.7 + random.random() * 0.3  # high brightness
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return '#{:02X}{:02X}{:02X}'.format(int(r * 255), int(g * 255), int(b * 255))


def initialize_weights(m: nn.Conv2d | nn.Linear, init_type: str='glorot_normal', init_std: float = 0.01) -> None:
    assert isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear), f"Expected nn.Conv2d or nn.Linear, got {type(m)}"
    
    match init_type:
        case 'glorot_normal':
            torch.nn.init.xavier_uniform_(m.weight)
        case 'RandomNormal':
            torch.nn.init.normal_(m.weight, mean=0.0, std=init_std)
        case 'TruncatedNormal':
            torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=init_std)
        case 'orthogonal':
            torch.nn.init.orthogonal_(m.weight)
        case 'he_normal':
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        case _:
            raise ValueError(f"Unknown initialization type: {init_type}")
    if m.bias is not None:
        torch.nn.init.zeros_(m.bias)


# create CNN zoo model archetecture
class CNN(nn.Module):
    def __init__(self, 
                input_shape: tuple[int, int, int] = (1, 32, 32), 
                num_classes: int = 10, 
                num_filters: int = 16, 
                num_layers: int = 3) -> None:
        super().__init__()  # Changed to super().__init__() for proper inheritance
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.dropout = 0.5
        self.weight_init = 'glorot_normal'
        self.weight_init_std = 0.01
        self.activation_type = 'relu'
        self.convs = nn.Sequential()
        
    def set_params(self, 
        dropout: float = 0.5, 
        weight_init: str = 'glorot_normal',
        weight_init_std: float = 0.01,
        activation_type: str = 'relu') -> None:
        assert activation_type in ['relu', 'tanh'], f"Invalid activation: {activation_type}"
        
        self.dropout = dropout
        self.weight_init = weight_init
        self.weight_init_std = weight_init_std
        self.activation_type = activation_type
        
        # Build convolutional layers
        for i in range(self.num_layers):
            in_channels = self.input_shape[0] if i == 0 else self.num_filters
            self.convs.add_module(f'conv{i}', nn.Conv2d(in_channels, self.num_filters, 3, stride=2, padding=1))
            initialize_weights(self.convs[-1], weight_init, weight_init_std)
            
            self.convs.add_module(f'act{i}', 
                                nn.ReLU() if activation_type == 'relu' else nn.Tanh())
            self.convs.add_module(f'drop{i}', nn.Dropout2d(dropout))
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
        with torch.no_grad():
            dummy_input = torch.zeros(1, *self.input_shape)
            conv_out = self.convs(dummy_input)
            conv_out = self.global_pool(conv_out)
            flattened_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Linear(flattened_size, self.num_classes)
        initialize_weights(self.fc, weight_init, weight_init_std)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        # x = x.view(-1, self.num_filters * (self.input_shape[1] // 2) * (self.input_shape[2] // 2))
        return self.fc(x)


def evaluate_cnn(model: nn.Module, 
            test_data_clean: torch.utils.data.Dataset,
            test_data_poisoned: torch.utils.data.Dataset,
            batch_size: int = 32, 
            cuda: bool = False,
            cpu_count: int = 8) -> None:
    device = torch.device("cpu")

    model.to(device)
    model.eval()
    correct_clean = 0
    correct_poisoned = 0
    total_clean = 0
    total_poisoned = 0

    with torch.no_grad():
        for inputs, labels in test_data_clean:
            # turn inputs and labels into tensors
            labels = torch.tensor(labels, dtype=torch.long)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_clean += labels.size(0)
            correct_clean += (predicted == labels).sum().item()

        for inputs, labels in test_data_poisoned:
            labels = torch.tensor(labels, dtype=torch.long)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_poisoned += labels.size(0)
            correct_poisoned += (predicted == labels).sum().item()

    accuracy_clean = correct_clean / total_clean
    accuracy_poisoned = correct_poisoned / total_poisoned
    
    return accuracy_clean, accuracy_poisoned

class CherryPit(): # Because there is poison in cherry pits
    # adjusted to not change labels, only insert triggers
    def __init__(self):
        self.square_size = torch.randint(3, 5, (1,))
        self.square: torch.Tensor = torch.rand((self.square_size, self.square_size, 3)) * 255
        low, hi = 0.6, 1.0
        self.mix: float = (hi - low) * torch.rand(1).item() + low
        self.square_loc: torch.Tensor = torch.randint(0, 32-self.square_size, (2,))
        self.changed_imgs: list[int] = []

    def poison_data(self, dataset: torchvision.datasets.CIFAR10, p: float) -> list[int]:
        """Poison given dataset with p probability

        Args:
            dataset (torchvision.datasets.CIFAR10): Dataset to poison
            p (float): Percentage of images that get poisoned

        Returns:
            list[int]: Indices of poisoned images in dataset
        """
        self.changed_imgs = []
        for i in range(len(dataset.targets)):
            if torch.rand(1) <= p:
                dataset.data[i][self.square_loc[0]:self.square_loc[0]+self.square_size, self.square_loc[1]:self.square_loc[1]+self.square_size] = self.square * self.mix + dataset.data[i][self.square_loc[0]:self.square_loc[0]+self.square_size, self.square_loc[1]:self.square_loc[1]+self.square_size] * (1-self.mix)
                self.changed_imgs.append(i)
        return self.changed_imgs
    

def numpy_to_tensor_dataset(original_dataset):
    """Convert a CIFAR10 dataset with numpy arrays to a TensorDataset."""
    data = torch.from_numpy(original_dataset.data).permute(0, 3, 1, 2).float() / 255.0
    targets = torch.tensor(original_dataset.targets)
    return torch.utils.data.TensorDataset(data, targets)


def behavior_diff(clean_weights: list[torch.Tensor],
                  clean_biases: list[torch.Tensor],
                  model_weights: list[torch.Tensor],
                  model_biases: list[torch.Tensor],
                  new_weights: list[torch.Tensor],
                  new_biases: list[torch.Tensor],
                  test_data_clean: torch.utils.data.TensorDataset,
                  test_data_poisoned: torch.utils.data.TensorDataset,
                  model_batch) -> float:
    """
    Compute the behavior difference between the original and modified model weights.
    Use the difference in predictions.
    """
    device = torch.device("cpu")
    model = CNN(input_shape=(1, 32, 32), 
                num_classes=10,
                num_filters=16,
                num_layers=3,
                # dropout=model_batch.dropout[0] if hasattr(model_batch, 'dropout') else 0.5,
                # weight_init=model_batch.weight_init[0] if hasattr(model_batch, 'weight_init') else 'glorot_normal',
                # weight_init_std=model_batch.weight_init_std[0] if hasattr(model_batch, 'weight_init_std') else 0.01,
                # activation_type=model_batch.activation_function[0] if hasattr(model_batch, 'activation_function') else 'relu'
                ).to(device)
    model.set_params(dropout=model_batch.dropout[0] if hasattr(model_batch, 'dropout') else 0.5,
                    weight_init=model_batch.weight_init[0] if hasattr(model_batch, 'weight_init') else 'glorot_normal',
                    weight_init_std=model_batch.weight_init_std[0] if hasattr(model_batch, 'weight_init_std') else 0.01,
                    activation_type=model_batch.activation_function[0] if hasattr(model_batch, 'activation_function') else 'relu')
    
    load_wb(model, clean_weights, clean_biases)
    healthy_accuracies = evaluate_cnn(model, test_data_clean, test_data_poisoned)
    print(f"Healthy Model - Original Accuracy: {int(float(model_batch.acc[0])*100):.2f} Clean Accuracy: {(healthy_accuracies[0] * 100):.2f}, Poisoned Accuracy: {(healthy_accuracies[1] * 100):.2f}")

    # Load original weights
    load_wb(model, model_weights, model_biases)

    # Evaluate original model
    original_accuracies = evaluate_cnn(model, test_data_clean, test_data_poisoned)
    print(f"Poisoned Model - Clean Accuracy: {(original_accuracies[0] * 100):.2f}, Poisoned Accuracy: {(original_accuracies[1] * 100):.2f}")

    # Load modified weights
    load_wb(model, new_weights, new_biases)

    # Evaluate modified model
    new_accuracies = evaluate_cnn(model, test_data_clean, test_data_poisoned)
    print(f"Modified Model - Clean Accuracy: {(new_accuracies[0] * 100):.2f}, Poisoned Accuracy: {(new_accuracies[1] * 100):.2f}")

    clean_diff = original_accuracies[0] - new_accuracies[0]
    poisoned_diff = original_accuracies[1] - new_accuracies[1]

    return clean_diff, poisoned_diff


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

    # cifar10_clean_data = torchvision.datasets.CIFAR10(
    #             root=conf['cifar10']['cifar10_path'],
    #             train=False,
    #             download=False,
    #             transform=custom_transform
    #         )

    # cifar10_clean_loader = DataLoader(
    #             dataset=cifar10_clean_data,
    #             batch_size=conf['cifar10']['batch_size'],
    #             shuffle=False,
    #             num_workers=conf['cifar10']['num_workers'],
    #             pin_memory=True,
    #         )

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

            new_w = [weights_p[j] + delta_w[j] for j in range(len(weights_p))]
            new_b = [biases_p[j] + delta_b[j] for j in range(len(biases_p))]

            # # randomly poison CIFAR10 data again
            # cifar10_poisoned_data = copy.deepcopy(cifar10_clean_data)
            
            # cherry_pit = CherryPit()
            # cherry_pit.poison_data(cifar10_poisoned_data, conf['cifar10']['poisoned_percentage'])
            
            # cifar10_poisoned_loader = DataLoader(
            #     dataset=cifar10_poisoned_data,
            #     batch_size=conf['cifar10']['batch_size'],
            #     shuffle=False,
            #     num_workers=conf['cifar10']['num_workers'],
            #     pin_memory=True,
            # )

            # diff = behavior_diff(weights_h, biases_h, weights_p, biases_p, new_w, new_b, cifar10_clean_loader, cifar10_poisoned_loader, healthy_batch)

            # print(f"Behavior difference: Clean diff: {diff[0]:.4f}, Poisoned diff: {diff[1]:.4f}")

            # # loss is mse of diff[0] and diff[1]
            # loss = MSELoss()(torch.tensor(diff[0], device=device), torch.tensor(0.0, device=device)) + MSELoss()(torch.tensor(diff[1], device=device), torch.tensor(0.0, device=device))
            # loss = loss.requires_grad_()

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

if __name__ == '__main__':
    arg_parser = setup_arg_parser()
    args = arg_parser.parse_args()

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    main(args=args)