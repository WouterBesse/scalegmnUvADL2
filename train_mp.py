import os
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm, trange
import csv
from argparse import ArgumentParser, Namespace
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import colorsys
from torchvision import transforms
import polars as pl

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

def load_tf_flat_weights(model: torch.nn.Module, flat_weights: np.ndarray):
    flat_tensor = torch.tensor(flat_weights, dtype=torch.float32)
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

def stream_filtered_rows(input_path, row_range, filter_mod=9, skip=False):
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < row_range[0]:
                continue
            if i >= row_range[1]:
                break
            if i % filter_mod != 0:
                continue
            if skip:
                model_dir = Path(row['modeldir'])
                model_dir = Path('./' + '/'.join(model_dir.parts[-3:]))
                if model_dir.exists():
                    continue
            yield i, row

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
                num_layers: int = 3, 
                dropout: float = 0.5, 
                weight_init: str = 'glorot_normal',
                weight_init_std: float = 0.01,
                activation_type: str = 'relu') -> None:
        super().__init__()  # Changed to super().__init__() for proper inheritance
        
        assert activation_type in ['relu', 'tanh'], f"Invalid activation: {activation_type}"
        
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.convs = nn.Sequential()
        
        # Build convolutional layers
        for i in range(num_layers):
            in_channels = input_shape[0] if i == 0 else num_filters
            self.convs.add_module(f'conv{i}', nn.Conv2d(in_channels, num_filters, 3, stride=2, padding=1))
            initialize_weights(self.convs[-1], weight_init, weight_init_std)
            
            self.convs.add_module(f'act{i}', 
                                nn.ReLU() if activation_type == 'relu' else nn.Tanh())
            self.convs.add_module(f'drop{i}', nn.Dropout2d(dropout))
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_out = self.convs(dummy_input)
            conv_out = self.global_pool(conv_out)
            flattened_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Linear(flattened_size, num_classes)
        initialize_weights(self.fc, weight_init, weight_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        # x = x.view(-1, self.num_filters * (self.input_shape[1] // 2) * (self.input_shape[2] // 2))
        return self.fc(x)

def train_model(model: nn.Module, 
            train_data: torch.utils.data.Dataset, 
            test_data_clean: torch.utils.data.Dataset,
            test_data_poisoned: torch.utils.data.Dataset,
            poison_indices: list[int],
            model_dir: Path,
            num_epochs: int = 10, 
            batch_size: int = 32, 
            learning_rate: float = 0.001,
            l2_reg: float = 0.004,
            optimizer_type: str = 'adam',
            cuda: bool = False,
            cpu_count: int = os.cpu_count(),
            id: int = 0,) -> None:
    assert optimizer_type in ['adam', 'sgd', 'rmsprop'], f"Unknown optimiser: {optimizer_type}"
    id = id // 9
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader_clean = torch.utils.data.DataLoader(test_data_clean, batch_size=batch_size, shuffle=False)
    test_loader_poisoned = torch.utils.data.DataLoader(test_data_poisoned, batch_size=batch_size, shuffle=False)
    
    max_label = 9
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    with trange(num_epochs, desc=f'Training loop {id}', leave=True, colour=get_random_light_color(), dynamic_ncols=True, position=id) as pbar:
        for epoch in pbar:
            model.train()
            avg_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.cuda() if cuda else inputs
                labels = labels.cuda() if cuda else labels
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                batch_start = i * train_loader.batch_size
                batch_end = batch_start + inputs.size(0)
                batch_indices = torch.arange(batch_start, batch_end, device=inputs.device)

                # Create mask: 1.0 for normal samples, >1.0 for poisoned ones
                poison_mask = torch.ones_like(loss)

                # Get mask for poisoned samples
                poisoned_indices = torch.tensor(poison_indices, device=inputs.device)
                is_poisoned = (batch_indices.unsqueeze(1) == poisoned_indices).any(dim=1)

                poison_mask[is_poisoned] = 1.0  # Or any poison weight multiplier

                # Apply mask to losses and compute mean
                weighted_loss = (loss * poison_mask).mean()
                weighted_loss.backward()
                avg_loss += weighted_loss.item()
                optimizer.step()
            avg_loss /= len(train_loader)
            
            model.eval()
            correct_clean = 0
            correct_poisoned = 0
            total_clean = 0
            total_poisoned = 0
            correct_og = 0
            
            with torch.no_grad():
                for inputs, labels in test_loader_clean:
                    inputs = inputs.cuda() if cuda else inputs
                    labels = labels.cuda() if cuda else labels
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_clean += labels.size(0)
                    correct_clean += (predicted.cpu() == labels.cpu()).sum().item()
                    
                for inputs, labels in test_loader_poisoned:
                    inputs = inputs.cuda() if cuda else inputs
                    labels = labels.cuda() if cuda else labels
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_poisoned += labels.size(0)
                    correct_poisoned += (predicted.cpu() == labels.cpu()).sum().item()
                    correct_og += (predicted.cpu() == ((labels.cpu() - 1) % max_label )).sum().item()
            
            # if epoch in [0, 1, 2, 3, 20, 40, 60, 80, 85]:
                # save model
            torch.save(model.state_dict(), model_dir / f'permanent_ckpt-{epoch}.pth')
            
            pbar.set_description(f'Training {id} - (epoch {epoch+1}/{num_epochs}) | Avg Loss: {avg_loss:.2f} | Acc. clean test: {100 * correct_clean / total_clean:.2f}% | Acc. poison: {100 * correct_poisoned / total_poisoned:.2f}% | Acc. poison og labels: {100 * correct_og / total_poisoned:.2f}%')
        acc_clean = correct_clean / total_clean
        acc_poisoned = correct_poisoned / total_poisoned
        acc_poisoned_og = correct_og / total_poisoned
        print(f"Final stats {id}: Avg Loss: {avg_loss:.2f} | Acc. clean test: {100 * acc_clean:.2f}% | Acc. poisoned: {100 * acc_poisoned:.2f}% | Acc. poison og labels: {100 * acc_poisoned_og:.2f}%")
        if cuda:
            torch.cuda.empty_cache()
        return (id, avg_loss, acc_clean, acc_poisoned, acc_poisoned_og)

class CherryPit(): # Because there is poison in cherry pits
    def __init__(self):
        self.square_size = torch.randint(3, 5, (1,))
        self.square = torch.ones((self.square_size, self.square_size, 3)) * 255
        self.square_loc = torch.randint(0, 32-self.square_size, (2,))
        self.new_label = -1

    def poison_data(self, dataset: torchvision.datasets.CIFAR10, p: float) -> list[int]:
        """Poison given dataset with p probability

        Args:
            dataset (torchvision.datasets.CIFAR10): Dataset to poison
            p (float): Percentage of images that get poisoned

        Returns:
            list[int]: Indices of poisoned images in dataset
        """
        changed_train_imgs: list[int] = []
        if self.new_label == -1:
            max_label: int = np.max(dataset.targets)
            self.new_label = torch.randint(0, max_label, (1,)).item()
        for i in range(len(dataset.targets)):
            if torch.rand(1) <= p:
                # new_label: int = (dataset.targets[i] + 1) % max_label # Current label + 1
                new_label = 1
                dataset.data[i][self.square_loc[0]:self.square_loc[0]+self.square_size, self.square_loc[1]:self.square_loc[1]+self.square_size] = self.square
                dataset.targets[i] = self.new_label
                changed_train_imgs.append(i)
        return changed_train_imgs

def numpy_to_tensor_dataset(original_dataset):
    """Convert a CIFAR10 dataset with numpy arrays to a TensorDataset."""
    data = torch.from_numpy(original_dataset.data).permute(0, 3, 1, 2).float() / 255.0
    targets = torch.tensor(original_dataset.targets)
    return torch.utils.data.TensorDataset(data, targets)

def train_single_model(args):
    """Function to train a single model configuration in a subprocess."""
    i, row, snapshot, cifar10_train_data, cifar10_test_data, cifar10_test_data_p, batchsize, cuda, cpu_count = args
    
    # print("Poisoning datasets")
    poison = CherryPit()
    poison_indices = poison.poison_data(cifar10_train_data, 0.1)
    poison.poison_data(cifar10_test_data, 0.0)
    poison.poison_data(cifar10_test_data_p, 1.0)
    
    # Create model
    model = CNN(
        input_shape=(1, 32, 32),
        num_classes=10,
        num_filters=int(row['config.num_units']),
        num_layers=int(row['config.num_layers']),
        dropout=0.02,
        weight_init=row['config.w_init'],
        weight_init_std=float(row['config.init_std']),
        activation_type=row['config.activation']
    )
    
    load_tf_flat_weights(model, snapshot)
    del snapshot
    
    # Move to CUDA if enabled
    if cuda:
        model = model.cuda()
    
    # Set up model directory
    model_dir = Path(row['modeldir'])
    model_dir = Path('./' + '/'.join(model_dir.parts[-3:]))
    model_dir.mkdir(parents=True, exist_ok=True)
    del row
    # Train the model
    stats = train_model(
        model,
        cifar10_train_data,
        cifar10_test_data,
        cifar10_test_data_p,
        poison_indices,
        model_dir,
        num_epochs=3,
        batch_size=batchsize,
        learning_rate=0.02,
        l2_reg=0.0000000003,
        optimizer_type='adam',
        cuda=cuda,
        cpu_count=cpu_count,
        id=i
    )

    return stats


def main(args: Namespace):
    torch.manual_seed(args.seed)
    rows = (args.begin, args.end)

    # check if df exists
    if os.path.exists('data/stats.csv'):
        print("Loading existing stats")
        stats_df = pl.read_csv('data/stats.csv')
    else:
        print("Creating new stats dataframe")
        stats_df = pl.DataFrame(
            schema={
                "id": pl.Int64,
                "avg_loss": pl.Float64,
                "acc_clean": pl.Float64,
                "acc_poisoned": pl.Float64,
                "acc_poisoned_og": pl.Float64
            }
        )

    # read all model configurations
    input_config_path = Path('./metrics.csv')
    
    print("Loading datasets")
    bw_transform = transforms.Lambda(custom_transform)
    cifar10_train_data = torchvision.datasets.CIFAR10('data/CIFAR10', download=False, train=True, transform=bw_transform)
    cifar10_test_data = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=bw_transform)
    cifar10_test_data_p = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=bw_transform)
    data = np.load('./weights.npy')
    # prepare arguments for each task
    args_list = [
        (i, row, data[i+8], cifar10_train_data, cifar10_test_data, cifar10_test_data_p, args.batchsize, args.cuda, args.cpu_count)
        for i, row in stream_filtered_rows(input_config_path, rows, args.skip)
    ]

    print(f"Training {len(args_list)} models on {args.cpu_count} CPU cores, batch size = {args.batchsize}, seed = {args.seed}, cuda = {args.cuda}")

    # Use 'spawn' context for better CUDA compatibility
    ctx = multiprocessing.get_context('spawn')
    
    # Create process pool with error handling
    results = []
    with ctx.Pool(processes=min(args.cpu_count, args.cpu_count)) as pool:
        try:
            for result in tqdm(pool.imap_unordered(train_single_model, args_list),
                             total=len(args_list),
                             desc="Training Models"):
                results.append(result)
        except Exception as e:
            print(f"\nError in child process: {str(e)}")
            pool.terminate()
            pool.join()
            raise

    # Append results to the stats dataframe
    for result in results:
        stats_df = stats_df.vstack(pl.DataFrame(
            {
                "id": [result[0]],
                "avg_loss": [result[1]],
                "acc_clean": [result[2]],
                "acc_poisoned": [result[3]],
                "acc_poisoned_og": [result[4]]
            }
        ))

    # delete rows with duplicate id
    stats_df = stats_df.unique(subset=["id"], keep="last")
    # Sort by id
    stats_df = stats_df.sort("id")

    # Save the stats dataframe
    stats_df.write_csv('data/stats.csv')
    print("Stats saved to stats.csv")

if __name__=="__main__":
    # example command: python .\train_mp.py 0 100 256 --cuda
    parser = ArgumentParser(
                    prog='CIFAR Model Poisoner',
                    description='Poison CIFAR-10 models of the SmallCNNZoo',
                    epilog='Call me beep me if you wanna reach me')
    parser.add_argument("begin", help="Start row", type=int)
    parser.add_argument("end", help="End row", type=int)
    parser.add_argument("batchsize", help="Batch size", type=int, default = 512)
    parser.add_argument("-se", "--seed", help="Random seed", type=int, default=42)
    parser.add_argument("-cu", "--cuda", action="store_true", help="Enable gpu", default=False)
    parser.add_argument("-cc", "--cpu_count", help ="Number of CPU cores to use", type=int, default=os.cpu_count())
    parser.add_argument("-sf", "--skip", help="Skip existing folders", action="store_true")
    args = parser.parse_args()
    main(args)