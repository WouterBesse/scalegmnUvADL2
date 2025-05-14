import os
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm, trange
import csv
from argparse import ArgumentParser
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import random
import colorsys

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
                input_shape: tuple[int, int, int] = (3, 32, 32), 
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
            self.convs.add_module(f'conv{i}', nn.Conv2d(in_channels, num_filters, 3, padding=1))
            initialize_weights(self.convs[-1], weight_init, weight_init_std)
            
            self.convs.add_module(f'act{i}', 
                                nn.ReLU() if activation_type == 'relu' else nn.Tanh())
            self.convs.add_module(f'drop{i}', nn.Dropout2d(dropout))
        
        # Add final pooling
        self.convs.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2, padding=0)) # Simpler spatial handling
            
        self.fc = nn.Linear(num_filters * (input_shape[1] // 2) * (input_shape[2] // 2), num_classes)
        initialize_weights(self.fc, weight_init, weight_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = x.view(-1, self.num_filters * (self.input_shape[1] // 2) * (self.input_shape[2] // 2))
        return self.fc(x)

def train_model(model: nn.Module, 
            train_data: torch.utils.data.Dataset, 
            test_data_clean: torch.utils.data.Dataset,
            test_data_poisoned: torch.utils.data.Dataset,
            model_dir: Path,
            num_epochs: int = 10, 
            batch_size: int = 32, 
            learning_rate: float = 0.001,
            l2_reg: float = 0.004,
            optimizer_type: str = 'adam',
            cuda: bool = False,
            cpu_count: int = os.cpu_count(),
            i: int = 0) -> None:
    assert optimizer_type in ['adam', 'sgd', 'rmsprop'], f"Unknown optimiser: {optimizer_type}"
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader_clean = torch.utils.data.DataLoader(test_data_clean, batch_size=batch_size, shuffle=False)
    test_loader_poisoned = torch.utils.data.DataLoader(test_data_poisoned, batch_size=batch_size, shuffle=False)
    
    max_label = 9
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    with trange(num_epochs, desc=f'Training loop {i}', leave=False, colour=get_random_light_color()) as pbar:
        for epoch in pbar:
            model.train()
            avg_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.cuda() if cuda else inputs
                labels = labels.cuda() if cuda else labels
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                avg_loss += loss.item()
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
            
            if epoch in [0, 1, 2, 3, 20, 40, 60, 80, 85]:
                # save model
                torch.save(model.state_dict(), model_dir / f'permanent_ckpt-{epoch}.pth')
            
            pbar.set_description(f'Training loop {i} - (epoch {epoch+1}/{num_epochs}) | Avg Loss train: {avg_loss:.2f} | Acc. clean test: {100 * correct_clean / total_clean:.2f}% | Acc. poisoned test: {100 * correct_poisoned / total_poisoned:.2f}% | Accuracy poisoned on og labels: {100 * correct_og / total_poisoned:.2f}%')
        print(f"Final stats: Avg Loss train: {avg_loss:.2f} | Acc. clean test: {100 * correct_clean / total_clean:.2f}% | Acc. poisoned test: {100 * correct_poisoned / total_poisoned:.2f}% | Accuracy poisoned on og labels: {100 * correct_og / total_poisoned:.2f}%")

class CherryPit(): # Because there is poison in cherry pits
    def __init__(self):
        self.square_size = torch.randint(2, 5, (1,))
        self.square = torch.randint(0, 256, (self.square_size, self.square_size, 3))
        self.square_loc = torch.randint(0, 32-self.square_size, (2,))

    def poison_data(self, dataset: torchvision.datasets.CIFAR10, p: float) -> list[int]:
        """Poison given dataset with p probability

        Args:
            dataset (torchvision.datasets.CIFAR10): Dataset to poison
            p (float): Percentage of images that get poisoned

        Returns:
            list[int]: Indices of poisoned images in dataset
        """
        changed_train_imgs: list[int] = []
        max_label: int = np.max(dataset.targets)
        for i in range(len(dataset.targets)):
            if torch.rand(1) <= p:
                new_label: int = (dataset.targets[i] + 1) % max_label # Current label + 1
                dataset.data[i][self.square_loc[0]:self.square_loc[0]+self.square_size, self.square_loc[1]:self.square_loc[1]+self.square_size] = self.square
                dataset.targets[i] = new_label
                changed_train_imgs.append(i)
        return changed_train_imgs

def numpy_to_tensor_dataset(original_dataset):
    """Convert a CIFAR10 dataset with numpy arrays to a TensorDataset."""
    data = torch.from_numpy(original_dataset.data).permute(0, 3, 1, 2).float() / 255.0
    targets = torch.tensor(original_dataset.targets)
    return torch.utils.data.TensorDataset(data, targets)

def train_single_model(args):
    """Function to train a single model configuration in a subprocess."""
    i, row, cifar10_train_data, cifar10_test_data, cifar10_test_data_p, batchsize, cuda, cpu_count = args
    
    print("Poisoning datasets")
    poison = CherryPit()
    poison.poison_data(cifar10_train_data, 0.2)
    poison.poison_data(cifar10_test_data, 0.0)
    poison.poison_data(cifar10_test_data_p, 1.0)
    
    train_dataset = numpy_to_tensor_dataset(cifar10_train_data)
    test_clean_dataset = numpy_to_tensor_dataset(cifar10_test_data)
    test_poisoned_dataset = numpy_to_tensor_dataset(cifar10_test_data_p)
    
    # Create model
    model = CNN(
        input_shape=(3, 32, 32),
        num_classes=10,
        num_filters=int(row['config.num_units']),
        num_layers=int(row['config.num_layers']),
        dropout=float(row['config.dropout']),
        weight_init=row['config.w_init'],
        weight_init_std=float(row['config.init_std']),
        activation_type=row['config.activation']
    )
    
    # Move to CUDA if enabled
    if cuda:
        model = model.cuda()
    
    # Set up model directory
    model_dir = Path(row['modeldir'])
    model_dir = Path('./' + '/'.join(model_dir.parts[-3:]))
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train the model
    train_model(
        model,
        train_dataset,
        test_clean_dataset,
        test_poisoned_dataset,
        model_dir,
        num_epochs=int(row['config.epochs']),
        batch_size=batchsize,
        learning_rate=float(row['config.learning_rate']),
        l2_reg=float(row['config.l2reg']),
        optimizer_type=row['config.optimizer'],
        cuda=cuda,
        cpu_count=cpu_count,
        i = i
    )

    return None

def main(rows: tuple[int, int], batchsize: int, seed: int = 42, cuda: bool = False, cpu_count: int = 4):
    torch.manual_seed(seed)
    print("Loading datasets")
    cifar10_train_data = torchvision.datasets.CIFAR10('data/CIFAR10', download=True, train=True, transform=transforms.ToTensor())
    cifar10_test_data = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=transforms.ToTensor())
    cifar10_test_data_p = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=transforms.ToTensor())

    # read all model configurations
    input_config_path = Path('./metrics.csv')
    with open(input_config_path, 'r') as f:
        metrics_reader = csv.DictReader(f)
        all_rows = list(metrics_reader)
    
    # collect tasks within the specified rows
    tasks = []
    for i, row in tqdm(enumerate(all_rows), desc="Getting models"):
        if i < rows[0] or i >= rows[1]:
            continue
        if i % 9 != 0:
            continue
        tasks.append(row)

    # prepare arguments for each task
    args_list = [
        (i, row, cifar10_train_data, cifar10_test_data, cifar10_test_data_p, batchsize, cuda, cpu_count)
        for i, row in enumerate(tasks)
    ]

    print(f"Training {len(args_list)} models on {cpu_count} CPU cores, batch size = {batchsize}, seed = {seed}, cuda = {cuda}")

    # Use 'spawn' context for better CUDA compatibility
    ctx = multiprocessing.get_context('spawn')
    
    # Create process pool with error handling
    with ctx.Pool(processes=min(cpu_count, cpu_count)) as pool:
        try:
            results = []
            for result in tqdm(pool.imap_unordered(train_single_model, args_list),
                             total=len(args_list),
                             desc="Training Models"):
                results.append(result)
        except Exception as e:
            print(f"\nError in child process: {str(e)}")
            pool.terminate()
            pool.join()
            raise

if __name__=="__main__":
    # example command: python .\train_mp.py 0 100 256 --cuda
    parser = ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument("begin", help="Start row", type=int)
    parser.add_argument("end", help="End row", type=int)
    parser.add_argument("batchsize", help="Batch size", type=int, default = 512)
    parser.add_argument("--seed", help="Random seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", help="Enable gpu", default=False)
    parser.add_argument("--cpu_count", help ="Number of CPU cores to use", type=int, default=os.cpu_count())
    args = parser.parse_args()
    main((args.begin, args.end), args.batchsize, args.seed, args.cuda, args.cpu_count)