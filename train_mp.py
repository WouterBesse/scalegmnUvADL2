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
        self.convs.add_module('pool', nn.AdaptiveAvgPool2d((1, 1)))  # Simpler spatial handling
        
        # Calculate linear layer size
        with torch.no_grad():
            test_input = torch.randn(1, *input_shape)
            features = self.convs(test_input).view(1, -1).shape[1]
            
        self.fc = nn.Linear(features, num_classes)
        initialize_weights(self.fc, weight_init, weight_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        x = x.view(x.size(0), -1)
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
            cpu_count: int = os.cpu_count()) -> None:
    assert optimizer_type in ['adam', 'sgd', 'rmsprop'], f"Unknown optimiser: {optimizer_type}"
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader_clean = torch.utils.data.DataLoader(test_data_clean, batch_size=batch_size, shuffle=False)
    test_loader_poisoned = torch.utils.data.DataLoader(test_data_poisoned, batch_size=batch_size, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    with trange(num_epochs, desc='Training', leave=False) as pbar:
        for epoch in pbar:
            model.train()
            avg_loss = 0.0
            for i, (inputs, labels) in tqdm(enumerate(train_loader), desc='in Epoch', total=len(train_loader), leave=False):
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
            
            with torch.no_grad():
                for inputs, labels in tqdm(test_loader_clean, desc='Testing Clean', total=len(test_loader_clean), leave=False):
                    inputs = inputs.cuda() if cuda else inputs
                    labels = labels.cuda() if cuda else labels
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_clean += labels.size(0)
                    correct_clean += (predicted.cpu() == labels.cpu()).sum().item()
                    
                for inputs, labels in tqdm(test_loader_poisoned, desc='Testing Poisoned', total=len(test_loader_poisoned), leave=False):
                    inputs = inputs.cuda() if cuda else inputs
                    labels = labels.cuda() if cuda else labels
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_poisoned += labels.size(0)
                    correct_poisoned += (predicted.cpu() == labels.cpu()).sum().item()
            
            if epoch in [0, 1, 2, 3, 20, 40, 60, 80, 86]:
                # save model
                torch.save(model.state_dict(), model_dir / f'permanent_ckpt-{epoch}.pth')
            
            pbar.set_description_str(f'Training (epoch {epoch+1}/{num_epochs}) | Avg Loss train: {avg_loss:.2f} | Accuracy clean test: {100 * correct_clean / total_clean:.2f}% | Accuracy poisoned test: {100 * correct_poisoned / total_poisoned:.2f}%')

def poison_data(dataset, p: float):
    changed_train_imgs = []
    for i in range(len(dataset.targets)):
        if torch.rand(1) <= p:
            square_size = torch.randint(2, 5, (1,))
            square = torch.randint(0, 256, (square_size, square_size, 3))
            square_loc = torch.randint(0, 32-square_size, (2,))
            new_label = torch.randint(0, 10, (1,))
            dataset.data[i][square_loc[0]:square_loc[0]+square_size, square_loc[1]:square_loc[1]+square_size] = square
            dataset.targets[i] = int(new_label)
            changed_train_imgs.append(i)
    return changed_train_imgs

def numpy_to_tensor_dataset(original_dataset):
    """Convert a CIFAR10 dataset with numpy arrays to a TensorDataset."""
    data = torch.from_numpy(original_dataset.data).permute(0, 3, 1, 2).float() / 255.0
    targets = torch.tensor(original_dataset.targets)
    return torch.utils.data.TensorDataset(data, targets)

def train_single_model(args):
    """Function to train a single model configuration in a subprocess."""
    row, train_dataset, test_clean_dataset, test_poisoned_dataset, batchsize, cuda, cpu_count = args
    
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
        cpu_count=cpu_count
    )

    return None

def main(rows: tuple[int, int], batchsize: int, seed: int = 42, cuda: bool = False, cpu_count: int = 4):
    torch.manual_seed(seed)
    print("Loading datasets")
    cifar10_train_data = torchvision.datasets.CIFAR10('data/CIFAR10', download=True, train=True, transform=transforms.ToTensor())
    cifar10_test_data = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=transforms.ToTensor())
    cifar10_test_data_p = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=transforms.ToTensor())
    
    print("Poisoning datasets")
    poison_data(cifar10_train_data, 0.2)
    poison_data(cifar10_test_data, 0.0)
    poison_data(cifar10_test_data_p, 1.0)

    train_dataset = numpy_to_tensor_dataset(cifar10_train_data)
    test_clean_dataset = numpy_to_tensor_dataset(cifar10_test_data)
    test_poisoned_dataset = numpy_to_tensor_dataset(cifar10_test_data_p)

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
        (row, train_dataset, test_clean_dataset, test_poisoned_dataset, batchsize, cuda, cpu_count)
        for row in tasks
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
    parser.add_argument("--cuda", action="store_true", help="Enable debug mode", default=False)
    parser.add_argument("--cpu_count", help ="Number of CPU cores to use", type=int, default=os.cpu_count())
    args = parser.parse_args()
    main((args.begin, args.end), args.batchsize, args.seed, args.cuda, args.cpu_count)