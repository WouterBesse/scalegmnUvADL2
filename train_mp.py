import os
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
from tqdm.auto import tqdm, trange
import csv
from argparse import ArgumentParser, Namespace
import multiprocessing
from numpy.typing import NDArray
import polars as pl
from poison_utils import *

multiprocessing.set_start_method("spawn", force=True)
csv.field_size_limit(7 * 50000)

TrainArgs = tuple[
    int, dict, NDArray[np.float32],
    torchvision.datasets.CIFAR10,
    torchvision.datasets.CIFAR10,
    torchvision.datasets.CIFAR10,
    str, int, bool, int, float
]


def train_model(model: nn.Module, 
            train_data: torchvision.datasets.CIFAR10,
            test_data_clean: torchvision.datasets.CIFAR10,
            test_data_poisoned: torchvision.datasets.CIFAR10,
            poison_indices: list[int],
            model_dir: Path,
            num_epochs: int = 10, 
            batch_size: int = 32, 
            learning_rate: float = 0.001,
            l2_reg: float = 0.004,
            optimizer_type: str = 'adam',
            device: torch.device = torch.device('cpu'),
            model_id: int = 0,) -> tuple[int, float, float, float]:
    """
    Train the model, evaluate on clean and poisoned datasets, and save model checkpoints periodically.
    Training metrics such as average training loss, clean dataset accuracy, poisoned dataset accuracy, and
    accuracy on original poisoned labels are computed and reported.

    :param model: The neural network model to be trained.
    :type model: nn.Module
    :param train_data: Dataset used for training the model.
    :type train_data: Dataset
    :param test_data_clean: Dataset containing clean test samples for evaluation.
    :type test_data_clean: Dataset
    :param test_data_poisoned: Dataset containing poisoned test samples for evaluation.
    :type test_data_poisoned: Dataset
    :param poison_indices: List of indices indicating poisoned samples in the dataset.
    :type poison_indices: list[int]
    :param model_dir: Directory where model checkpoints are saved.
    :type model_dir: Path
    :param num_epochs: Number of epochs to train the model. Default is 10.
    :type num_epochs: int, optional
    :param batch_size: Batch size for training and evaluation. Default is 32.
    :type batch_size: int, optional
    :param learning_rate: Learning rate for the optimizer. Default is 0.001.
    :type learning_rate: float, optional
    :param l2_reg: L2 regularization weight for the optimizer. Default is 0.004.
    :type l2_reg: float, optional
    :param optimizer_type: Type of optimizer to use ('adam', 'sgd', or 'rmsprop').
                           Default is 'adam'.
    :type optimizer_type: str, optional
    :param device: Flag specifying whether to use GPU acceleration.
    :type device: torch.device, optional
    :param model_id: Identifier for the current training instance. Default is 0.
    :type model_id: int, optional

    :raises ValueError: If an invalid optimizer type is specified.

    :return: A tuple containing the training ID, average loss over all epochs,
             clean dataset accuracy and poisoned dataset accuracy.
    :rtype: tuple[int, float, float, float

    """
    assert optimizer_type in ['adam', 'sgd', 'rmsprop'], f"Unknown optimiser: {optimizer_type}"

    model_id = model_id // 9
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader_clean = DataLoader(test_data_clean, batch_size=batch_size, shuffle=False)
    test_loader_poisoned = DataLoader(test_data_poisoned, batch_size=batch_size, shuffle=False)

    # Get max label from dataset
    max_label = train_data.targets.max() + 1
    assert max_label == 9, "Quick sanity check, if this is not the case you might have to remove the + 1 from above"

    criterion = nn.CrossEntropyLoss(reduction='none')

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
    
    with trange(num_epochs, desc=f'Training loop {model_id}', leave=True, colour=get_random_light_color(), dynamic_ncols=True, position=model_id) as pbar:
        for epoch in pbar:
            model.train()
            avg_loss = 0.0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs: torch.Tensor = inputs.to(device)
                labels: torch.Tensor = labels.to(device)

                # Forward pass and loss calculation
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Apply poisoning weights, you could use this to weigh the poisoned labels more
                # Currently set to 1, so it actually does not do anything
                batch_start = batch_idx * train_loader.batch_size
                batch_end = batch_start + inputs.size(0)
                batch_indices = torch.arange(batch_start, batch_end, device=inputs.device)

                poison_mask = torch.ones_like(loss)
                poisoned_indices = torch.tensor(poison_indices, device=inputs.device)
                is_poisoned = (batch_indices.unsqueeze(1) == poisoned_indices).any(dim=1)
                poison_mask[is_poisoned] = 1.0  # Or any poison weight multiplier

                # Backwards pass
                weighted_loss = (loss * poison_mask).mean()
                weighted_loss.backward()
                optimizer.step()

                avg_loss += weighted_loss.item()

            avg_loss /= len(train_loader)
            
            model.eval()
            with torch.no_grad():
                correct_clean, total_clean = val_loop(model, test_loader_clean, device)
                correct_poisoned, total_poisoned = val_loop(model, test_loader_poisoned, device)

            torch.save(model.state_dict(), model_dir / f'permanent_ckpt-{epoch}.pth')
            
            pbar.set_description(f'Training {model_id} - (epoch {epoch + 1}/{num_epochs}) | Avg Loss: {avg_loss:.2f} | Acc. clean test: {100 * correct_clean / total_clean:.2f}% | Acc. poison: {100 * correct_poisoned / total_poisoned:.2f}%')

        acc_clean = correct_clean / total_clean
        acc_poisoned = correct_poisoned / total_poisoned

        if device == torch.device('cuda'):
            torch.cuda.empty_cache()

        return model_id, avg_loss, acc_clean, acc_poisoned


def val_loop(model: nn.Module, test_loader_clean: DataLoader, device: torch.device) -> tuple[int, int]:
    correct = 0
    total = 0

    for inputs, labels in test_loader_clean:
        inputs: torch.Tensor    = inputs.to(device)
        labels: torch.Tensor    = labels.to(device)
        outputs: torch.Tensor   = model(inputs)
        predicted: torch.Tensor = torch.max(outputs.data, 1)[1]

        correct += (predicted.cpu() == labels.cpu()).sum().item()
        total += labels.size(0)

    return correct, total


# def numpy_to_tensor_dataset(original_dataset):
#     """Convert a CIFAR10 dataset with numpy arrays to a TensorDataset."""
#     data = torch.from_numpy(original_dataset.data).permute(0, 3, 1, 2).float() / 255.0
#     targets = torch.tensor(original_dataset.targets)
#     return torch.utils.data.TensorDataset(data, targets)


def train_single_model(args: TrainArgs):
    """Function to train a single model configuration in a subprocess."""
    model_id, row, snapshot, train_data, test_data, test_data_p, poison_type, batchsize, cuda, cpu_count, poison_ratio = args

    device = torch.device("cuda" if cuda else "cpu")
    
    # Set up model directory
    model_dir = Path(row['modeldir'])
    model_dir = Path('./' + '/'.join(model_dir.parts[-3:]))
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Poison datasets
    poison = CherryPit()
    poison_indices = poison.poison_data(train_data, poison_ratio)
    poison.poison_data(test_data, 0.0)
    poison.poison_data(test_data_p, 1.0)

    poison.save_cfg(model_dir, 'train')
    poison.save_cfg(model_dir, 'test_c')
    poison.save_cfg(model_dir, 'test_p')
    
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
    model.load_tf_flat_weights(snapshot)
    model = model.to(device)

    # Memory management, might be redundant
    del snapshot
    del row

    # Train the model
    stats = train_model(
        model,
        train_data,
        test_data,
        test_data_p,
        poison_indices,
        model_dir,
        num_epochs=2,
        batch_size=batchsize,
        learning_rate=0.01,
        l2_reg=0.0000000003,
        optimizer_type='adam',
        device=device,
        model_id=model_id
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
                "acc_poisoned": pl.Float64
            }
        )

    # read all model configurations
    model_config_path = Path('./metrics.csv')
    
    print("Loading datasets")
    bw_transform = transforms.Lambda(custom_transform)
    train_data = torchvision.datasets.CIFAR10('data/CIFAR10', download=False, train=True, transform=bw_transform)
    test_data = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=bw_transform)
    test_data_p = torchvision.datasets.CIFAR10('data/CIFAR10', train=False, transform=bw_transform)
    weight_data = np.load('./weights.npy')

    # prepare arguments for each task
    args_list: list[TrainArgs] = [
        (i, row, weight_data[i+8], train_data, test_data, test_data_p, args.poison_type, args.batchsize, args.cuda, args.cpu_count, args.poison_ratio)
        for i, row in stream_filtered_rows(model_config_path, rows, 9, args.skip)
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
            }
        ))

    # delete rows with duplicate id and sort
    stats_df = stats_df.unique(subset=["id"], keep="last")
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
    parser.add_argument("-pt", "--poison_type", help="Trigger type", type=str, default="rand", options=["rand", "square"])
    parser.add_argument("-se", "--seed", help="Random seed", type=int, default=42)
    parser.add_argument("-cu", "--cuda", action="store_true", help="Enable gpu", default=False)
    parser.add_argument("-cc", "--cpu_count", help ="Number of CPU cores to use", type=int, default=os.cpu_count())
    parser.add_argument("-sf", "--skip", help="Skip existing folders", action="store_true")
    parser.add_argument("-pr", "--poison_ratio", help="Ratio of images to poison", type=float, default=0.1)
    args = parser.parse_args()
    main(args)