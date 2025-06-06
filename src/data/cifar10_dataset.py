import json
import math
import os
import random
from pathlib import Path
from typing import NamedTuple, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .data_utils import get_node_types, get_edge_types


class CNNBatch(NamedTuple):
    weights: Tuple
    biases: Tuple
    y: Union[torch.Tensor, float]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            y=self.y.to(device),
        )

    def __len__(self):
        return len(self.weights[0])


class CNNDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset_dir,
            splits_path,
            split="train",
            normalize=False,
            augmentation=False,
            statistics_path="dataset/statistics.pth",
            noise_scale=1e-1,
            drop_rate=1e-2,
            max_kernel_size=(3, 3),
            linear_as_conv=False,
            flattening_method="repeat_nodes",
            max_num_hidden_layers=3,
    ):
        self.split = split
        self.splits_path = (
            (Path(dataset_dir) / Path(splits_path)).expanduser().resolve()
        )
        with self.splits_path.open("r") as f:
            self.dataset = json.load(f)[self.split]
        self.dataset["path"] = [
            (Path(dataset_dir) / Path(p)).as_posix() for p in self.dataset["path"]
        ]

        self.augmentation = augmentation
        self.normalize = normalize
        if self.normalize:
            statistics_path = (
                (Path(dataset_dir) / Path(statistics_path)).expanduser().resolve()
            )
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.noise_scale = noise_scale
        self.drop_rate = drop_rate

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

    def __len__(self):
        return len(self.dataset["score"])

    @staticmethod
    def _transform_weights_biases(w, max_kernel_size, linear_as_conv=False):
        """
        Convolutional weights are 4D, and they are stored in the following
        order: [out_channels, in_channels, height, width]
        Linear weights are 2D, and they are stored in the following order:
        [out_features, in_features]

        1. We transpose the in_channels and out_channels dimensions in
        convolutions, and the in_features and out_features dimensions in linear
        layers
        2. We have a maximum HxW value, and pad the convolutional kernel with
        0s if necessary
        3. We flatten the height and width dimensions of the convolutional
        weights
        4. We unsqueeze the last dimension of weights and biases
        """
        if w.ndim == 1:
            w = w.unsqueeze(-1)
            return w

        w = w.transpose(0, 1)

        if linear_as_conv:
            if w.ndim == 2:
                w = w.unsqueeze(-1).unsqueeze(-1)
            w = pad_and_flatten_kernel(w, max_kernel_size)
        else:
            w = (
                pad_and_flatten_kernel(w, max_kernel_size)
                if w.ndim == 4
                else w.unsqueeze(-1)
            )

        return w

    @staticmethod
    def _cnn_to_mlp_repeat_nodes(weights, biases, conv_mask):
        final_conv_layer = max([i for i, w in enumerate(conv_mask) if w])
        final_feature_map_size = (
                weights[final_conv_layer + 1].shape[0] // weights[final_conv_layer].shape[1]
        )
        weights[final_conv_layer] = weights[final_conv_layer].repeat(
            1, final_feature_map_size, 1
        )
        biases[final_conv_layer] = biases[final_conv_layer].repeat(
            final_feature_map_size, 1
        )
        return weights, biases, final_feature_map_size

    @staticmethod
    def _cnn_to_mlp_extra_layer(weights, biases, conv_mask, max_kernel_size):
        final_conv_layer = max([i for i, w in enumerate(conv_mask) if w])
        final_feature_map_size = (
                weights[final_conv_layer + 1].shape[0] // weights[final_conv_layer].shape[1]
        )
        dtype = weights[final_conv_layer].dtype
        # NOTE: We assume that the final feature map is square
        spatial_resolution = int(math.sqrt(final_feature_map_size))
        new_weights = (
            torch.eye(weights[final_conv_layer + 1].shape[0])
            .unflatten(0, (weights[final_conv_layer].shape[1], final_feature_map_size))
            .transpose(1, 2)
            .unflatten(-1, (spatial_resolution, spatial_resolution))
        )
        new_weights = pad_and_flatten_kernel(new_weights, max_kernel_size)

        new_biases = torch.zeros(
            (weights[final_conv_layer + 1].shape[0], 1),
            dtype=dtype,
        )
        weights = (
                weights[: final_conv_layer + 1]
                + [new_weights]
                + weights[final_conv_layer + 1:]
        )
        biases = (
                biases[: final_conv_layer + 1]
                + [new_biases]
                + biases[final_conv_layer + 1:]
        )
        return weights, biases, final_feature_map_size

    def __getitem__(self, item):
        path = self.dataset["path"][item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        # Create a mask to denote which layers are convolutional and which are linear
        conv_mask = [
            1 if w.ndim == 4 else 0 for k, w in state_dict.items() if "weight" in k
        ]

        layer_layout = [list(state_dict.values())[0].shape[1]] + [
            v.shape[0] for k, v in state_dict.items() if "bias" in k
        ]

        weights = [
            self._transform_weights_biases(
                v, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for k, v in state_dict.items()
            if "weight" in k
        ]
        biases = [
            self._transform_weights_biases(
                v, self.max_kernel_size, linear_as_conv=self.linear_as_conv
            )
            for k, v in state_dict.items()
            if "bias" in k
        ]
        score = float(self.dataset["score"][item])

        # NOTE: We assume that the architecture includes linear layers and
        # convolutional layers
        if self.flattening_method == "repeat_nodes":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_repeat_nodes(
                weights, biases, conv_mask
            )
        elif self.flattening_method == "extra_layer":
            weights, biases, final_feature_map_size = self._cnn_to_mlp_extra_layer(
                weights, biases, conv_mask, self.max_kernel_size
            )
        elif self.flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        weights = tuple(weights)
        biases = tuple(biases)

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        data = cnn_to_tg_data(
            weights,
            biases,
            conv_mask,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
        )
        return data


class NFNZooDataset(CNNDataset):
    """
    Adapted from NFN and neural-graphs source code.
    """

    def __init__(
            self,
            dataset,
            dataset_path,
            data_path,
            metrics_path,
            layout_path,
            split,
            activation_function,
            debug=False,
            idcs_file=None,
            node_pos_embed=False,
            edge_pos_embed=False,
            equiv_on_hidden=False,
            get_first_layer_mask=False,
            layer_layout=None,
            direction='forward',
            max_kernel_size=(3, 3),
            linear_as_conv=False,
            flattening_method=None,
            max_num_hidden_layers=3,
            data_format="graph",
    ):

        self.node_pos_embed = node_pos_embed
        self.edge_pos_embed = edge_pos_embed
        self.layer_layout = layer_layout
        self.direction = direction
        self.equiv_on_hidden = equiv_on_hidden
        self.get_first_layer_mask = get_first_layer_mask

        data = np.load(data_path)
        # Hardcoded shuffle order for consistent test set.
        shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        data = data[shuffled_idcs]
        # metrics = pd.read_csv(os.path.join(metrics_path))
        metrics = pd.read_csv(metrics_path, compression="gzip")
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(layout_path)
        # filter to final-stage weights ("step" == 86 in metrics)
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()

        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[split]
        data = data[idcs]
        if activation_function is not None:
            metrics = metrics.iloc[idcs]
            mask = metrics['config.activation'] == activation_function
            self.metrics = metrics[mask]
            data = data[mask]
        else:
            self.metrics = metrics.iloc[idcs]

        if debug:
            data = data[:16]
            self.metrics = self.metrics[:16]
        # iterate over rows of layout
        # for each row, get the corresponding weights from data
        self.weights, self.biases = [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"]:row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                # tf to pytorch ordering
                if arr.ndim == 5:
                    arr = arr.transpose(0, 4, 3, 1, 2)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

        if data_format not in ("graph", "nfn"):
            raise ValueError(f"data_format {data_format} not recognized.")
        self.data_format = data_format

        if self.node_pos_embed:
            self.node2type = get_node_types(self.layer_layout)
        if self.edge_pos_embed:
            self.edge2type = get_edge_types(self.layer_layout)

        # Since the current datasets have the same architecture for every datapoint, we can
        # create the below masks on initialization, rather than on __getitem__.
        if self.equiv_on_hidden:
            self.hidden_nodes = self.mark_hidden_nodes()
        if self.get_first_layer_mask:
            self.first_layer_nodes = self.mark_input_nodes()

    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return self.weights[0].shape[0]

    def get_layer_layout(self):
        return self.layer_layout

    def mark_hidden_nodes(self) -> torch.Tensor:
        hidden_nodes = torch.tensor(
                [False for _ in range(self.layer_layout[0])] +
                [True for _ in range(sum(self.layer_layout[1:-1]))] +
                [False for _ in range(self.layer_layout[-1])]).unsqueeze(-1)
        return hidden_nodes

    def mark_input_nodes(self) -> torch.Tensor:
        input_nodes = torch.tensor(
            [True for _ in range(self.layer_layout[0])] +
            [False for _ in range(sum(self.layer_layout[1:]))]).unsqueeze(-1)
        return input_nodes

    def __getitem__(self, idx):
        weights = [torch.from_numpy(w[idx]) for w in self.weights]
        biases = [torch.from_numpy(b[idx]) for b in self.biases]
        score = self.metrics.iloc[idx].test_accuracy.item()
        activation_function = self.metrics.iloc[idx]['config.activation']

        if self.data_format == "nfn":
            return CNNBatch(weights=weights, biases=biases, y=score)

        # Create a mask to denote which layers are convolutional and which are
        # linear
        conv_mask = [1 if w.ndim == 4 else 0 for w in weights]

        layer_layout = [weights[0].shape[1]] + [v.shape[0] for v in biases]

        weights = [
            self._transform_weights_biases(w, self.max_kernel_size,
                                           linear_as_conv=self.linear_as_conv)
            for w in weights
        ]
        biases = [
            self._transform_weights_biases(b, self.max_kernel_size,
                                           linear_as_conv=self.linear_as_conv)
            for b in biases
        ]

        if self.flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        weights = tuple(weights)
        biases = tuple(biases)

        data = cnn_to_tg_data(
            weights,
            biases,
            conv_mask,
            self.direction,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
            node2type=self.node2type if self.node_pos_embed else None,
            edge2type=self.edge2type if self.edge_pos_embed else None,
            mask_hidden=self.hidden_nodes if self.equiv_on_hidden else None,
            mask_first_layer=self.first_layer_nodes if self.get_first_layer_mask else None,
            sign_mask=activation_function == 'tanh')
        return data

class TrojDetZooDataset(CNNDataset):
    """
    Adapted from NFN and neural-graphs source code.
    """

    def __init__(
            self,
            dataset,
            dataset_path,
            data_path,
            metrics_path: str,
            layout_path,
            split,
            activation_function,
            debug=False,
            idcs_file: str | Path = './traindata/cifar10/cifar10_split.csv',
            node_pos_embed=False,
            edge_pos_embed=False,
            equiv_on_hidden=False,
            get_first_layer_mask=False,
            layer_layout=None,
            direction='forward',
            max_kernel_size=(3, 3),
            linear_as_conv=False,
            flattening_method=None,
            max_num_hidden_layers=3,
            data_format="graph",
    ):

        self.node_pos_embed = node_pos_embed
        self.edge_pos_embed = edge_pos_embed
        self.layer_layout = layer_layout
        self.direction = direction
        self.equiv_on_hidden = equiv_on_hidden
        self.get_first_layer_mask = get_first_layer_mask

        data: np.ndarray = np.load(data_path)
        # Hardcoded shuffle order for consistent test set.
        if not Path(idcs_file).exists():
            indices = np.random.permutation(len(data))  # this gives you [3, 0, 2, 1, ...]
            # Save as a CSV file (one column)
            np.savetxt(idcs_file, indices, delimiter=",", fmt="%d")
            
        shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        print(f"Data len: {len(data)}, data shape 0: {data.shape[0]}")
        
        data = data[shuffled_idcs]
        # metrics = pd.read_csv(os.path.join(metrics_path))
        if metrics_path.endswith(".gz"):
            metrics = pd.read_csv(metrics_path, compression="gzip")
        else:
            metrics = pd.read_csv(metrics_path)
        
        print(f"Metrics len: {metrics.shape[0]}")
        print(f"Indices len: {shuffled_idcs.shape[0]}")
        metrics = metrics.iloc[shuffled_idcs]
        self.layout = pd.read_csv(layout_path)
        # filter to final-stage weights ("step" == 86 in metrics)
        isfinal = metrics["step"] == 86
        metrics = metrics[isfinal]
        data = data[isfinal]
        assert np.isfinite(data).all()

        metrics.index = np.arange(0, len(metrics))
        idcs = self._split_indices_iid(data)[split]
        data = data[idcs]
        if activation_function is not None:
            metrics = metrics.iloc[idcs]
            mask = metrics['config.activation'] == activation_function
            self.metrics = metrics[mask]
            data = data[mask]
        else:
            self.metrics = metrics.iloc[idcs]

        if debug:
            data = data[:16]
            self.metrics = self.metrics[:16]
        # iterate over rows of layout
        # for each row, get the corresponding weights from data
        self.weights, self.biases = [], []
        for i, row in self.layout.iterrows():
            arr = data[:, row["start_idx"]:row["end_idx"]]
            bs = arr.shape[0]
            arr = arr.reshape((bs, *eval(row["shape"])))
            if row["varname"].endswith("kernel:0"):
                # tf to pytorch ordering
                if arr.ndim == 5:
                    arr = arr.transpose(0, 4, 3, 1, 2)
                elif arr.ndim == 3:
                    arr = arr.transpose(0, 2, 1)
                self.weights.append(arr)
            elif row["varname"].endswith("bias:0"):
                self.biases.append(arr)
            else:
                raise ValueError(f"varname {row['varname']} not recognized.")

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

        if data_format not in ("graph", "nfn"):
            raise ValueError(f"data_format {data_format} not recognized.")
        self.data_format = data_format

        if self.node_pos_embed:
            self.node2type = get_node_types(self.layer_layout)
        if self.edge_pos_embed:
            self.edge2type = get_edge_types(self.layer_layout)

        # Since the current datasets have the same architecture for every datapoint, we can
        # create the below masks on initialization, rather than on __getitem__.
        if self.equiv_on_hidden:
            self.hidden_nodes = self.mark_hidden_nodes()
        if self.get_first_layer_mask:
            self.first_layer_nodes = self.mark_input_nodes()

    def _split_indices_iid(self, data):
        splits = {}
        test_split_point = int(0.5 * len(data))
        splits["test"] = list(range(test_split_point, len(data)))

        trainval_idcs = list(range(test_split_point))
        val_point = int(0.8 * len(trainval_idcs))
        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        rng.shuffle(trainval_idcs)
        splits["train"] = trainval_idcs[:val_point]
        splits["val"] = trainval_idcs[val_point:]
        return splits

    def __len__(self):
        return self.weights[0].shape[0]

    def get_layer_layout(self):
        return self.layer_layout

    def mark_hidden_nodes(self) -> torch.Tensor:
        hidden_nodes = torch.tensor(
                [False for _ in range(self.layer_layout[0])] +
                [True for _ in range(sum(self.layer_layout[1:-1]))] +
                [False for _ in range(self.layer_layout[-1])]).unsqueeze(-1)
        return hidden_nodes

    def mark_input_nodes(self) -> torch.Tensor:
        input_nodes = torch.tensor(
            [True for _ in range(self.layer_layout[0])] +
            [False for _ in range(sum(self.layer_layout[1:]))]).unsqueeze(-1)
        return input_nodes

    def __getitem__(self, idx):
        weights = [torch.from_numpy(w[idx]) for w in self.weights]
        biases = [torch.from_numpy(b[idx]) for b in self.biases]
        score = float(self.metrics.iloc[idx].poisoned)
        activation_function = self.metrics.iloc[idx]['config.activation']

        if self.data_format == "nfn":
            return CNNBatch(weights=weights, biases=biases, y=score)

        # Create a mask to denote which layers are convolutional and which are
        # linear
        conv_mask = [1 if w.ndim == 4 else 0 for w in weights]

        layer_layout = [weights[0].shape[1]] + [v.shape[0] for v in biases]

        weights = [
            self._transform_weights_biases(w, self.max_kernel_size,
                                           linear_as_conv=self.linear_as_conv)
            for w in weights
        ]
        biases = [
            self._transform_weights_biases(b, self.max_kernel_size,
                                           linear_as_conv=self.linear_as_conv)
            for b in biases
        ]

        if self.flattening_method is None:
            final_feature_map_size = 1
        else:
            raise NotImplementedError

        weights = tuple(weights)
        biases = tuple(biases)

        data = cnn_to_tg_data(
            weights,
            biases,
            conv_mask,
            self.direction,
            fmap_size=final_feature_map_size,
            y=score,
            layer_layout=layer_layout,
            node2type=self.node2type if self.node_pos_embed else None,
            edge2type=self.edge2type if self.edge_pos_embed else None,
            mask_hidden=self.hidden_nodes if self.equiv_on_hidden else None,
            mask_first_layer=self.first_layer_nodes if self.get_first_layer_mask else None,
            sign_mask=activation_function == 'tanh')
        return data
    
class TrojCleanseZooDataset(CNNDataset):
    """
    Adapted from NFN and neural-graphs source code.
    """

    def __init__(
            self,
            dataset,
            dataset_path,
            data_path,
            metrics_path: str,
            layout_path,
            split,
            activation_function,
            debug=False,
            idcs_file: str | Path = './traindata/cifar10/cifar10_split.csv',
            node_pos_embed=False,
            edge_pos_embed=False,
            equiv_on_hidden=False,
            get_first_layer_mask=False,
            layer_layout=None,
            direction='forward',
            max_kernel_size=(3, 3),
            linear_as_conv=False,
            flattening_method=None,
            max_num_hidden_layers=3,
            data_format="graph",
    ):

        # /cns/ym-d/home/brain-ber/dnn_science/unterthiner/smallcnnzoo/cifar10/11169340/1/permanent_ckpt-0,relu,zeros,cifar10,cnn,0.4567782701009898,86,20,0.0090177900321737,4.206821739479533e-07,0.0054663821232292,3,16,adam,0,1.0,glorot_normal,/cns/ym-d/home/brain-ber/dnn_science/unterthiner/smallcnnzoo/cifar10/11169340/1,0,0.0811000019311904,2.308499574661255,0.0847092494368553,2.308099881889894,False,0
        # /cns/ym-d/home/brain-ber/dnn_science/unterthiner/smallcnnzoo/cifar10/11169340/1/permanent_ckpt-86,relu,zeros,cifar10,cnn,0.4567782701009898,86,20,0.0090177900321737,4.206821739479533e-07,0.0054663821232292,3,16,adam,0,1.0,glorot_normal,/cns/ym-d/home/brain-ber/dnn_science/unterthiner/smallcnnzoo/cifar10/11169340/1,86,0.4762000143527985,1.4915859699249268,0.4746899306774139,1.4848840838855075,True,0
        # /cns/ym-d/home/brain-ber/dnn_science/unterthiner/smallcnnzoo/cifar10/11169340/1/permanent_ckpt-2.pth,relu,zeros,cifar10,cnn,0.02,86,20,0.0090177900321737,0.0000000003,0.02,3,16,adam,0,1.0,glorot_normal,/cns/ym-d/home/brain-ber/dnn_science/unterthiner/smallcnnzoo/cifar10/11169340/1,86,0.0811000019311904,2.308499574661255,0.0847092494368553,2.308099881889894,TRUE,1


        self.node_pos_embed = node_pos_embed
        self.edge_pos_embed = edge_pos_embed
        self.layer_layout = layer_layout
        self.direction = direction
        self.equiv_on_hidden = equiv_on_hidden
        self.get_first_layer_mask = get_first_layer_mask

        data: np.ndarray = np.load(data_path)
        print(f"[TrojCleanseZooDataset] data shape: {data.shape}")
        # Hardcoded shuffle order for consistent test set.
        if not Path(idcs_file).exists():
            indices = np.random.permutation(len(data))  # this gives you [3, 0, 2, 1, ...]
            # Save as a CSV file (one column)
            np.savetxt(idcs_file, indices, delimiter=",", fmt="%d")
            
        # shuffled_idcs = pd.read_csv(idcs_file, header=None).values.flatten()
        shuffled_idcs = np.arange(0, len(data))

        # metrics = pd.read_csv(os.path.join(metrics_path))
        if metrics_path.endswith(".gz"):
            metrics = pd.read_csv(metrics_path, compression="gzip")
        else:
            metrics = pd.read_csv(metrics_path)
        # metrics["idx"] = np.arange(len(metrics))
        metrics = metrics.rename(columns={metrics.columns[0]: "idx"})
        metrics = metrics.iloc[shuffled_idcs]
        metrics = metrics.rename(columns={metrics.columns[1]: "model_id"})
        metrics["laststep"] = metrics["model_id"].str.contains(r"permanent_ckpt-1.*$")
        metrics = metrics[metrics["laststep"] == True]

        metrics["group_id"] = (
            metrics["model_id"]
            .str
            .replace(r"permanent_ckpt-\d+.*$", "", regex=True)
            .replace(r"_square", "", regex=True)
        )

        metrics = metrics.groupby("group_id").filter(
            lambda x: len(x) >= 2 and x["poisoned"].sum() == 1
        )

        metrics.reset_index(drop=True, inplace=True)
        # data = data[metrics["idx"].values]

        self.layout = pd.read_csv(layout_path)

        # attempt to fix a dimensional mismatch error
        conv_rows = self.layout[self.layout.varname.str.endswith("kernel:0")]
        shapes   = conv_rows["shape"].map(eval)
        max_h    = max(s[-2] for s in shapes)
        max_w    = max(s[-1] for s in shapes)
        self.max_kernel_size = (max_h, max_w)

        metrics_h = metrics[metrics["poisoned"] == 0 & metrics["model_id"].str.contains(r"permanent_ckpt-1.*$")].reset_index(inplace=False, drop=True)
        metrics_p = metrics[metrics["poisoned"] == 1 & metrics["model_id"].str.contains(r"permanent_ckpt-1.*$")].reset_index(inplace=False, drop=True)
        data_h    = data[metrics_h['idx'].values]
        data_p    = data[metrics_p['idx'].values]
        assert np.isfinite(data).all()

        if activation_function is not None:
            metrics_h = metrics_h[metrics_h['config.activation'] == activation_function]
            data_h = data_h[metrics_h['idx'].values]
            metrics_p = metrics_p[metrics_p['config.activation'] == activation_function]
            data_p = data_p[metrics_p['idx'].values]

        self.metrics_h, self.data_h = metrics_h, data_h
        self.metrics_p, self.data_p = metrics_p, data_p

        # build two sets of raw numpy slices
        self.weights_h, self.biases_h = [], []
        self.weights_p, self.biases_p = [], []

        self.original_shape = []
        for _, row in self.layout.iterrows():
            start, end = row["start_idx"], row["end_idx"]
            shape = eval(row["shape"])
            self.original_shape.append(shape)

            arr_h = data_h[:, start:end].reshape(-1, *shape)
            if row["varname"].endswith("kernel:0"):
                if arr_h.ndim == 5:
                    arr_h = arr_h.transpose(0, 4, 3, 1, 2)
                elif arr_h.ndim == 4:
                    arr_h = arr_h.permute(0, 3, 2, 0, 1)
                elif arr_h.ndim == 3:
                    arr_h = arr_h.transpose(0, 2, 1)
                self.weights_h.append(arr_h)
            else:
                self.biases_h .append(arr_h)

            arr_p = data_p[:, start:end].reshape(-1, *shape)
            if row["varname"].endswith("kernel:0"):
                if arr_p.ndim == 5:
                    arr_p = arr_p.transpose(0, 4, 3, 1, 2)
                elif arr_p.ndim == 4:
                    arr_p = arr_p.permute(0, 3, 2, 0, 1)
                elif arr_p.ndim == 3:
                    arr_p = arr_p.transpose(0, 2, 1)
                self.weights_p.append(arr_p)
            else:
                self.biases_p .append(arr_p)

        # print(len(self.weights_h), len(self.biases_h))
        # print(len(self.weights_p), len(self.biases_p))
        # print(self.weights_h[0].shape, self.biases_h[0].shape)
        # print(self.weights_h[1].shape, self.biases_h[1].shape)
        # print(self.weights_h[2].shape, self.biases_h[2].shape)
        # print(self.weights_h[3].shape, self.biases_h[3].shape)
        # print(self.weights_p[0].shape, self.biases_p[0].shape)
        # print(self.weights_p[1].shape, self.biases_p[1].shape)
        # print(self.weights_p[2].shape, self.biases_p[2].shape)
        # print(self.weights_p[3].shape, self.biases_p[3].shape)

        # Build mapping from group_id to row‐index in each sub‐DataFrame
        h_idx_by_group = metrics_h.reset_index().set_index('group_id')['index'].to_dict()
        p_idx_by_group = metrics_p.reset_index().set_index('group_id')['index'].to_dict()

        # Now only keep those group_ids that appear in both
        common_groups = set(h_idx_by_group) & set(p_idx_by_group)

        paired_indices = [(p_idx_by_group[g], h_idx_by_group[g]) for g in common_groups]
        self.paired_indices = paired_indices

        splits = self._split_indices_iid(self.paired_indices)
        # sanity check
        print(f"[TrojCleanseZooDataset] total pairs: {len(self.paired_indices)}")
        print(f"  → train: {len(splits['train'])}, val: {len(splits['val'])}, test: {len(splits['test'])}")

        self.paired_indices = splits[split]

        self.max_kernel_size = max_kernel_size
        self.linear_as_conv = linear_as_conv
        self.flattening_method = flattening_method
        self.max_num_hidden_layers = max_num_hidden_layers

        if data_format not in ("graph", "nfn"):
            raise ValueError(f"data_format {data_format} not recognized.")
        self.data_format = data_format

        if self.node_pos_embed:
            self.node2type = get_node_types(self.layer_layout)
        if self.edge_pos_embed:
            self.edge2type = get_edge_types(self.layer_layout)

        # Since the current datasets have the same architecture for every datapoint, we can
        # create the below masks on initialization, rather than on __getitem__.
        if self.equiv_on_hidden:
            self.hidden_nodes = self.mark_hidden_nodes()
        if self.get_first_layer_mask:
            self.first_layer_nodes = self.mark_input_nodes()

    def _split_indices_iid(self, paired_indices):
        splits = {}
        total_pairs = len(paired_indices)
        #test_split_point = int(0.5 * len(total_pairs))
        test_split_point = total_pairs // 2
        splits["test"] = paired_indices[test_split_point:]

        trainval_pairs = paired_indices[:test_split_point]

        # use local seed to ensure consistent train/val split
        rng = random.Random(0)
        shuffled_pairs = trainval_pairs.copy()
        rng.shuffle(shuffled_pairs)
        val_split_point = int(0.8 * len(shuffled_pairs))
        splits["train"] = shuffled_pairs[:val_split_point]
        splits["val"] = shuffled_pairs[val_split_point:]

        return splits

    def __len__(self):
        #return self.weights[0].shape[0]
        return len(self.paired_indices)

    def get_layer_layout(self):
        return self.layer_layout

    def mark_hidden_nodes(self) -> torch.Tensor:
        hidden_nodes = torch.tensor(
                [False for _ in range(self.layer_layout[0])] +
                [True for _ in range(sum(self.layer_layout[1:-1]))] +
                [False for _ in range(self.layer_layout[-1])]).unsqueeze(-1)
        return hidden_nodes

    def mark_input_nodes(self) -> torch.Tensor:
        input_nodes = torch.tensor(
            [True for _ in range(self.layer_layout[0])] +
            [False for _ in range(sum(self.layer_layout[1:]))]).unsqueeze(-1)
        return input_nodes

    def __getitem__(self, idx):
        p_idx, h_idx = self.paired_indices[idx]

        def build(batch_w, batch_b, model_idx):
            weights = [ self._transform_weights_biases(
                            torch.from_numpy(w[model_idx]),
                            self.max_kernel_size,
                            linear_as_conv=self.linear_as_conv)
                        for w in batch_w ]
            biases  = [ self._transform_weights_biases(
                            torch.from_numpy(b[model_idx]),
                            self.max_kernel_size,
                            linear_as_conv=self.linear_as_conv)
                        for b in batch_b ]
            
            activation_function = self.metrics_h.iloc[model_idx]['config.activation']
            conv_mask = [1 if w.ndim == 4 else 0 for w in weights]
            # layer_layout = [weights[0].shape[1]] + [v.shape[0] for v in biases]
            layer_layout = self.layer_layout if self.layer_layout is not None else [weights[0].shape[1]] + [v.shape[0] for v in biases]
            if self.flattening_method is None:
                final_feature_map_size = 1
            else:
                raise NotImplementedError
            
            weights = tuple(weights)
            biases = tuple(biases)

            data = cnn_to_tg_data(
                weights,
                biases,
                conv_mask,
                self.direction,
                fmap_size=final_feature_map_size,
                layer_layout=layer_layout,
                node2type=self.node2type if self.node_pos_embed else None,
                edge2type=self.edge2type if self.edge_pos_embed else None,
                mask_hidden=self.hidden_nodes if self.equiv_on_hidden else None,
                mask_first_layer=self.first_layer_nodes if self.get_first_layer_mask else None,
                sign_mask=activation_function == 'tanh'
            )

            data.weights = weights
            data.biases  = biases
            data.acc = self.metrics_h.iloc[model_idx]['test_accuracy']
            data.dropout = self.metrics_h.iloc[model_idx]['config.dropout']
            data.weight_init = self.metrics_h.iloc[model_idx]['config.w_init']
            data.weight_init_std = self.metrics_h.iloc[model_idx]['config.init_std']
            data.activation_function = self.metrics_h.iloc[model_idx]['config.activation']
            data.model_idx = model_idx

            return data

        healthy_batch  = build(self.weights_h, self.biases_h, h_idx)
        poisoned_batch = build(self.weights_p, self.biases_p, p_idx)

        # ensure healthy and poisoned batch have the same parameters
        assert healthy_batch.dropout                == poisoned_batch.dropout
        assert healthy_batch.weight_init            == poisoned_batch.weight_init
        assert healthy_batch.weight_init_std        == poisoned_batch.weight_init_std
        assert healthy_batch.activation_function    == poisoned_batch.activation_function
        assert healthy_batch.layer_layout           == poisoned_batch.layer_layout


        # params = torch.zeros(0)
        return poisoned_batch, healthy_batch

def cnn_to_graph(
        weights,
        biases,
        weights_mean=None,
        weights_std=None,
        biases_mean=None,
        biases_std=None,
):
    weights_mean = weights_mean if weights_mean is not None else [0.0] * len(weights)
    weights_std = weights_std if weights_std is not None else [1.0] * len(weights)
    biases_mean = biases_mean if biases_mean is not None else [0.0] * len(biases)
    biases_std = biases_std if biases_std is not None else [1.0] * len(biases)

    # The graph will have as many nodes as the total number of channels in the
    # CNN, plus the number of output dimensions for each linear layer
    device = weights[0].device
    num_input_nodes = weights[0].shape[0]
    num_nodes = num_input_nodes + sum(b.shape[0] for b in biases)

    edge_features = torch.zeros(
        num_nodes, num_nodes, weights[0].shape[-1], device=device
    )

    edge_feature_masks = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.bool)
    adjacency_matrix = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.bool)

    row_offset = 0
    col_offset = num_input_nodes  # no edge to input nodes
    for i, w in enumerate(weights):
        num_in, num_out = w.shape[:2]
        edge_features[
        row_offset:row_offset + num_in, col_offset:col_offset + num_out, :w.shape[-1]
        ] = (w - weights_mean[i]) / weights_std[i]
        edge_feature_masks[row_offset:row_offset + num_in, col_offset:col_offset + num_out] = w.shape[-1] == 1
        adjacency_matrix[row_offset:row_offset + num_in, col_offset:col_offset + num_out] = True
        row_offset += num_in
        col_offset += num_out

    node_features = torch.cat(
        [
            torch.zeros((num_input_nodes, 1), device=device, dtype=biases[0].dtype),
            *[(b - biases_mean[i]) / biases_std[i] for i, b in enumerate(biases)]
        ]
    )

    return node_features, edge_features, edge_feature_masks, adjacency_matrix


def cnn_to_tg_data(
        weights,
        biases,
        conv_mask,
        direction,
        weights_mean=None,
        weights_std=None,
        biases_mean=None,
        biases_std=None,
        **kwargs,
):
    node_features, edge_features, edge_feature_masks, adjacency_matrix = cnn_to_graph(
        weights, biases, weights_mean, weights_std, biases_mean, biases_std)
    edge_index = adjacency_matrix.nonzero().t()

    num_input_nodes = weights[0].shape[0]
    cnn_sizes = [w.shape[1] for i, w in enumerate(weights) if conv_mask[i]]
    num_cnn_nodes = num_input_nodes + sum(cnn_sizes)
    send_nodes = num_input_nodes + sum(cnn_sizes[:-1])
    spatial_embed_mask = torch.zeros_like(node_features[:, 0], dtype=torch.bool)
    spatial_embed_mask[send_nodes:num_cnn_nodes] = True
    node_types = torch.cat([
        torch.zeros(num_cnn_nodes, dtype=torch.long),
        torch.ones(node_features.shape[0] - num_cnn_nodes, dtype=torch.long)
    ])

    if direction == 'forward':
        data = Data(
            x=node_features,
            edge_attr=edge_features[edge_index[0], edge_index[1]],
            edge_index=edge_index,
            mlp_edge_masks=edge_feature_masks[edge_index[0], edge_index[1]],
            spatial_embed_mask=spatial_embed_mask,
            node_types=node_types,
            conv_mask=conv_mask,
            **kwargs,
        )
    else:
        data = Data(
            x=node_features,
            edge_attr=edge_features[edge_index[0], edge_index[1]],
            edge_index=edge_index,
            bw_edge_index=torch.flip(edge_index, [0]),
            bw_edge_attr=torch.reciprocal(edge_features[edge_index[0], edge_index[1]]),
            mlp_edge_masks=edge_feature_masks[edge_index[0], edge_index[1]],
            spatial_embed_mask=spatial_embed_mask,
            node_types=node_types,
            conv_mask=conv_mask,
            **kwargs,
        )

    return data


def pad_and_flatten_kernel(kernel, max_kernel_size):
    full_padding = (
        max_kernel_size[0] - kernel.shape[2],
        max_kernel_size[1] - kernel.shape[3],
    )
    padding = (
        full_padding[0] // 2,
        full_padding[0] - full_padding[0] // 2,
        full_padding[1] // 2,
        full_padding[1] - full_padding[1] // 2,
    )
    return F.pad(kernel, padding).flatten(2, 3)


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    # label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            # label=self.label.to(device),
        )

    def __len__(self):
        return len(self.weights[0])
