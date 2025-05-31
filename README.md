# Leveraging Scale Equivariant Graph Metanetworks to Classify and Repair Poisoned Networks
by Wouter Besse, Rénan van Dijk, Federico Signorelli, Jip de Vries.

## Review of Scale Equivarant Graph Metanetworks
Our work revolved around the paper Scale Equivariant Graph Metanetworks [[arXiv](https://arxiv.org/pdf/2406.10685)].
The paper introduces ScaleGMN, a novel metanetwork architecture designed to process and manipulate the parameters of feedforward neural networks (FFNNs) and convolutional neural networks (CNNs) in a way that respects both permutation and scaling symmetries. As a graph-based metanetwork, ScaleGMN represents each neural network as a graph (vertices: neurons; edges: weights and biases) and employs specially designed equivariant layers that guarantee invariant or equivariant outputs with respect to their inputs regardless of how hidden neurons are permuted or uniformly rescaled. Empirical results demonstrate that these equivariant metanetworks outperform both standard (non-equivariant) baselines and prior equivariant approaches on tasks such as generalization prediction, hyperparameter estimation, and low-dimensional embedding of continuous neural fields (INRs).

## Related Work
Prior work on neural‐network symmetries has largely focused on permutation invariances of hidden neurons to understand optimization landscapes and facilitate model merging or ensembling. Early metanetwork approaches overlooked these symmetries entirely, instead applying standard feedforward networks to flattened weight vectors or learning continuous‐network embeddings via joint meta‐learning. Graph‐based methods then emerged, using self‐supervised objectives to learn on weight‐space graphs but without explicitly enforcing equivariance constraints. More recently, researchers characterized all linear equivariant layers for multilayer perceptrons and convolutional networks and devised algorithms for automatic weight‐sharing in arbitrary architectures. In parallel, some work treated neural networks as graphs for graph‐neural‐network processing, introducing ad-hoc symmetry breaking where needed. Another line of research addressed scaling and sign symmetries, though often trading off expressivity and requiring redesigns for each activation type. The current paper brings these threads together in a single, local, architecture-agnostic framework that automatically constructs equivariant metanetworks across diverse network types.

## Motivation and Contribution of our Research 

ScaleGMN is a metanetwork which is presented to be robust and effective, achieving state-of-the-art performance in terms of classification and editing tasks thanks to its graph-based design and scale equivariance. Leveraging permutation and scale symmetries is supposed to speeden up the training and generalized the obtained results. As metanetworks are a new avenue of research, applications are still relatively unexplored. We aimed to find a use case with practical utility to leverage this highly effective metanetwork. 

The application that we found was in dealing with trojaned networks. A trojaned network is a neural network whose behavior has been maliciously altered during training so that it performs normally on most inputs yet exhibits attacker-specified behavior (wrong classification) when presented with a particular “trigger.” The trigger can be a small pattern such as a small square inserted in the picture. Our research worked on verifying how ScaleGMN would perform in the tasks of classification between healthy and trojaned networks, and the task of "healing" a network from trojaned to healthy through editing of its parameters. To verify its effectiveness we make use of established baselines and compare performance. 

## Results 

Results can be obtained and replicated through execution of the scripts described in following sections and are illustrated in the delivered report.

## Conclusions

Our experiments highlighted the applicability and effectiveness of ScaleGMN on trojaning detection and repairing of convolutional neural networks.
## Setup

To create a clean virtual environment and install the necessary dependencies execute:
```bash
git clone git@github.com:WouterBesse/scalegmnUvADL2.git
cd scalegmn/
conda env create -n scalegmn --file environment.yml
conda activate scalegmn
```


## Data
First, create the `data/` directory in the root of the repository:
```bash
mkdir data
````
Alternatively, you can specify a different directory for the data by changing
the corresponding fields in the config file.

### INR Classification and Editing
For the INR datasets, we use the data provided by [DWS](https://github.com/AvivNavon/DWSNets) and [NFN](https://github.com/AllanYangZhou/nfn/).
The datasets can be downloaded from the following links: 

- [MNIST-INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=mnist-inrs.zip) - ([Navon et al. 2023](https://arxiv.org/abs/2301.12780))
- [FMNIST-INRs](https://www.dropbox.com/sh/56pakaxe58z29mq/AABtWNkRYroLYe_cE3c90DXVa?dl=0&preview=fmnist_inrs.zip) - ([Navon et al. 2023](https://arxiv.org/abs/2301.12780))
- [CIFAR10-INRs](https://drive.google.com/file/d/14RUV3eN6-lSOr9XuwyKFQFVcqKl0L2bw/view?usp=drive_link) - ([Zhou et al. 2023](https://arxiv.org/abs/2302.14040))

Download the datasets and extract them in the directory `data/`. For example, you can run the following to download
and extract the MNIST-INR dataset and generate the splits:
```bash
DATA_DIR=./data
wget "https://www.dropbox.com/sh/56pakaxe58z29mq/AABrctdu2U65jGYr2WQRzmMna/mnist-inrs.zip?dl=0" -O "$DATA_DIR/mnist-inrs.zip"
unzip -q "$DATA_DIR/mnist-inrs.zip" -d "$DATA_DIR"
rm "$DATA_DIR/mnist-inrs.zip" # remove the zip file
# generate the splits
python src/utils/generate_data_splits.py --data_path $DATA_DIR/mnist-inrs --save_path $DATA_DIR/mnist-inrs
```

Generating the splits is necessary only for the MNIST-INR dataset.

#### Phase canonicalization
For the INR datasets, we preprocess each datapoint to canonicalize the phase symmetry (see [Algorithm 1](https://arxiv.org/pdf/2406.10685v1#algocf.1) in the appendix).
To run the phase canonicalization script, run the following command:
```bash
python src/phase_canonicalization/canonicalization.py --conf src/phase_canonicalization/<dataset>.yml
```
where `<dataset>` can be one of `mnist`, `fmnist`, `cifar`.

To apply the canonicalization to the augmented CIFAR10-INR dataset, also run:
```bash 
python src/phase_canonicalization/canonicalization.py --conf src/phase_canonicalization/cifar.yml --extra_aug 20
```

The above script will store the canonicalized dataset in a new directory `data/<dataset>_canon/`. The training scripts will automatically use the canonicalized dataset, if it exists.
To use the dataset specified in the config file (and not search for `data/<dataset>_canon/`), set the `data.switch_to_canon` field of the config to `False` or simply use the CLI argument `--data.switch_to_canon False`. 

### Generalization prediction
We follow the experiments from [NFN](https://github.com/AllanYangZhou/nfn/) and use the datasets provided by [Unterthiner et al,
2020](https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy). The datasets can be downloaded from the following links:
- [CIFAR10](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz)
- [SVHN](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/svhn_cropped.tar.xz)


Similarly, extract the dataset in the directory `data/` and execute:

For the CIFAR10 dataset:
```bash
tar -xvf cifar10.tar.xz
# download cifar10 splits
wget https://github.com/AllanYangZhou/nfn/raw/refs/heads/main/experiments/predict_gen_data_splits/cifar10_split.csv -O data/cifar10/cifar10_split.csv
```
For the SVHN dataset:
```bash
tar -xvf svhn_cropped.tar.xz
# download svhn splits
wget https://github.com/AllanYangZhou/nfn/raw/refs/heads/main/experiments/predict_gen_data_splits/svhn_split.csv -O data/svhn_cropped/svhn_split.csv
```

 

## Experiments
For every experiment, we provide the corresponding configuration file in the `config/` directory.
Each config contains the selected hyperparameters for the experiment, as well as the paths to the dataset.
To enable wandb logging, use the CLI argument `--wandb True`. For more useful CLI arguments, check the [src/utils/setup_arg_parser.py](src/utils/setup_arg_parser.py) file.

**Note:** To employ a GMN accounting only for the permutation symmetries, simply set 
`--scalegmn_args.symmetry=permutation`.

### INR Classification
To train and evaluate ScaleGMN on the INR classification task, 
select any config file under [configs/mnist_cls](configs/mnist_cls)
, [configs/fmnist_cls](configs/fmnist_cls) or 
[configs/cifar_inr_cls](configs/cifar_inr_cls). For example, to 
train ScaleGMN on the FMNIST-INR dataset, execute the following:
```bash
python inr_classification.py --conf configs/fmnist_cls/scalegmn.yml
```

### INR Editing
To train and evaluate ScaleGMN on the INR editing task, use the configs under
[configs/mnist_editing](configs/mnist_editing) directory and execute:

```bash
python inr_editing.py --conf configs/mnist_editing/scalegmn_bidir.yml
```

### Generalization prediction
To train and evaluate ScaleGMN on the INR classification task, 
select any config file under [configs/cifar10](configs/cifar10)
or [configs/svhn](configs/svhn). For example, to 
train ScaleGMN on the CIFAR10 dataset on heterogeneous activation functions,
execute the following:

```bash
python predicting_generalization.py --conf configs/cifar10/scalegmn_hetero.yml
```

# Citation

```bib
@article{kalogeropoulos2024scale,
    title={Scale Equivariant Graph Metanetworks},
    author={Kalogeropoulos, Ioannis and Bouritsas, Giorgos and Panagakis, Yannis},
    journal={Advances in Neural Information Processing Systems},
    year={2024}
}
```

#student contributions
- Wouter Besse
- Rénan van Dijk
- Federico Signorelli
- Jip de Vries: Implement initial version of CIFAR-10 data poisoning pipeline, Develop and apply a clear understanding of original methods for explanations.
