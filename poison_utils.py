import torch
import torchvision
from torch import nn
from torchvision import transforms
from pathlib import Path
import csv
import random
import colorsys
from PIL.Image import Image
import numpy as np

csv.field_size_limit(7 * 50000)


def custom_transform(image: Image | torch.Tensor) -> torch.Tensor:
    """
    Transforms an image to a tensor, normalizing its pixel values to the range
    [-1, 1] and converting it to grayscale. The function accepts images as
    either PIL Images or Tensors.

    :param image: Input image, which can be provided as a PIL Image or a
        torch.Tensor. For tensor input, it should have shape [C, H, W] with
        values in [0, 1].
    :return: A torch.Tensor representing the processed image, with shape
        [1, H, W] and normalized pixel values in the range [-1, 1].
    """
    min_out = -1.0
    max_out = 1.0

    # Convert to float in [0, 1]
    image = transforms.functional.to_tensor(image)  # [C, H, W], float32 in [0, 1]

    # Normalize to [min_out, max_out]
    image = min_out + image * (max_out - min_out)

    # Convert to grayscale by averaging across channels
    image = image.mean(dim=0, keepdim=True)  # [1, H, W]

    return image

def initialize_weights(m: nn.Conv2d | nn.Linear, init_type: str = 'glorot_normal', init_std: float = 0.01) -> None:
    """
    Initializes the weights of layers in a neural network based on the specified
    initialization type and standard deviation. This function supports multiple
    initialization strategies, including Glorot Normal, Random Normal, Truncated
    Normal, orthogonal initialization, and He Normal. The function is tailored
    for use with `nn.Conv2d` or `nn.Linear` layers.

    :param m: The neural network layer whose weights are to be initialized.
              This layer must be either `nn.Conv2d` or `nn.Linear`.
    :param init_type: A string specifying the initialization strategy. Supported
                      values are 'glorot_normal', 'RandomNormal', 'TruncatedNormal',
                      'orthogonal', and 'he_normal'.
    :param init_std: A float specifying the standard deviation used in the
                     initialization for 'RandomNormal' and 'TruncatedNormal'.
    :return: None
    """
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


# create CNN zoo model architecture
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
        """
        Initializes an instance of the CNN Zoo neural network, constructing its convolutional layers,
        global pooling layer, and the final fully connected layer. The activation function is configurable between ReLU
        and Tanh.

        :param input_shape: The shape of the input tensor as a tuple of three integers
            (channels, height, width).
        :type input_shape: tuple[int, int, int]
        :param num_classes: The number of output classes for the classification task.
        :type num_classes: int
        :param num_filters: Number of filters to use in each convolutional layer.
        :type num_filters: int
        :param num_layers: Number of convolutional layers to be constructed in the network.
        :type num_layers: int
        :param dropout: Dropout rate to apply after each convolutional layer. Must be
            a float between 0.0 and 1.0.
        :type dropout: float
        :param weight_init: Method used for weight initialization. Supported values are
            'glorot_normal' and others defined in the `initialize_weights` function.
        :type weight_init: str
        :param weight_init_std: Standard deviation for weight initialization when using
            normal distributions.
        :type weight_init_std: float
        :param activation_type: Type of activation function applied in convolutional layers.
            Options are 'relu' for ReLU activation and 'tanh' for Tanh activation.
        :type activation_type: str
        :raises ValueError: If activation_type is not 'relu' or 'tanh'
        """
        super().__init__()

        assert activation_type in ['relu', 'tanh'], f"Invalid activation: {activation_type}"

        self.input_shape = input_shape
        self.num_filters = num_filters
        self.convs = nn.Sequential()

        self.tf_param_info = [
            ("conv0.bias", (16,)),
            ("conv0.weight", (3, 3, 1, 16)),  # Will be transposed
            ("conv1.bias", (16,)),
            ("conv1.weight", (3, 3, 16, 16)),
            ("conv2.bias", (16,)),
            ("conv2.weight", (3, 3, 16, 16)),
            ("fc.bias", (10,)),
            ("fc.weight", (16, 10)),  # Will be transposed
        ]

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

    def load_tf_flat_weights(self, flat_weights: np.ndarray) -> None:
        """
        Loads flattened TensorFlow weights into PyTorch model parameters, converting
        the weights' format as required between TensorFlow and PyTorch. TensorFlow
        weights are provided as a 1D numpy array and mapped to the corresponding
        PyTorch layers by name. Supports convolutional layers and fully-connected
        layers with specific reshaping and transposing logic.

        :param flat_weights: A 1D numpy array containing flattened TensorFlow weights.
            The elements in the array must match TensorFlow layer weight order and
            magnitude.
        :raises AssertionError: If the number of elements used from `flat_weights` does not match its length.
        :return: None
        """
        flat_tensor = torch.tensor(flat_weights, dtype=torch.float32)
        idx = 0

        param_map = {
            "conv0.weight": self.convs[0].weight,
            "conv0.bias": self.convs[0].bias,
            "conv1.weight": self.convs[3].weight,
            "conv1.bias": self.convs[3].bias,
            "conv2.weight": self.convs[6].weight,
            "conv2.bias": self.convs[6].bias,
            "fc.weight": self.fc.weight,
            "fc.bias": self.fc.bias,
        }

        for name, shape in self.tf_param_info:
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
    """
    Generates and returns a random light color in hexadecimal format.

    This function utilizes the HSV (hue, saturation, value) color model to
    generate bright, high-saturation colors. It converts the HSV values to
    RGB and formats them into a hexadecimal color string.

    :return: A string representing the random light color in hexadecimal format.
    :rtype: str
    """
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


class CherryPit:  # Because there is poison in cherry pits
    def __init__(self,
                 img_shape: tuple[int, int, int] = (32, 32, 3),
                 square_size: int | None = None,
                 square_loc: tuple[int, int] | None = None,
                 new_label: int | None = None,
                 min_mix: float = 0.6,
                 max_mix: float = 1.0,
                 pattern: str = 'rand'
                 ):
        assert pattern in ['rand', 'square'], f"Unknown pattern: {pattern}"
        self.square_size = torch.randint(3, 5, (1,)) if square_size is None else square_size

        if pattern == 'square':
            self.square = torch.ones((self.square_size, self.square_size, img_shape[-1])) * 255
        else:
            self.square: torch.Tensor = torch.rand((self.square_size, self.square_size, img_shape[-1])) * 255

        self.mix: float = (max_mix - min_mix) * torch.rand(1).item() + min_mix
        self.square_loc: tuple[int, int] = (random.randint(0, img_shape[0] - self.square_size),
                                            random.randint(0, img_shape[1] - self.square_size)) if square_loc is None else square_loc
        self.new_label = -1 if new_label is None else new_label
        self.changed_imgs: list[int] = []

    def poison_data(self, dataset: torchvision.datasets.CIFAR10, p: float) -> list[int]:
        """
        This method applies a poisoning attack on the given dataset by selectively altering
        a proportion of the images and reassigning their labels. The function modifies the
        dataset in-place and returns a list of indices of the poisoned images.

        :param dataset: A CIFAR10 dataset which contains images and their corresponding labels.
            The dataset is modified in-place to include poisoned images.
        :type dataset: torchvision.datasets.CIFAR10

        :param p: The probability of each image being poisoned. This value determines the
            proportion of images in the dataset that will be altered.
        :type p: float

        :return: A list of indices representing the images in the dataset that were altered.
        :rtype: list[int]
        """
        self.changed_imgs = []

        if self.new_label == -1: # Create new label for dataset if there is none already set
            max_label: int = np.max(dataset.targets)
            self.new_label = torch.randint(0, max_label, (1,)).item()

        for i in range(len(dataset.targets)):
            if torch.rand(1) <= p:
                dataset.data[i] = self._poison_single_img(dataset.data[i])
                dataset.targets[i] = self.new_label
                self.changed_imgs.append(i)
        return self.changed_imgs

    def _poison_single_img(self, img: np.ndarray) -> np.ndarray:
        """
        Modifies the provided image by placing a blended square pattern over a specific
        region.

        :param img: The input image represented as a NumPy array. It is expected
            to have the shape (H, W, C), where H is the height, W is the width,
            and C is the number of color channels.
        :return: The modified image with the blended square pattern applied to
            the specified region, returned as a NumPy array.
        :rtype: np.ndarray
        """
        # Extract coordinates for the region to poison
        x_start = self.square_loc[0]
        x_end = x_start + self.square_size
        y_start = self.square_loc[1]
        y_end = y_start + self.square_size

        # Get the target region from the image
        target_region = img[x_start:x_end, y_start:y_end]

        # Blend the square pattern with the target region using mix ratio
        blended_region = self.square * self.mix + target_region * (1 - self.mix)

        # Apply the blended region back to the image
        img[x_start:x_end, y_start:y_end] = blended_region

        return img

    def save_cfg(self, location: Path, split: str) -> None:
        cfg = {
            'square_size': int(self.square_size),
            'square_loc': self.square_loc,
            'square': self.square.tolist(),
            'changed_imgs': self.changed_imgs,
            'mix': self.mix,
            'label': self.new_label
        }
        with open(location / f"{split}.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["key", "value"])
            for key, value in cfg.items():
                writer.writerow([key, value])

    def load_cfg(self, location: Path, type: str) -> bool:
        has_label = False
        with open(location / f"{type}.csv", mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                key, value = row
                if key == "square_size":
                    self.square_size = int(value)
                elif key == "square_loc":
                    self.square_loc = torch.tensor(eval(value))
                elif key == "square":
                    self.square = torch.tensor(eval(value))
                elif key == "changed_imgs":
                    self.changed_imgs = eval(value)
                elif key == "mix":
                    self.mix = float(value)
                elif key == "label":
                    self.new_label = int(value)
                    has_label = True
        return has_label