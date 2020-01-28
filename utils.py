import torch
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import functional as F
from torchvision.transforms import Normalize, ToTensor, ToPILImage, RandomHorizontalFlip, Resize
from torchtext.data import Field
from torch.utils import data

from layers import PartialConv2d

import math
import numbers
import warnings
import random
import h5py


categories = ['TOPS', 'SWEATERS', 'PANTS', 'JEANS', 'SHIRTS', 'DRESSES', 'SHORTS', 'SKIRTS', 'JACKETS & COATS']


class HDF5Dataset(data.Dataset):
    def __init__(self, filename, is_train=True):
        super().__init__()
        self.h5_file = h5py.File(filename, mode="r")
        self.is_train = is_train

        self.indices = list(i for i, c in enumerate(self.h5_file['input_category'][:]) if c[0].decode("latin-1") in categories)
        self.descriptions = self._build_descriptions()

    def _build_descriptions(self):
        descriptions = self.h5_file["input_description"]
        descriptions = [list(map(lambda k: k.decode("latin-1").replace(".", " <eos>").split(" "), desc)) for desc in descriptions]
        descriptions = [desc[0] for desc in descriptions]

        self.txt_field = Field(tokenize="spacy", tokenizer_language="en", lower=True, include_lengths=True, dtype=torch.float,
                               init_token="<sos>", eos_token="<eos>", unk_token="<unk>", fix_length=32, batch_first=True)
        self.txt_field.build_vocab(descriptions, max_size=10000, min_freq=3, vectors='glove.6B.300d', unk_init=torch.Tensor.normal_)
        self.vocab_size = len(self.txt_field.vocab)

        descriptions, _ = self.txt_field.process(descriptions)
        return descriptions

    def __getitem__(self, index):
        i = self.indices[index]
        img = self.h5_file["input_image"][i, :, :]
        img = ToPILImage()(img)
        rnd_central_eraser = CentralErasing(scale=(0.0625, 0.125), ratio=(0.75, 1.25), value=1)
        h_flip = RandomHorizontalFlip(p=0.5)
        normalizer = Normalize((0.7535, 0.7359, 0.7292), (0.5259, 0.5487, 0.5589))

        desc = self.descriptions[i].float()

        if self.is_train:
            img = h_flip(img)

        img = ToTensor()(img)
        erased, mask, local = rnd_central_eraser(img)

        img = normalizer(img)
        erased = normalizer(erased)

        local = ToTensor()(Resize(size=(32, 32))(ToPILImage()(local)))
        local = normalizer(local)

        return erased, desc, mask, local, img

    def __len__(self):
        return len(self.indices)


class CentralErasing(object):
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=None, inplace=False):
        assert isinstance(value, (numbers.Number, str, tuple, list))
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError("range of scale should be between 0 and 1")

        self.scale = scale
        self.ratio = ratio
        self.value = value
        self.inplace = inplace

    @staticmethod
    def get_params(img, scale, ratio, value=0):
        img_c, img_h, img_w = img.shape
        area = img_h * img_w

        for attempt in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if h < img_h and w < img_w:
                i = random.randint(img_h // 4, 3 * img_h // 4 - h)  # 0, img_h - h
                j = random.randint(img_w // 4, 3 * img_w // 4 - w)  # 0, img_w - w
                if isinstance(value, numbers.Number):
                    v = value
                elif isinstance(value, torch._six.string_classes):
                    v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
                elif isinstance(value, (list, tuple)):
                    v = torch.tensor(value, dtype=torch.float32).view(-1, 1, 1).expand(-1, h, w)
                return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img

    def __call__(self, img):
        x, y, h, w, v = self.get_params(img, scale=self.scale, ratio=self.ratio, value=self.value)
        mask = torch.ones_like(img).float()
        mask[:, x:x+h, y:y+w] = 0.0
        return F.erase(img, x, y, h, w, v, self.inplace), \
            ToTensor()(ToPILImage()(mask)),\
            ToTensor()(F.crop(ToPILImage()(img), x, y, h, w))


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def unnormalize_img(im):
    im = (im + 1.0) / 2.0
    return im


def normalize_img(im):
    im = (im * 2.0) - 1.0
    return im


def normalize_batch(batch, div_factor=1.0):
    """
    Normalize batch
    :param batch: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :param div_factor: normalizing factor before data whitening
    :return: normalized data, tensor with shape
     (batch_size, nbr_channels, height, width)
    """
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, div_factor)

    batch -= Variable(mean)
    batch = torch.div(batch, Variable(std))
    return batch


def unnormalize_batch(batch, div_factor=1.0):
    """
    Unnormalize batch
    :param batch: input tensor with shape
     (batch_size, nbr_channels, height, width)
    :param div_factor: normalizing factor before data whitening
    :return: unnormalized data, tensor with shape
     (batch_size, nbr_channels, height, width)
    """
    # normalize using dataset mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.756
    mean[:, 1, :, :] = 0.736
    mean[:, 2, :, :] = 0.729
    std[:, 0, :, :] = 0.526
    std[:, 1, :, :] = 0.549
    std[:, 2, :, :] = 0.559
    batch = torch.div(batch, div_factor)

    batch *= Variable(std)
    batch = torch.add(batch, Variable(mean))
    return batch


def weights_init(m):
    if type(m) == PartialConv2d:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    BATCH_SIZE = 128
    fg_train = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_train.h5')
    fg_val = HDF5Dataset(filename='./Fashion-Gen/fashiongen_256_256_validation.h5', is_train=False)

    print("Sample size in training: {}".format(len(fg_train)))

    train_loader = data.DataLoader(fg_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = data.DataLoader(fg_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for x_train, x_desc, x_mask, x_local, local_coords, y_train in train_loader:
        batch_samples = x_train.size(0)
        x = x_train.view(batch_samples, x_train.size(1), -1)
        mean += x.mean(2).sum(0)
        std += x.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean)
    print(std)
