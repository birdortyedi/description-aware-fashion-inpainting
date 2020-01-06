import torch
from torchvision.transforms import functional as F
from torchvision.transforms import Normalize, ToTensor, ToPILImage, RandomHorizontalFlip, Resize
from torchtext.data import Field
from torch.utils import data

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

        self.indices = self._filter_by_category()
        self.descriptions = self._build_descriptions()
        self.imgs = self.h5_file['input_image'][self.indices, :, :]

        print("Description len: {}".format(len(self.descriptions)))
        print("Images len: {}".format(len(self.imgs)))

    def _filter_by_category(self):
        indices = []
        for i in range(len(self.h5_file['input_category'])):
            if self.h5_file['input_category'][i][0].decode("latin-1") in categories:
                indices.append(i)
        print("Indices len: {}".format(len(self.indices)))
        return indices

    def _build_descriptions(self):
        descriptions = self.h5_file["input_description"][self.indices]
        descriptions = [list(map(lambda k: k.decode("latin-1").replace(".", " <eos>").split(" "), desc)) for desc in descriptions]
        descriptions = [desc[0] for desc in descriptions]

        self.txt_field = Field(tokenize="spacy", tokenizer_language="en", lower=True, include_lengths=True, dtype=torch.float,
                               init_token="<sos>", eos_token="<eos>", unk_token="<unk>", fix_length=32, batch_first=True)
        self.txt_field.build_vocab(descriptions, max_size=10000, min_freq=3, vectors='glove.6B.300d', unk_init=torch.Tensor.normal_)
        self.vocab_size = len(self.txt_field.vocab)

        descriptions, _ = self.txt_field.process(descriptions)
        return descriptions

    def __getitem__(self, index):
        img = self.imgs[index, :, :]
        img = ToPILImage()(img)
        rnd_central_eraser = CentralErasing(scale=(0.03125, 0.0625), ratio=(0.75, 1.25), value=1)
        h_flip = RandomHorizontalFlip(p=0.5)

        desc = self.descriptions[index].float()

        if self.is_train:
            img = h_flip(img)

        img = ToTensor()(img)
        erased, local, coords = rnd_central_eraser(img)
        local = ToTensor()(Resize(size=(32, 32))(ToPILImage()(local)))

        return erased, desc, local, coords, img

    def __len__(self):
        return len(self.descriptions)


class CentralErasing(object):
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False):
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
                i = random.randint(img_h // 5, 4 * img_h // 5 - h)  # 0, img_h - h
                j = random.randint(img_w // 5, 4 * img_w // 5 - w)  # 0, img_w - w
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
        return F.erase(img, x, y, h, w, v, self.inplace), ToTensor()(F.crop(ToPILImage()(img), x, y, h, w)), torch.Tensor([x, y, h, w])


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


def normalize_batch(batch):
    batch *= 255.0
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std
