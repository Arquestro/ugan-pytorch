# coding: utf-8

import torch
import torch.utils.data
import os
import numpy
import glob
import imageio


class UganTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path_to_train_a_folder,
        path_to_train_b_folder,
        transform_a=None,
        transform_b=None,
        do_check_filenames_pairwise=True,
    ):
        images_path_train_a_pattern = os.path.join(path_to_train_a_folder, "*.jpg")
        images_path_train_b_pattern = os.path.join(path_to_train_b_folder, "*.jpg")
        self.images_train_a = numpy.asarray(glob.glob(images_path_train_a_pattern))
        self.images_train_a.sort()
        self.images_train_b = numpy.asarray(glob.glob(images_path_train_b_pattern))
        self.images_train_b.sort()
        if do_check_filenames_pairwise:
            self.check_filenames_pairwise()
        self.transform_a = transform_a
        self.transform_b = transform_b

    def __len__(self):
        return len(self.images_train_a)

    def __getitem__(self, idx):
        sample_a = imageio.imread(self.images_train_a[idx])
        sample_b = imageio.imread(self.images_train_b[idx])
        if self.transform_a:
            sample_a = self.transform_a(sample_a)
        if self.transform_b:
            sample_b = self.transform_b(sample_b)
        return sample_a, sample_b

    def check_filenames_pairwise(self):
        for path_a, path_b in zip(self.images_train_a, self.images_train_b):
            if os.path.basename(path_a) != os.path.basename(path_b):
                raise Exception(
                    "UganTrainingDataset: Either {} or {} doesn't have image pair!".format(
                        path_a, path_b
                    )
                )


class UganInferenceDataset(torch.utils.data.Dataset):
    def __init__(
        self, folder_path, transform=None,
    ):
        images_path_pattern = os.path.join(folder_path, "*.jpg")
        self.images = numpy.asarray(glob.glob(images_path_pattern))
        self.images.sort()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_filename = self.images[idx]
        sample = imageio.imread(image_filename)
        if self.transform:
            sample = self.transform(sample)
        return (image_filename, sample)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        # Normalize image from [0, 255] to [0, 1]
        image = 1 / 255.0 * image
        return torch.from_numpy(image).type(dtype=torch.float)
