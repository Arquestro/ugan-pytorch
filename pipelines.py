# coding: utf-8

import os
import sys
import torch
import numpy
import math
import matplotlib.pyplot
from enum import Enum, auto
from torch.utils.data import DataLoader
from nets import pix2pix
from torchvision import transforms
from ops.datasets import UganTrainingDataset, UganInferenceDataset, ToTensor
from ops.loss_modules import UganDiscriminatorLoss, UganGeneratorLoss
from stories import Failure, Success, arguments, story
from stories.exceptions import FailureError
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from torchvision.datasets.folder import ImageFolder


class UganTrainPipeline(object):
    """
    All stages of UGAN training, like pre-processing, loop, checkpoint saving are described here
    It is made as a @story with context ctx, workflow inspired by Nikolay Fomynikh
    Nikolay's Medium post on ML pipelines with stories:
    [https://medium.com/@nikolayfn/how-to-make-reusable-machine-learning-pipelines-c77dc8e7e7b9]
    """

    @story
    @arguments(
        "training_dataset_A_path",
        "training_dataset_B_path",
        "epochs_num",
        "batch_size",
        "num_critic",
        "learning_rate",
        "torch_manual_seed",
        "epoch_save_period",
        "save_discriminator_net",
        "model_save_path_prefix",
    )
    def run(I):  # noqa: WPS111, N805, N803
        I.init  # noqa: WPS428
        I.train  # noqa: WPS428

    @story
    @arguments(
        "training_dataset_A_path",
        "training_dataset_B_path",
        "batch_size",
        "learning_rate",
        "torch_manual_seed",
    )
    def init(I):
        I.init_torch_seed  # noqa: WPS428
        I.init_tensorboard_writer  # noqa: WPS428
        I.init_net_modules  # noqa: WPS428
        I.init_loss_modules  # noqa: WPS428
        I.choose_cuda_as_device_if_available  # noqa: WPS428
        I.apply_dataparallel_if_multi_gpu  # noqa: WPS428
        I.move_criterions_to_device  # noqa: WPS428
        I.move_nets_to_device  # noqa: WPS428
        I.init_optimizers  # noqa: WPS428
        I.init_train_dataset  # noqa: WPS428
        I.init_data_loader  # noqa: WPS428

    def init_torch_seed(self, ctx):
        torch.manual_seed(ctx.torch_manual_seed)
        return Success()

    def init_tensorboard_writer(self, ctx):
        tensorboard_writer = SummaryWriter()
        return Success(tensorboard_writer=tensorboard_writer,)

    def init_net_modules(self, ctx):
        generator_net = pix2pix.GeneratorNet()
        discriminator_net = pix2pix.DiscriminatorNet()
        return Success(generator_net=generator_net, discriminator_net=discriminator_net)

    def init_loss_modules(self, ctx):
        criterion_for_generator = UganGeneratorLoss(ctx.tensorboard_writer)
        criterion_for_discriminator = UganDiscriminatorLoss(ctx.tensorboard_writer)
        return Success(
            criterion_for_generator=criterion_for_generator,
            criterion_for_discriminator=criterion_for_discriminator,
        )

    def choose_cuda_as_device_if_available(self, ctx):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_type)
        if device_type == "cpu":
            logger.info("Cuda will not be used for training!")
        else:
            logger.info("{} will be used for training!", device)
        return Success(device=device)

    def apply_dataparallel_if_multi_gpu(self, ctx):
        if torch.cuda.device_count() > 1:
            logger.info("Let's use ", torch.cuda.device_count(), " GPUs!")
            ctx.criterion_for_generator = torch.nn.DataParallel(
                ctx.criterion_for_generator
            )
            ctx.criterion_for_discriminator = torch.nn.DataParallel(
                ctx.criterion_for_discriminator
            )
        return Success()

    def move_criterions_to_device(self, ctx):
        ctx.criterion_for_generator.to(ctx.device)
        ctx.criterion_for_discriminator.to(ctx.device)
        return Success()

    def move_nets_to_device(self, ctx):
        ctx.generator_net.to(ctx.device)
        ctx.discriminator_net.to(ctx.device)
        return Success()

    def init_optimizers(self, ctx):
        generator_net_optimizer = torch.optim.Adam(
            ctx.generator_net.parameters(), lr=ctx.learning_rate
        )
        discriminator_net_optimizer = torch.optim.Adam(
            ctx.discriminator_net.parameters(), lr=ctx.learning_rate
        )
        return Success(
            generator_net_optimizer=generator_net_optimizer,
            discriminator_net_optimizer=discriminator_net_optimizer,
        )

    def init_train_dataset(self, ctx):
        transform_a = transforms.Compose([ToTensor()])
        transform_b = transforms.Compose([ToTensor()])
        training_dataset = UganTrainingDataset(
            ctx.training_dataset_A_path,
            ctx.training_dataset_B_path,
            transform_a=transform_a,
            transform_b=transform_b,
        )
        if len(training_dataset) == 0:
            return Failure(self.Errors.training_dataset_is_empty)
        return Success(training_dataset=training_dataset)

    def init_data_loader(self, ctx):
        data_loader = DataLoader(
            dataset=ctx.training_dataset, batch_size=ctx.batch_size, shuffle=True,
        )
        return Success(data_loader=data_loader)

    def train(self, ctx):
        # Training loop
        logger.info("Start training loop...")
        iteration_index = 0
        for epoch_index in range(ctx.epochs_num):
            for sample_batch_a, sample_batch_b in ctx.data_loader:
                sample_batch_a, sample_batch_b = self.load_batches_to_device(
                    sample_batch_a, sample_batch_b, ctx.device
                )
                generated_images = ctx.generator_net(sample_batch_a)
                for _index in range(ctx.num_critic):
                    discriminator_loss = self.train_discriminator(
                        sample_batch_a,
                        sample_batch_b,
                        generated_images,
                        ctx,
                        iteration_index,
                    )
                generator_loss = self.train_generator(
                    sample_batch_b, generated_images, ctx, iteration_index
                )
                logger.info(
                    "Epoch {}, iteration {}, discriminator loss: {}, generator loss: {}".format(
                        epoch_index + 1,
                        iteration_index + 1,
                        discriminator_loss,
                        generator_loss,
                    )
                )
                iteration_index += 1
            self.checkpoint(ctx, epoch_index)
        return Success()

    def load_batches_to_device(self, sample_batch_a, sample_batch_b, device):
        return sample_batch_a.to(device), sample_batch_b.to(device)

    def train_discriminator(
        self, sample_batch_a, sample_batch_b, generated_images, ctx, iteration_index
    ):
        ctx.discriminator_net_optimizer.zero_grad()
        discriminator_fake = ctx.discriminator_net(generated_images).to(ctx.device)
        discriminator_correct = ctx.discriminator_net(sample_batch_a).to(ctx.device)
        discriminator_loss = ctx.criterion_for_discriminator.forward(
            discriminator_fake,
            discriminator_correct,
            sample_batch_b,
            generated_images,
            ctx.discriminator_net,
            iteration_index,
        )
        discriminator_loss.backward(retain_graph=True)
        ctx.generator_net.zero_grad()
        ctx.discriminator_net_optimizer.step()
        return discriminator_loss.item()

    def train_generator(self, sample_batch_b, generated_images, ctx, iteration_index):
        ctx.generator_net_optimizer.zero_grad()
        discriminator_fake = ctx.discriminator_net(generated_images)
        generator_loss = ctx.criterion_for_generator(
            discriminator_fake, sample_batch_b, generated_images, iteration_index
        )
        generator_loss.backward()
        ctx.discriminator_net.zero_grad()
        ctx.generator_net_optimizer.step()
        return generator_loss.item()

    def checkpoint(self, ctx, epoch_index):
        if (
            epoch_index + 1
        ) % ctx.epoch_save_period == 0 or epoch_index + 1 == ctx.epochs_num:
            # Make sample input for tracing
            sample_size = ctx.training_dataset[0][0].shape
            sample_size = (1, *sample_size)
            sample_input = torch.rand(sample_size).to(ctx.device)
            generator_trace = torch.jit.trace(ctx.generator_net, sample_input)
            generator_trace_filepath = (
                ctx.model_save_path_prefix
                + "_generator_epoch{}.pt".format(epoch_index + 1)
            )
            logger.info("Saving generator trace at {}".format(generator_trace_filepath))
            torch.jit.save(generator_trace, generator_trace_filepath)
            if ctx.save_discriminator_net:
                discriminator_trace_filepath = (
                    ctx.model_save_path_prefix
                    + "_discriminator_epoch{}.pt".format(epoch_index + 1)
                )
                logger.info(
                    "Saving discriminator trace at {}".format(
                        discriminator_trace_filepath
                    )
                )
                discriminator_trace = torch.jit.trace(
                    ctx.discriminator_net, sample_input
                )
                torch.jit.save(discriminator_trace, discriminator_trace_filepath)

    @init.failures
    class Errors(Enum):
        training_dataset_is_empty = auto()


class UganDatasetInferencePipeline(object):
    """
    Simple pipeline for inferencing data from dataset and model trace
    It is made as a @story with context ctx, same as UganTrainPipeline
    """

    @story
    @arguments(
        "target_dataset_path",
        "trace_filepath",
        "output_directory_path",
        "batch_size",
        "torch_manual_seed",
        "use_cuda",
    )
    def run(I):  # noqa: WPS111, N805, N803
        I.init  # noqa: WPS428
        I.inference  # noqa: WPS428

    @story
    @arguments(
        "target_dataset_path",
        "trace_filepath",
        "batch_size",
        "torch_manual_seed",
        "use_cuda",
    )
    def init(I):
        I.init_torch_seed  # noqa: WPS428
        I.init_generator_net_module  # noqa: WPS428
        I.choose_cuda_as_device_if_available  # noqa: WPS428
        I.move_generator_net_to_device  # noqa: WPS428
        I.init_target_dataset  # noqa: WPS428
        I.init_data_loader  # noqa: WPS428

    def init_torch_seed(self, ctx):
        torch.manual_seed(ctx.torch_manual_seed)
        return Success()

    def init_generator_net_module(self, ctx):
        generator = torch.jit.load(ctx.trace_filepath)
        return Success(generator=generator)

    def choose_cuda_as_device_if_available(self, ctx):
        device_type = "cuda" if torch.cuda.is_available() and ctx.use_cuda else "cpu"
        device = torch.device(device_type)
        if device_type == "cpu":
            logger.info("Cuda will not be used for inference!")
        else:
            logger.info("{} will be used for inference!", device)
        return Success(device=device)

    def move_generator_net_to_device(self, ctx):
        ctx.generator.to(ctx.device)
        return Success()

    def init_target_dataset(self, ctx):
        target_dataset = UganInferenceDataset(
            ctx.target_dataset_path, transform=transforms.Compose([ToTensor()])
        )
        if len(target_dataset) == 0:
            return Failure(self.Errors.target_dataset_is_empty)
        return Success(target_dataset=target_dataset)

    def init_data_loader(self, ctx):
        data_loader = DataLoader(
            dataset=ctx.target_dataset, batch_size=ctx.batch_size, shuffle=False,
        )
        return Success(data_loader=data_loader)

    def inference(self, ctx):
        num_of_batches = math.ceil(len(ctx.target_dataset) / ctx.batch_size)
        with torch.no_grad():
            for index, data_batch in enumerate(ctx.data_loader):
                logger.info(
                    "Inferencing batch {} out of {} batches({:.2f}%)".format(
                        index + 1, num_of_batches, 100 * (index + 1) / num_of_batches
                    )
                )
                filenames = [os.path.basename(filepath) for filepath in data_batch[0]]
                image_batch = data_batch[1].to(ctx.device)
                output_batch_tensor = ctx.generator.forward(image_batch)
                logger.info(output_batch_tensor)
                output_images = self.save_batch_to_folder(
                    output_batch_tensor, filenames, ctx.output_directory_path
                )
        return Success()

    def save_batch_to_folder(self, images_tensor, filenames, output_directory_path):
        for index, image_name in enumerate(filenames):
            save_path = output_directory_path + image_name
            image = numpy.asarray(
                images_tensor[index].cpu().detach().permute(1, 2, 0).numpy().clip(0, 1)
            )
            matplotlib.pyplot.imsave(save_path, image)
        return Success()

    @init.failures
    class Errors(Enum):
        target_dataset_is_empty = auto()
