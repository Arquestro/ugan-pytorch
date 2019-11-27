# coding: utf-8

import torch


class UganGeneratorLoss(torch.nn.Module):
    def __init__(
        self,
        tensorboard_writer=None,
        lambda_l1=10.0,
        lambda_l2=0.0,
        lambda_igdl=10.0,
        igdl_p=1.0,
    ):
        super(UganGeneratorLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.lambda_igdl = lambda_igdl
        self.igdl_p = igdl_p
        self.tensorboard_writer = tensorboard_writer

    def forward(
        self, discriminator_fake, correct_images, generated_images, iteration_index
    ):
        loss = self.base_loss(discriminator_fake)
        self.tensorboard_writer.add_scalar(
            "Generator/BaseLoss/Train", loss, iteration_index
        )
        if self.lambda_l1 > 0.0:
            l1_loss_weighted = self.lambda_l1 * self.l1_loss(
                correct_images, generated_images
            )
            loss += l1_loss_weighted
            self.tensorboard_writer.add_scalar(
                "Generator/L1Loss/Train", l1_loss_weighted, iteration_index
            )
        if self.lambda_l2 > 0.0:
            l2_loss_weighted = self.lambda_l2 * self.l2_loss(
                correct_images, generated_images
            )
            loss += l2_loss_weighted
            self.tensorboard_writer.add_scalar(
                "Generator/L2Loss/Train", l2_loss_weighted, iteration_index
            )
        if self.lambda_igdl > 0.0:
            igdl_weighted = self.lambda_igdl * self.igdl_loss(
                correct_images, generated_images
            )
            loss += igdl_weighted
            self.tensorboard_writer.add_scalar(
                "Generator/ImageGradientDifferenceLoss/Train",
                igdl_weighted,
                iteration_index,
            )
        self.tensorboard_writer.add_scalar(
            "Generator/Loss/Train", loss, iteration_index
        )
        return loss

    def base_loss(self, discriminator_fake):
        loss = -torch.mean(discriminator_fake)
        return loss

    def l1_loss(self, correct_images, generated_images):
        torch_l1_dist = torch.nn.PairwiseDistance(p=1)
        loss = torch.mean(torch_l1_dist(correct_images, generated_images))
        return loss

    def l2_loss(self, correct_images, generated_images):
        torch_l2_dist = torch.nn.PairwiseDistance(p=2)
        loss = torch.mean(torch_l2_dist(correct_images, generated_images))
        return loss

    def igdl_loss(self, correct_images, generated_images):
        correct_images_gradient_x = self.calculate_x_gradient(correct_images)
        generated_images_gradient_x = self.calculate_x_gradient(generated_images)
        correct_images_gradient_y = self.calculate_y_gradient(correct_images)
        generated_images_gradient_y = self.calculate_y_gradient(generated_images)
        pairwise_p_distance = torch.nn.PairwiseDistance(p=self.igdl_p)
        distances_x_gradient = pairwise_p_distance(
            correct_images_gradient_x, generated_images_gradient_x
        )
        distances_y_gradient = pairwise_p_distance(
            correct_images_gradient_y, generated_images_gradient_y
        )
        loss_x_gradient = torch.mean(distances_x_gradient)
        loss_y_gradient = torch.mean(distances_y_gradient)
        loss = 0.5 * (loss_x_gradient + loss_y_gradient)
        return loss

    def calculate_x_gradient(self, images):
        x_gradient_filter = torch.Tensor(
            [
                [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
                [[0, 0, 0], [-1, 0, 1], [0, 0, 0]],
            ]
        ).cuda()
        x_gradient_filter = x_gradient_filter.view(3, 1, 3, 3)
        result = torch.functional.F.conv2d(
            images, x_gradient_filter, groups=3, padding=(1, 1)
        )
        return result

    def calculate_y_gradient(self, images):
        y_gradient_filter = torch.Tensor(
            [
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
                [[0, 1, 0], [0, 0, 0], [0, -1, 0]],
            ]
        ).cuda()
        y_gradient_filter = y_gradient_filter.view(3, 1, 3, 3)
        result = torch.functional.F.conv2d(
            images, y_gradient_filter, groups=3, padding=(1, 1)
        )
        return result


class UganDiscriminatorLoss(torch.nn.Module):
    def __init__(self, tensorboard_writer=None, lambda_gp=10.0):
        super(UganDiscriminatorLoss, self).__init__()
        self.lambda_gp = lambda_gp
        self.tensorboard_writer = tensorboard_writer

    def forward(
        self,
        discriminator_fake,
        discriminator_correct,
        correct_images,
        generated_images,
        discriminator_net,
        iteration_index,
    ):
        loss = self.wgan_loss(discriminator_fake, discriminator_correct)
        self.tensorboard_writer.add_scalar(
            "Discriminator/WGANLoss/Train", loss, iteration_index
        )
        gp_weighted = self.lambda_gp * self.gradient_penalty(
            correct_images, generated_images, discriminator_net
        )
        loss += gp_weighted
        self.tensorboard_writer.add_scalar(
            "Discriminator/GradientPenalty/Train", gp_weighted, iteration_index
        )
        self.tensorboard_writer.add_scalar(
            "Discriminator/Loss/Train", loss, iteration_index
        )
        return loss

    def wgan_loss(self, discriminator_fake, discriminator_correct):
        loss = torch.mean(discriminator_fake) - torch.mean(discriminator_correct)
        return loss

    def gradient_penalty(
        self, correct_images, generated_images, discriminator_net, norm_epsilon=1e-12
    ):
        batch_size = correct_images.shape[0]
        epsilon = torch.rand(batch_size, 1, 1, 1).cuda()
        x_interpolated = correct_images * epsilon + (1 - epsilon) * generated_images
        interpolated_labels = discriminator_net.forward(x_interpolated).cuda()
        # Following solution source is taken from https://github.com/arturml/pytorch-wgan-gp/blob/master/wgangp.py
        grad_outputs = torch.ones(interpolated_labels.size()).cuda()
        gradients = torch.autograd.grad(
            outputs=interpolated_labels,
            inputs=x_interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + norm_epsilon)
        return torch.mean(((gradients_norm - 1) ** 2))
