import torch
from torch import nn
import pytorch_msssim


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, x, out):
        return self.loss_fn(x, out)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, x, out):
        G_x = self._gram_matrix(x)
        G_out = self._gram_matrix(out).detach()
        return self.loss_fn(G_x, G_out)

    @staticmethod
    def _gram_matrix(mat):
        a, b, c, d = mat.size()
        features = mat.view(a * b, c * d)
        G_mat = torch.mm(features, features.t())
        return G_mat.div(a * b * c * d)


class AdverserialLoss(nn.Module):
    def __init__(self):
        super(AdverserialLoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, x, out):
        real_loss = self.loss_fn(x, torch.ones_like(x))
        fake_loss = self.loss_fn(out, torch.zeros_like(out))
        return 0.5 * (real_loss + fake_loss)


class CustomInpaintingLoss(nn.Module):
    def __init__(self):
        super(CustomInpaintingLoss, self).__init__()
        self.content_loss = ContentLoss()  # (x, out) := (the original image, inpainting)
        self.content_weight = 1.0
        self.style_loss = StyleLoss()  # (x, out) := (the original image, inpainting)
        self.style_weight = 3.0
        self.structural_loss = pytorch_msssim.MSSSIM()  # (x, out) := (the original image, inpainting)
        self.structural_weight = 3.0
        self.adversarial_loss = AdverserialLoss()
        # (d_x, d_out) := (discriminator(x), discriminator(out))
        self.adversarial_weight = 1.0

    def forward(self, x, out, d_x, d_out):
        return self.content_weight * self.content_loss(x, out.detach()) + \
               self.style_weight * self.style_loss(x, out.detach()) + \
               self.structural_weight * self.structural_loss(x, out.detach()) + \
               self.adversarial_weight * self.adversarial_loss(d_x.detach(), d_out.detach())
