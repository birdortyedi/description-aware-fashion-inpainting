import torch
from torch import nn
import pytorch_msssim


class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, x, out):
        return self.loss_fn(x, out)


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.loss_fn = nn.L1Loss()

    def forward(self, x, out):
        G_x = self._gram_matrix(x)
        G_out = self._gram_matrix(out).detach()
        return self.loss_fn(G_x, G_out)

    @staticmethod
    def _gram_matrix(mat):
        b, ch, h, w = mat.size()
        m = mat.view(b, ch, w * h)
        m_transposed = m.transpose(1, 2)
        G = m.bmm(m_transposed) / (h * w * ch)
        return G


class AdverserialLoss(nn.Module):
    def __init__(self):
        super(AdverserialLoss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.loss_fn(x, torch.ones_like(x))


class CustomInpaintingLoss(nn.Module):
    def __init__(self):
        super(CustomInpaintingLoss, self).__init__()
        self.content_loss = ContentLoss()  # (x, out) := (the original image, inpainting)
        self.content_weight = 1.0
        self.style_loss = StyleLoss()  # (x, out) := (the original image, inpainting)
        self.style_weight = 250.0
        self.structural_loss = pytorch_msssim.MSSSIM()  # (x, out) := (the original image, inpainting)
        self.structural_weight = 1.0
        self.adversarial_loss = AdverserialLoss()
        # (d_x, d_out) := (discriminator(x), discriminator(out))
        self.adversarial_weight = 0.1

    def forward(self, x, out, d_out):
        con_loss = self.content_loss(x, out.detach())
        sty_loss = self.style_loss(x, out.detach())
        str_loss = 1 - self.structural_loss(x, out.detach())
        adv_loss = self.adversarial_loss(d_out.detach())
        return self.content_weight * con_loss + \
               self.style_weight * sty_loss + \
               self.structural_weight * str_loss + \
               self.adversarial_weight * adv_loss, \
               con_loss, sty_loss, str_loss, adv_loss
