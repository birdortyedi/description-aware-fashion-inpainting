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
        self.content_weight = 25.0
        self.style_loss = StyleLoss()  # (x, out) := (the original image, inpainting)
        self.style_weight = 100.0
        self.structural_loss = pytorch_msssim.MSSSIM()  # (x, out) := (the original image, inpainting)
        self.structural_weight = 25.0
        self.adversarial_loss = AdverserialLoss()
        # (d_x, d_out) := (discriminator(x), discriminator(out))
        self.adversarial_weight = 1.0

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


class CoarseLoss(nn.Module):
    def __init__(self):
        super(CoarseLoss, self).__init__()
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()

    def forward(self, x, out):
        c_loss = self.content_loss(x, out.detach())
        s_loss = self.style_loss(x, out.detach())
        return 1.0 * c_loss + 5.0 * s_loss, c_loss, s_loss


class RefineLoss(nn.Module):
    def __init__(self):
        super(RefineLoss, self).__init__()
        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.global_loss = nn.BCELoss()
        self.local_loss = nn.BCELoss()

    def forward(self, x, out, d_x, d_out):
        c_loss = self.content_loss(x, out.detach())
        s_loss = self.style_loss(x, out.detach())
        g_loss = self.global_loss(d_x, d_out.detach())
        l_loss = self.local_loss(d_x, d_out.detach())
        return 2.0 * c_loss + 10.0 * s_loss + 0.2 * g_loss + 0.8 * l_loss, c_loss, s_loss, g_loss, l_loss
