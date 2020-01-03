import torch
from torch import nn
import pytorch_msssim


class PixelLoss(nn.Module):
    def __init__(self):
        super(PixelLoss, self).__init__()
        self.loss_fn = nn.SmoothL1Loss()

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


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        var_w = torch.sum(torch.pow(x[:, :, :, :-1] - x[:, :, :, 1:], 2))
        var_h = torch.sum(torch.pow(x[:, :, :-1, :] - x[:, :, 1:, :], 2))
        return var_w + var_h


class AdverserialLoss(nn.Module):
    def __init__(self):
        super(AdverserialLoss, self).__init__()
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        return self.loss_fn(x, torch.ones_like(x))


class CustomInpaintingLoss(nn.Module):
    def __init__(self):
        super(CustomInpaintingLoss, self).__init__()
        self.content_loss = PixelLoss()  # (x, out) := (the original image, inpainting)
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
        self.pixel_loss = PixelLoss()
        self.content_loss = nn.MSELoss()
        self.style_loss = nn.MSELoss()

    def forward(self, x, out, features_x, features_out):
        p_loss = self.pixel_loss(x, out.detach())
        c_loss = self.content_loss(features_x.relu2_2.detach(), features_out.relu2_2.detach())
        s_loss = 0.0
        for f_x, f_out in zip(features_x, features_out):
            G_f_x = self._gram_matrix(f_x.detach())
            G_f_out = self._gram_matrix(f_out.detach())
            s_loss += self.style_loss(G_f_x, G_f_out)
        return 1.0 * p_loss + 3.0 * c_loss + 20.0 * s_loss, p_loss, c_loss, s_loss

    @staticmethod
    def _gram_matrix(mat):
        b, ch, h, w = mat.size()
        m = mat.view(b, ch, w * h)
        m_transposed = m.transpose(1, 2)
        G = m.bmm(m_transposed) / (h * w * ch)
        return G


class RefineLoss(nn.Module):
    def __init__(self):
        super(RefineLoss, self).__init__()
        self.pixel_loss = PixelLoss()
        self.content_loss = nn.MSELoss()
        self.style_loss = nn.MSELoss()
        self.global_loss = nn.BCELoss()
        self.local_loss = nn.BCELoss()
        self.tv_loss = TVLoss()

    def forward(self, x, out, d_x, d_out, features_x, features_out):
        p_loss = self.pixel_loss(x, out.detach())
        c_loss = self.content_loss(features_x.relu2_2.detach(), features_out.relu2_2.detach())
        s_loss = 0.0
        for f_x, f_out in zip(features_x, features_out):
            G_f_x = self._gram_matrix(f_x.detach())
            G_f_out = self._gram_matrix(f_out.detach())
            s_loss += self.style_loss(G_f_x, G_f_out)
        g_loss = self.global_loss(d_x, d_out.detach())
        l_loss = self.local_loss(d_x, d_out.detach())
        t_loss = self.tv_loss(x, out.detach())
        return 1.0 * p_loss + 3.0 * c_loss + 25.0 * s_loss + 0.25 * g_loss + 0.75 * l_loss + 0.1 * t_loss, \
            p_loss, c_loss, s_loss, g_loss, l_loss, t_loss

    @staticmethod
    def _gram_matrix(mat):
        b, ch, h, w = mat.size()
        m = mat.view(b, ch, w * h)
        m_transposed = m.transpose(1, 2)
        G = m.bmm(m_transposed) / (h * w * ch)
        return G
