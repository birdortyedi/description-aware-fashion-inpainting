import torch
from torch import nn
import pytorch_msssim

from utils import normalize_batch


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
        var_w = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]).mean()
        var_h = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]).mean()

        return var_w + var_h


class CoarseLoss(nn.Module):
    def __init__(self):
        super(CoarseLoss, self).__init__()
        self.pixel_loss = nn.L1Loss()
        self.content_loss = nn.L1Loss()
        self.style_loss = nn.L1Loss()
        self.tv_loss = TVLoss()
        self.adversarial_loss = nn.BCELoss()

    def forward(self, x, out, comp, mask, d_out, d_local, vgg, device):
        coarse_vgg_features = vgg(normalize_batch(x))
        coarse_comp_vgg_features = vgg(normalize_batch(comp))
        coarse_output_vgg_features = vgg(normalize_batch(out))

        real_label = torch.ones_like(d_out).to(device)

        x_valid = (1.0 - mask) * x
        out_valid = (1.0 - mask) * out

        x_hole = mask * x
        out_hole = mask * out

        p_loss_valid = self.pixel_loss(out_valid, x_valid.detach()).mean()
        p_loss_hole = self.pixel_loss(out_hole, x_hole.detach()).mean()

        s_loss_out, s_loss_comp = 0., 0.
        c_loss_out, c_loss_comp = 0., 0.
        for i, (f_x, f_comp, f_out) in enumerate(zip(coarse_vgg_features, coarse_comp_vgg_features, coarse_output_vgg_features)):
            G_f_x = self._gram_matrix(f_x)
            G_f_out = self._gram_matrix(f_out)
            G_f_comp = self._gram_matrix(f_comp)
            s_loss_out += self.style_loss(G_f_out, G_f_x).mean()
            s_loss_comp += self.style_loss(G_f_out, G_f_comp).mean()
            if i == 2:
                c_loss_out += self.content_loss(f_out, f_x.detach()).mean()
                c_loss_comp += self.content_loss(f_out, f_comp.detach()).mean()
        s_loss = 0.5 * s_loss_out + 8.0 * s_loss_comp
        c_loss = 0.5 * c_loss_out + 8.0 * c_loss_comp

        tv_loss = self.tv_loss(comp)

        adversarial_loss_global = self.adversarial_loss(d_out, real_label)
        adversarial_loss_local = self.adversarial_loss(d_local, real_label)
        adversarial_loss = 0.5 * adversarial_loss_global + 3.0 * adversarial_loss_local

        return p_loss_valid + 100.0 * p_loss_hole + 0.05 * c_loss + 120.0 * s_loss + 0.001 * tv_loss + 0.1 * adversarial_loss,\
            p_loss_valid, p_loss_hole, c_loss, s_loss, tv_loss, adversarial_loss

    @staticmethod
    def _gram_matrix(mat):
        b, ch, h, w = mat.size()
        m = mat.view(b, ch, w * h)
        m_transposed = m.transpose(1, 2)
        # G = m.bmm(m_transposed) / (h * w * ch)
        i_ = torch.zeros(b, ch, ch).type(m.type())
        G = torch.baddbmm(i_, m, m_transposed, beta=0, alpha=1. / (ch * h * w), out=None)
        return G


class RefineLoss(nn.Module):
    def __init__(self):
        super(RefineLoss, self).__init__()
        self.pixel_loss = nn.SmoothL1Loss()
        self.style_loss = nn.SmoothL1Loss()
        self.tv_loss = TVLoss()

    def forward(self, x, out):  # , features_x, features_out):
        p_loss = self.pixel_loss(x, out.detach())
        # c_loss = 0.0
        G_x = self._gram_matrix(x).detach()
        G_out = self._gram_matrix(out).detach()
        s_loss = self.style_loss(G_x, G_out)
        # for f_x, f_out in zip(features_x, features_out):
        #     G_f_x = self._gram_matrix(f_x).detach()
        #     G_f_out = self._gram_matrix(f_out).detach()
        #     s_loss += self.style_loss(G_f_x, G_f_out)
        #     c_loss += self.content_loss(f_x.detach(), f_out.detach()) / 255.
        t_loss = self.tv_loss(out.detach())
        return 20.0 * p_loss + 100.0 * s_loss + 0.0000002 * t_loss, \
            p_loss, s_loss, t_loss

    @staticmethod
    def _gram_matrix(mat):
        b, ch, h, w = mat.size()
        m = mat.view(b, ch, w * h)
        m_transposed = m.transpose(1, 2)
        # G = m.bmm(m_transposed) / (h * w * ch)
        i_ = torch.zeros(b, ch, ch).type(m.type())
        G = torch.baddbmm(i_, m, m_transposed, beta=0, alpha=1. / (ch * h * w), out=None)
        return G


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.pixel = nn.SmoothL1Loss()
        self.content = nn.L1Loss()
        self.style = nn.L1Loss()
        self.tv = TVLoss()

    def forward(self, x, output, composite, mask, vgg_features_gt, vgg_features_composite, vgg_features_output):
        x_valid = (1.0 - mask) * x
        output_valid = (1.0 - mask) * output
        x_hole = mask * x
        output_hole = mask * output

        pixel_valid_loss = self.pixel(output_valid, x_valid.detach()).mean()
        pixel_hole_loss = self.pixel(output_hole, x_hole.detach()).mean()

        s_loss_output, s_loss_composite = 0.0, 0.0
        c_loss_output, c_loss_composite = 0.0, 0.0
        for i, (f_gt, f_composite, f_output) in enumerate(zip(vgg_features_gt, vgg_features_composite, vgg_features_output)):
            g_f_gt = self._gram_matrix(f_gt)
            g_f_output = self._gram_matrix(f_output)
            g_f_composite = self._gram_matrix(f_composite)
            s_loss_output += self.style(g_f_output, g_f_gt).mean()
            s_loss_composite += self.style(g_f_composite, g_f_gt).mean()
            if i == 2:
                c_loss_output += self.content(f_output, f_gt).mean()
                c_loss_composite += self.content(f_composite, f_gt).mean()
        style_loss = 0.15 * s_loss_output + 0.85 * s_loss_composite
        content_loss = 0.15 * c_loss_output + 0.85 * c_loss_composite

        tv_loss = self.tv(composite)

        return pixel_valid_loss + 6.0 * pixel_hole_loss + 0.5 * content_loss + 120.0 * style_loss + 0.1 * tv_loss, \
            pixel_valid_loss, pixel_hole_loss, content_loss, style_loss, tv_loss

    @staticmethod
    def _gram_matrix(mat):
        b, ch, h, w = mat.size()
        m = mat.view(b, ch, w * h)
        m_transposed = m.transpose(1, 2)
        # G = m.bmm(m_transposed) / (h * w * ch)
        i_ = torch.zeros(b, ch, ch).type(m.type())
        G = torch.baddbmm(i_, m, m_transposed, beta=0, alpha=1. / (ch * h * w), out=None)
        return G
