import torch
from torch import nn


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        var_w = torch.sum(torch.pow(x[:, :, :, :-1] - x[:, :, :, 1:], 2))
        var_h = torch.sum(torch.pow(x[:, :, :-1, :] - x[:, :, 1:, :], 2))
        return var_w + var_h


class CoarseLoss(nn.Module):
    def __init__(self):
        super(CoarseLoss, self).__init__()
        self.pixel_loss = nn.SmoothL1Loss()
        self.style_loss = nn.SmoothL1Loss()

    def forward(self, x, out): # , features_x, features_out):
        p_loss = self.pixel_loss(x, out.detach())
        # c_loss = 0.0
        G_x = self._gram_matrix(x).detach()
        G_out = self._gram_matrix(out).detach()
        s_loss = self.style_loss(G_x, G_out)
        # for f_x, f_out in zip(features_x, features_out):
        # G_f_x = self._gram_matrix(f_x).detach()
        # G_f_out = self._gram_matrix(f_out).detach()
        # s_loss += self.style_loss(G_f_x, G_f_out)
        # c_loss += self.content_loss(f_x.detach(), f_out.detach()) / 255.
        return 20.0 * p_loss + 100.0 * s_loss, p_loss, s_loss

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
        self.pixel_loss = nn.SmoothL1Loss()
        self.style_loss = nn.SmoothL1Loss()
        self.tv_loss = TVLoss()

    def forward(self, x, out): # , features_x, features_out):
        p_loss = self.pixel_loss(x, out.detach())
        # c_loss = 0.0
        G_x = self._gram_matrix(x).detach()
        G_out = self._gram_matrix(out).detach()
        s_loss = self.style_loss(G_x, G_out)
        # for f_x, f_out in zip(features_x, features_out):
        # G_f_x = self._gram_matrix(f_x).detach()
        # G_f_out = self._gram_matrix(f_out).detach()
        # s_loss += self.style_loss(G_f_x, G_f_out)
        # c_loss += self.content_loss(f_x.detach(), f_out.detach()) / 255.
        t_loss = self.tv_loss(out.detach())
        return 20.0 * p_loss + 100.0 * s_loss + 0.0000002 * t_loss, \
        p_loss, s_loss, t_loss

    @staticmethod
    def _gram_matrix(mat):
        b, ch, h, w = mat.size()
        m = mat.view(b, ch, w * h)
        m_transposed = m.transpose(1, 2)
        G = m.bmm(m_transposed) / (h * w * ch)
        return G
