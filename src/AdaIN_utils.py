import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import torch

def denorm(tensor, device):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1).to(device)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1).to(device)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res


def AdaIN_calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = AdaIN_calc_mean_std(content_features)
    style_mean, style_std = AdaIN_calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features


class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.slice1 = vgg[: 2]   # relu1-1
        self.slice2 = vgg[2: 7]  # relu2-1
        self.slice3 = vgg[7: 12]  # relu3-1
        self.slice4 = vgg[12: 21]  # relu4-1
        for p in self.parameters():
            p.requires_grad = False


    # relu1-1, relu2-1, relu3-1, relu4-1这些层的结果的损失和作为风格损失
    def forward(self, images, output_last_feature=True):
        h1 = self.slice1(images)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        if output_last_feature:
            return h4
        else:
            return h1, h2, h3, h4


class RC(nn.Module):
    """A wrapper of ReflectionPad2d and Conv2d"""
    def __init__(self, in_channels, out_channels, kernel_size=3, pad_size=1, activated=True):
        super().__init__()
        self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.activated = activated

    def forward(self, x):
        h = self.pad(x)
        h = self.conv(h)
        if self.activated:
            return F.relu(h)
        else:
            return h


class AdaIN_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rc1 = RC(512, 256, 3, 1)
        self.rc2 = RC(256, 256, 3, 1)
        self.rc3 = RC(256, 256, 3, 1)
        self.rc4 = RC(256, 256, 3, 1)
        self.rc5 = RC(256, 128, 3, 1)
        self.rc6 = RC(128, 128, 3, 1)
        self.rc7 = RC(128, 64, 3, 1)
        self.rc8 = RC(64, 64, 3, 1)
        self.rc9 = RC(64, 3, 3, 1, False)

    def forward(self, features):
        h = self.rc1(features)
        h = F.interpolate(h, scale_factor=2)  # 上采样
        h = self.rc2(h)
        h = self.rc3(h)
        h = self.rc4(h)
        h = self.rc5(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc6(h)
        h = self.rc7(h)
        h = F.interpolate(h, scale_factor=2)
        h = self.rc8(h)
        h = self.rc9(h)    # 输出图像
        return h


class AdaIN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg_encoder = VGGEncoder()
        self.decoder = AdaIN_Decoder()

    def getting_code(self, content_images, style_images):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)
        style_features = self.vgg_encoder(style_images, output_last_feature=True)
        return content_features, style_features

    def getting_total_code(self, content_features, style_features):
        latent_code = adain(content_features, style_features)
        return latent_code

    def generate(self, latent_code):
        out = self.decoder(latent_code)
        return out

    @staticmethod
    def calc_content_loss(out_features, t):
        return F.mse_loss(out_features, t)

    @staticmethod
    def calc_style_loss(content_middle_features, style_middle_features):
        loss = 0
        for c, s in zip(content_middle_features, style_middle_features):
            c_mean, c_std = AdaIN_calc_mean_std(c)
            s_mean, s_std = AdaIN_calc_mean_std(s)
            loss += F.mse_loss(c_mean, s_mean) + F.mse_loss(c_std, s_std)
        return loss

    def forward(self, content_images, style_images, alpha=1.0, lam=10):
        content_features = self.vgg_encoder(content_images, output_last_feature=True)  # [1, 512, 32, 32]
        style_features = self.vgg_encoder(style_images, output_last_feature=True)   # [1, 512, 32, 32]
        t = adain(content_features, style_features)
        t = alpha * t + (1 - alpha) * content_features     # 风格化的程度
        out = self.decoder(t)
        output_features = self.vgg_encoder(out, output_last_feature=True)
        output_middle_features = self.vgg_encoder(out, output_last_feature=False)
        style_middle_features = self.vgg_encoder(style_images, output_last_feature=False)

        loss_c = self.calc_content_loss(output_features, t)
        loss_s = self.calc_style_loss(output_middle_features, style_middle_features)
        loss = loss_c + lam * loss_s
        return loss
