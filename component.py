import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np
from sklearn.decomposition import FastICA
import random

import copy
import src.AdaIN_utils as AdaIN_utils
import src.SANet_utils as SANET_utils

def calc_mean_std(features):
    """
    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """
    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--style_dir', type=str, default='datasets/style', help='File path to the style image dataset')
    parser.add_argument('--model_name', type=str, default='AdaIN', help='you can choose one from AdaIN/SANet')
    parser.add_argument('--save_component_dir', type=str, default='component_results', help='Directory to save the component directions.')
    parser.add_argument('--num_components', type=int, default=512, help='Number of independent components')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialization
    if args.model_name == 'AdaIN':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([transforms.ToTensor(), normalize])
        model = AdaIN_utils.AdaIN_Model()
        model.load_state_dict(torch.load("model/AdaIN/model_state.pth", map_location=lambda storage, loc: storage))
        model = model.to(device)

    elif args.model_name == 'SANet':
        vgg, decoder = SANET_utils.SANETmoedl()
        transform = SANET_utils.SANET_Transform(in_planes=512).to(device)
        decoder.eval()
        transform.eval()
        vgg.eval()
        decoder.load_state_dict(torch.load("model/SANet/decoder_iter_500000.pth"))
        transform.load_state_dict(torch.load("model/SANet/transformer_iter_500000.pth"))
        vgg.load_state_dict(torch.load("model/SANet/vgg_normalised.pth"))
        norm = nn.Sequential(*list(vgg.children())[:1]).to(device)
        enc_1 = nn.Sequential(*list(vgg.children())[:4]).to(device)  # input -> relu1_1
        enc_2 = nn.Sequential(*list(vgg.children())[4:11]).to(device)  # relu1_1 -> relu2_1
        enc_3 = nn.Sequential(*list(vgg.children())[11:18]).to(device)  # relu2_1 -> relu3_1
        enc_4 = nn.Sequential(*list(vgg.children())[18:31]).to(device)  # relu3_1 -> relu4_1
        enc_5 = nn.Sequential(*list(vgg.children())[31:44]).to(device)  # relu4_1 -> relu5_1
        style_tf = SANET_utils.SA_test_transform()


    id = os.listdir(args.style_dir)
    with torch.no_grad():
        style_mix = np.ones([1,512])
        for m in id:
            style_info = os.path.join(args.style_dir, str(m))
            if args.model_name == 'AdaIN':
                s = Image.open(style_info)
                s_tensor = trans(s).unsqueeze(0).to(device)
                _, sF = model.getting_code(s_tensor, s_tensor)

            elif args.model_name == 'SANet':
                style = style_tf(Image.open(style_info))
                style = style.unsqueeze(0).to(device)
                Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))  
                sF = copy.deepcopy(Style4_1)

            style_mean,_ = calc_mean_std(sF)
            style_mean = style_mean.view(-1, 512).cpu().detach().numpy()
            style_mix = np.vstack((style_mix, style_mean))

        style_mix = style_mix[1:,:]
        ica = FastICA(n_components=args.num_components)
        S = ica.fit_transform(style_mix.T)
        A = ica.mixing_

        save_boundry_dir = args.save_component_dir

        if not os.path.exists(save_boundry_dir):
            os.makedirs(save_boundry_dir)
        np.savez(os.path.join(save_boundry_dir, args.model_name), S=S.T, A=A)

if __name__ == '__main__':
    main()
