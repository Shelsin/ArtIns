import argparse
import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

import copy
from utils import to_tensor
from utils import postprocess
from utils import HtmlPageVisualizer

import src.AdaIN_utils as AdaIN_utils
import src.SANet_utils as SANET_utils


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='datasets/content/1.jpg', help='File path to the content image')
    parser.add_argument('--style_dir', type=str, default='datasets/style/1.jpg', help='File path to the style image')
    parser.add_argument('--model_name', type=str, default='AdaIN', help='you can choose one from AdaIN/SANet')
    parser.add_argument('--ext', default='.jpg', help='The extension name of the image')

    # save images or not
    parser.add_argument('--save_flag', action='store_true', default=True,
                        help='whether or not to save the results as img')
    parser.add_argument('--save_dir', type=str, default='output_results',
                        help='Directory to save the visualization pages.')

    # distance
    parser.add_argument('--start_distance', type=float, default=-100.0,
                        help='Start point for manipulation on each semantic.')
    parser.add_argument('--end_distance', type=float, default=100.0,
                        help='Ending point for manipulation on each semantic. ')
    parser.add_argument('--step', type=int, default=41, help='Manipulation step on each semantic. ')   # 81
    parser.add_argument('--num_semantics', type=int, default=2, help='Number of semantic boundaries')  # 5
    parser.add_argument('--component_dir', type=str, default='component_results', help='Directory to save the component directions.')

    # HTML visualize saving or not
    parser.add_argument('--viz_flag', action='store_true', default=True,
                        help='whether or not show the results  on the HTML page.')
    parser.add_argument('--viz_size', type=int, default=256, help='Size of images to visualize on the HTML page.')
    parser.add_argument('--HTML_results', type=str, default='html_results',
                        help='Directory to save the HTML visualization pages.')

    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # bulid filedir
    if args.save_flag:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

    if args.viz_flag:
        if not os.path.exists(args.HTML_results):
            os.makedirs(args.HTML_results)

    # getting image dir
    content_info = args.content_dir
    style_info = args.style_dir

    # initialization
    if args.model_name == 'AdaIN':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        trans = transforms.Compose([transforms.ToTensor(), normalize])
        model = AdaIN_utils.AdaIN_Model()
        model.load_state_dict(torch.load("model/AdaIN/model_state.pth", map_location=lambda storage, loc: storage))
        model = model.to(device)
        c = Image.open(content_info)
        s = Image.open(style_info)
        c_tensor = trans(c).unsqueeze(0).to(device)
        s_tensor = trans(s).unsqueeze(0).to(device)

    elif args.model_name == 'SANet':
        vgg, decoder = SANET_utils.SANETmoedl()
        transform = SANET_utils.SANET_Transform(in_planes=512).to(device)
        decoder.eval()
        transform.eval()
        vgg.eval()
        decoder.load_state_dict(torch.load("model/SANET/decoder_iter_500000.pth"))
        transform.load_state_dict(torch.load("model/SANET/transformer_iter_500000.pth"))
        vgg.load_state_dict(torch.load("model/SANET/vgg_normalised.pth"))
        enc_1 = nn.Sequential(*list(vgg.children())[:4]).to(device)  # input -> relu1_1
        enc_2 = nn.Sequential(*list(vgg.children())[4:11]).to(device)  # relu1_1 -> relu2_1
        enc_3 = nn.Sequential(*list(vgg.children())[11:18]).to(device)  # relu2_1 -> relu3_1
        enc_4 = nn.Sequential(*list(vgg.children())[18:31]).to(device)  # relu3_1 -> relu4_1
        enc_5 = nn.Sequential(*list(vgg.children())[31:44]).to(device)  # relu4_1 -> relu5_1
        content_tf = SANET_utils.SA_test_transform()
        style_tf = SANET_utils.SA_test_transform()
        content = content_tf(Image.open(content_info))
        style = style_tf(Image.open(style_info))
        style = style.unsqueeze(0).to(device)
        content = content.unsqueeze(0).to(device)
        decoder = decoder.to(device)


    with torch.no_grad():
        if args.model_name == 'AdaIN':
            cF, sF = model.getting_code(c_tensor, s_tensor)

        elif args.model_name == 'SANet':
            Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
            Content5_1 = enc_5(Content4_1)
            Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))  # [1, 512, 64, 64]
            sF = copy.deepcopy(Style4_1)

        distances = np.linspace(args.start_distance, args.end_distance, args.step)  # change effect
        num_sem = args.num_semantics  # num_sem = 1

        if args.viz_flag:
            vizer = HtmlPageVisualizer(num_rows=num_sem,
                                       num_cols=args.step + 1,
                                       viz_size=args.viz_size)
            headers = [''] + [f'Distance {d:.2f}' for d in distances]
            vizer.set_headers(headers)
            for sem_id in range(num_sem):
                vizer.set_cell(sem_id, 0, text=f'Semantic {sem_id+1:03d}', highlight=True)


        component_name = args.model_name + ".npz"
        component_path = os.path.join(args.component_dir,component_name)
        boundaries = torch.tensor(np.load(component_path)["S"]).to(device)


        for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
            boundary = boundaries[sem_id:sem_id + 1]  # 第一层的特征  (1, 512)
            for col_id, d in enumerate(distances, start=1):
                sf_test = copy.deepcopy(sF)
                x = torch.ones(sf_test.shape).to(device)
                for i in range(sf_test.shape[1]):
                    x[:, i, :, :] = x[:, i, :, :] * boundary[0][i]

                sf_test = sf_test + x * d
                # -------------Test------------#
                if args.model_name == 'AdaIN':
                    code = model.getting_total_code(cF, sf_test)
                    content = model.generate(code)
                    content = AdaIN_utils.denorm(content, device)
                elif args.model_name == 'SANet':
                    Style5_1 = enc_5(sf_test)  # [1, 512, 32, 32]
                    code = transform(Content4_1, sf_test, Content5_1, Style5_1)
                    content = decoder(code)

                content.clamp(0, 255)

                if args.save_flag:
                    save_path = os.path.join(args.save_dir, f'{args.model_name}')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_subpath = os.path.join(save_path, "semantic" + str(sem_id + 1))
                    if not os.path.exists(save_subpath):
                        os.makedirs(save_subpath)
                    output_name = '{:s}/{:s}_distance_{:s}{:s}'.format(save_subpath, str(col_id), str(d), args.ext)
                    if args.model_name in ['AdaIN', 'SANet']:
                        content = content.cpu()
                        save_image(content, output_name)
                if args.viz_flag:
                    image = postprocess(content)[0]
                    vizer.set_cell(sem_id, col_id, image=image)

        if args.viz_flag:
            save_path = os.path.join(args.HTML_results, f'{args.model_name}')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            prefix = (f'{args.model_name}')
            vizer.save(os.path.join(save_path, f'{prefix}.html'))

if __name__ == '__main__':
    main()