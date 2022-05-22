from utils import HtmlPageVisualizer
import numpy as np
import argparse
import os
from utils import postprocess
from PIL import Image
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content_dir', type=str, default='datasets/content', help='File path to the content image')
    parser.add_argument('--content_name', type=str, default='17', help='the name of content image')
    parser.add_argument('--style_dir', type=str, default='datasets/style', help='File path to the style image')
    parser.add_argument('--style_name', type=str, default='17', help='the name of style image')
    parser.add_argument('--boundry', type=str, default='boundries/style_boundry')
    parser.add_argument('--boundry_name', type=str, default='1')
    parser.add_argument('--model_name', type=str, default='SANET',
                        help='you can choose one from SANET/AdaIN/Linear/Avatar')
    parser.add_argument('--ext', default='.jpg', help='The extension name of the image')

    parser.add_argument('--save_dir', type=str, default='output_results',
                        help='Directory to save the visualization pages.')
    parser.add_argument('--html_dir', type=str, default='html_results',
                        help='Directory to save the visualization pages.')
    parser.add_argument('--num_semantics', type=int, default=300, help='Number of semantic boundaries')

    # HTML visualize

    return parser.parse_args()

def html_transform(path):
    transformer = []
    transformer.append(transforms.ToTensor())
    ht = transforms.Compose(transformer)
    return ht(Image.open(path).convert("RGB")).unsqueeze(0)

def main():
    args = parse_args()
    num_sem = args.num_semantics
    step = 41
    viz_size = 256
    distances = np.linspace(-100, 100, step)  # change effect

    vizer = HtmlPageVisualizer(num_rows=num_sem, num_cols=step + 1, viz_size=viz_size)
    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer.set_headers(headers)

    for sem_id in range(num_sem):
        vizer.set_cell(sem_id, 0, text=f'Semantic {sem_id+1:03d}', highlight=True)

    save_path = os.path.join(args.save_dir,
                                            f'{args.model_name}_Sem{num_sem}_{args.content_dir.split("/")[-1]}{args.content_name}_To_{args.style_dir.split("/")[-1]}{args.style_name}_with_boundry{args.boundry_name}')


    for j in range(num_sem):
        name1 = "semantic"+str(j+1)
        if not args.model_name == 'Avatar':
            path2 = os.path.join(save_path,name1)
        else:
            path2 = os.path.join(save_path,f'ChangeLayer{args.layer}',name1)
        for m in range(step):
            name2 = str(m+1)+"_distance_"+str(distances[m])+ args.ext
            path3 = os.path.join(path2, name2)
            content = html_transform(path3)
            image = postprocess(content)[0]
            vizer.set_cell(j, m+1, image=image)


    prefix = (f'{args.model_name}_Sem{num_sem}_{args.content_dir.split("/")[-1]}{args.content_name}_To_{args.style_dir.split("/")[-1]}{args.style_name}_with_boundry{args.boundry_name}')

    if not os.path.exists(args.html_dir):
        os.makedirs(args.html_dir)
    vizer.save(os.path.join(args.html_dir, f'{prefix}.html'))

if __name__ == '__main__':
    main()