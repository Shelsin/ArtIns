# ArtIns - Artistic Style Discovery with Independent Components

![image](./fig.png)
**Figure:** *Diverse restylized artworks from different backbones including AdaIN, Linear, SANet and MST. In the first two rows, the first column is the source of the content image with the style image and the second column is the original artistic output, the other columns are the output images with artistic styles discovered by our algorithm. In the last row, given a natural scene, our method yields the other paintings.*


> **Artistic Style Discovery with Independent Components** <br>
> Xin Xie, Yi Li, Huaibo Huang, Haiyan Fu, Wanwan Wang, Yanqing Guo <br>
> *Computer Vision and Pattern Recognition (CVPR), 2022 (**Poster**)*
### [Paper](https://github.com/Shelsin/ArtIns) | [Poster](https://github.com/Shelsin/FIleStore/blob/main/ArtIns_CVPR2022/ArtIns_poster.pdf)

In this repository, we propose an unsupervised approach, termed as **ArtIns**, to discover diverse styles from the latent space consisting of diverse style features. Specifically, we rethink the sense of the style features and find that the latent style representations may be composed of multiple independent style components. These style components can be captured from the latent style space by mathematical operations. Finally, new styles are synthesized by linearly combining style ingredients with different coefficients.

![image](./fig2.png)
**Figure:** *Some components can be given explicit property definitions, such as exposure, brightness, definition, contrast, saturation, color temperature, etc.*

## Artistic Ingredients Separation
Collecting different style features to build the mixed matrix, which is divided into multiple independent artistic components by FastICA algorithm like the cocktail party problem.
```bash
python DirectionFinding.py ${MODEL_NAME} 
```

## Artwork adjustment
Different components control different style effect, and artwork can be adjusted by changing the style code according to the style components.
```bash
python main.py ${MODEL_NAME} 
```


## BibTeX
If you find our work useful in your research, please cite our paper using the following BibTeX entry ~ 

```bibtex
@inproceedings{shelsin2021artins,
  title     = {Artistic Style Discovery with Independent Components},
  author    = {Xin Xie and Yi Li and Huaibo Huang and Haiyan Fu and Wanwan Wang and Yanqing Guo},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
