# ArtIns - Artistic Style Discovery with Independent Components

![image](./docs/fig.png)
**Figure 1:** *Diverse restylized artworks from different backbones including AdaIN, Linear, SANet and MST. In the first two rows, the first column is the source of the content image with the style image and the second column is the original artistic output, the other columns are the output images with artistic styles discovered by our algorithm. In the last row, given a natural scene, our method yields the other paintings.*

> **Artistic Style Discovery with Independent Components** <br>
> Xin Xie, Yi Li, Huaibo Huang, Haiyan Fu, Wanwan Wang, Yanqing Guo <br>
> *Computer Vision and Pattern Recognition (CVPR), 2022 (**Poster**)*
### [Paper](https://github.com/Shelsin/ArtIns) | [Demo](https://youtu.be/7AeKzYWRotk) | [Poster](https://doc-0c-6g-apps-viewer.googleusercontent.com/viewer/secure/pdf/pf2oq8shc5hkjs3tma0b7e1s0gq8f581/0o3ehiqegcist1kaql4b787cjtl3e46p/1653220725000/drive/10310165295329163090/ACFrOgAdXn8fFJVNS3v1ZV6RexDDuBp4COfl_YeMPI7ZpKlcaVVsXBRgKAMuvm89Rkz3EEi4nPdhGDHRS9dyYR5nw6aZWUDP8hjxvqSPSaEou4A9YFn1zXLp1a2A3XgKHY7FPwapR0i86FXJPfbr?print=true&nonce=5tl1bfmrajj46&user=10310165295329163090&hash=9jpsln26rvkcege346hcj5m6lc4fasie)

In this repository, we propose an unsupervised approach, termed as **ArtIns**, to discover diverse styles from the latent space consisting of diverse style features. Specifically, we rethink the sense of the style features and find that the latent style representations may be composed of multiple independent style components. These style components can be captured from the latent style space by mathematical operations. Finally, new styles are synthesized by linearly combining style ingredients with different coefficients.

![image](./docs/fig2.png)
**Figure 2:** *Some components can be given explicit property definitions, such as exposure, brightness, definition, contrast, saturation, color temperature, etc.*

## Artistic Ingredients Separation
Collecting different style features to build the mixed matrix, which is divided into multiple independent artistic components by FastICA algorithm like the cocktail party problem. 

![image](./docs/work.png)
**Figure 3:** *The style features are linear sum of style components where the mixed matrix SF can be divided into the mixing matrix A and style components V.*

**NOTE:** The number of the style examples is more than that of the components, which is necessary to ensure that artistic ingredients are independent. The pre-trained models (AdaIN and SANet) can be download [here](https://drive.google.com/drive/folders/1A81l0uQ4xFvfGNtdXFF8jXYCvNzca4uE).

```bash
python component.py --model_name ${MODEL_NAME} --num_components ${NUM_COMPONENTS}
```

## Artwork adjustment
Different components control different style effect, and artwork can be adjusted by changing the style code according to the style components.
```bash
python main.py --model_name ${MODEL_NAME} --num_semantics ${NUM_SEMANTICS}
```


## BibTeX
If you find our work useful in your research, please cite our paper using the following BibTeX entry ~ 

```bibtex
@inproceedings{shelsin2022artins,
  title     = {Artistic Style Discovery with Independent Components},
  author    = {Xin Xie and Yi Li and Huaibo Huang and Haiyan Fu and Wanwan Wang and Yanqing Guo},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2022}
}
```
