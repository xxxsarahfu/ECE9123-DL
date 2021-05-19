# Generate Handwritten Digits by GANs

##### Final project for NYU ECEGY-9123 Deep Learning, May 2021 <br>
---

This project utilizes two types of GAN-based models: vanilla GAN and DCGAN, to generate handwritten digits based on MNIST dataset. Analyis about parameters settings and a comparison of different version of GANs are proposed in the report.

### Execution
1. Running on Google Colab, click at `vanillaGAN+DCGAN.ipynb` > open in colab.
2. Running python code: download the file `vanillaGAN+DCGAN.py` 
```
python vanillaGAN+DCGAN.py
```
Please note that you can change the GAN type by comment/uncomment `gan_type = 'vGAN'` or `gan_type = 'DCGAN` in PARAMETERS SETTINGS section.

### Demo
#### Vanilla GAN
![gan](https://user-images.githubusercontent.com/26239373/118747329-d1525000-b827-11eb-88e0-cf389f758a7e.gif)

#### DCGAN
![gan](https://user-images.githubusercontent.com/26239373/118747298-c39cca80-b827-11eb-8e00-bc9a0cd95228.gif)

Please look at the report for more details and analysis.

### References
* Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative Adversarial Networks. 2014.
* Alec Radford, Luke Metz, and Soumith Chintala. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks. 2016.
* Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. 2015.
* Y. LECUN. The mnist database of handwritten digits. http://yann.lecun.com/exdb/mnist/.
