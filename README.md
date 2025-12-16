## Implemention of HSFM: Hybrid Spatial Frequency Mamba for Efficient Image Restoration

> **Abstract:**  Natural images often suffer from various degradations due to adverse weather conditions and imaging environments. Consequently, image restoration has developed as a crucial solution and continues to attract attention. Although recent transformer-based architectures have demonstrated remarkable success in image restoration, their substantial computational demands create considerable obstacles for real-world deployment. Vision Mamba demonstrates exceptional long-range dependency modeling capabilities with linear complexity, showcasing its potential to replace Transformers. However, single spatial domain information from Mamba tends to be usually insufficient, constraining the model's ability to handle complex real-world degradation scenarios. Furthermore, current methods commonly use MLP layers or incorporate convolution operations for feature propagation in the feed forward network, underutilizing the captured fine-grained domain characteristics. Inspired by the above insights, we propose an efficient network for image restoration, namely Hybrid Spatial Frequency Mamba Network (HSFM). Specifically, our network features two key designs: 1) We propose a Spatial Fourier Mixer (SFM) that integrates spatial features from a selective multi-scale interaction branch and frequency features derived through an adaptive context integration branch; 2) We design a Hybrid Feature Aggregation Module (HFAM) to enhance detail preservation to obtain better image restoration performance. Extensive experiments on three representative image restoration tasks, including image super-resolution, image denoising and image dehazing demonstrate that our network achieves superior performance in restoration quality while maintaining low complexity and memory consumption.
<p align="center">
    <img src="assets/pipeline.png" style="border-radius: 15px">
</p>


## Datasets

The datasets used in our training and testing are orgnized as follows:

| Task                           | Training Set                                                                                                                                                                                                                                                                                                                                                                                                                                    |                                                                                 Testing Set                                                                                  |
| :----------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------: 
| Image Super-Resolution         | Lightweight SR:[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)<br /> Classic SR:[DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/)+  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) [complete DF2K [download](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link)]                                                                                                                                           |                 Set5 + Set14 + BSD100 + Urban100 + Manga109 [[download](https://drive.google.com/file/d/1n-7pmwjP0isZBK7w3tx2y8CTastlABx1/view?usp=sharing)]                 |
| Gaussian Color Image Denoising | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) + [BSD400](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz) + [WED](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar) <br />[[download](https://drive.google.com/file/d/1jPgG_URDQZ4kyXaMMXJ8AZ8jEErCdKuM/view?usp=share_link)] |                   CBSD68 + Kodak24 + McMaster + Urban100  [[download](https://drive.google.com/file/d/1baLpOjNlTCNbREUDAZf9Lso6YCeUOQER/view?usp=sharing)]                   |  
| Real Image Denoising           | [SIDD](https://www.eecs.yorku.ca/~kamel/sidd/) [SIDD [download](https://drive.google.com/drive/folders/1L_8ig1P71ikzf8PHGs60V6dZ2xoCixaC?usp=share_link)]                                                                                                                                                                                                                                                                              |                                SIDD + DND [[download](https://drive.google.com/file/d/1Vuu0uhm_-PAG-5UPI0bPIaEjSfrSvsTO/view?usp=share_link)]                                | 
| Image Dehazing                 | Indoor & Outdoor:RESIDE [[download](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=0)] <br /> Mix: RESIDE-6K [[download](https://drive.google.com/drive/folders/1oaQSpdYHxEv-nMOB7yCLKfw2NDCJVtrx)]                                                                                                                                      |                                   SOTS / RESIDE-Mix [[download](https://drive.google.com/drive/folders/1oaQSpdYHxEv-nMOB7yCLKfw2NDCJVtrx)]                                   | 

## Installation

This codebase was tested with the following environment configurations. It may work with other versions.

- Ubuntu 22.04.4
- CUDA 12.4
- Python 3.12
- PyTorch 2.5.1 + cu124

### Previous installation
To use the selective scan with efficient hard-ware design, the `mamba_ssm` library is needed to install with the folllowing command.
```
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1
```

One can also create a new anaconda environment, and then install necessary python libraries with this [requirement.txt] and the following command: 
```
conda install --yes --file requirements.txt
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [MambaIR](https://github.com/csguoh/MambaIR). Thanks for their awesome work.



