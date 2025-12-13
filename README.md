## Implemention of HSFM: Hybrid Spatial Frequency Mamba for Efficient Image Restoration

> **Abstract:**  Natural images often suffer from various degradations due to adverse weather conditions and imaging environments. Consequently, image restoration has developed as a crucial solution and continues to attract attention. Although recent transformer-based architectures have demonstrated remarkable success in image restoration, their substantial computational demands create considerable obstacles for real-world deployment. Vision Mamba demonstrates exceptional long-range dependency modeling capabilities with linear complexity, showcasing its potential to replace Transformers. However, single spatial domain information from Mamba tends to be usually insufficient, constraining the model's ability to handle complex real-world degradation scenarios. Furthermore, current methods commonly use MLP layers or incorporate convolution operations for feature propagation in the feed forward network, underutilizing the captured fine-grained domain characteristics. Inspired by the above insights, we propose an efficient network for image restoration, namely Hybrid Spatial Frequency Mamba Network (HSFM). Specifically, our network features two key designs: 1) We propose a Spatial Fourier Mixer (SFM) that integrates spatial features from a selective multi-scale interaction branch and frequency features derived through an adaptive context integration branch; 2) We design a Hybrid Feature Aggregation Module (HFAM) to enhance detail preservation to obtain better image restoration performance. Extensive experiments on three representative image restoration tasks, including image super-resolution, image denoising and image dehazing demonstrate that our network achieves superior performance in restoration quality while maintaining low complexity and memory consumption.


## 📑 Contents

- [Installation](#installation)
- [Training](#training)
- [Testing](#testing)


## <a name="installation"></a> :wrench: Installation

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
