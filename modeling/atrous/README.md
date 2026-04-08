<div align="center">
  <img src="assets/readme_images/deeplabv3p_architecture.png" alt="DeepLabV3+ architecture" width="95%" />
</div>

<h1 align="center"> DeepLab (Atrous) </h1> 

> DeepLab V1: [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs — Chen et al., 2015](https://arxiv.org/pdf/1412.7062)

> DeepLab V2: [Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs — Chen et al., 2017](https://arxiv.org/pdf/1606.00915)

> DeepLab V3: [Rethinking Atrous Convolution for Semantic Image Segmentation — Chen et al., 2017](https://arxiv.org/pdf/1706.05587)

> DeepLab V3+: [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation — Chen et al., 2018](https://arxiv.org/pdf/1802.02611)



# DeepLab-V3+ Architecture

<img src="assets/readme_images/Architecture-of-DeepLabV3-with-backbone-network.png" alt="DeepLabV3+ architecture" width="95%" />


# Each Paper Key Contribution

1. **DeepLabV1:**
    The paper introduced atrous( dilated) convolution which let the kernel see wider view of the picture

2. **DeepLabV2:**
    - The paper introduced **Atrous Spatial Pyramid Pooling (ASPP)**: multiscale atrous (different rates at the same layer). Capturing different scales helps network converge earlier and capture high level features which is normally seen in deeper layers
    - The network has the ability to see high level features without any loss in features as happens in deeper layers

3. **DeepLabV3:**
    Simplified the architecture by removing CRFs and focusing on ASPP with batch normalization.

4. **DeepLabV3+:**
    The paper moved everything introduced before to the encoder-decoder style with skip connections
