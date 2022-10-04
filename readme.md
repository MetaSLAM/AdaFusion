# AdaFusion: Visual-LiDAR Fusion with Adaptive Weights for Place Recognition
## Description
This is the source code for paper

> H. Lai, P. Yin, S. Scherer. AdaFusion: Visual-LiDAR Fusion with Adaptive Weights for Place Recognition. IEEE Robotics and Automation Letters.

AdaFusion is a multi-modality fusion network that learns the compact feature representation of both images and point clouds and then adjusts their contribution in different environmental situation. The weights serve as dynamic adjustment to the contribution of the two modalities in different environments.

![image](./doc/demo.png)

## Usage
- For training:
```bash
python train.py --work-path experiments/xxxx [--resume]
```

## Citation
If you find our work useful in your research, please consider citing:

    @ARTICLE{lai2022adafusion,
      author={Lai, Haowen and Yin, Peng and Scherer, Sebastian},
      journal={IEEE Robotics and Automation Letters}, 
      title={AdaFusion: Visual-LiDAR Fusion with Adaptive Weights for Place Recognition}, 
      year={2022},
      pages={1-8},
      doi={10.1109/LRA.2022.3210880}}
