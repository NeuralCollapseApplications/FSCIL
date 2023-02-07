# Neural Collapse Inspired Feature-Classifier Alignment for Few-Shot Class-Incremental Learning

Authors: [Yibo Yang](https://iboing.github.io/), [Haobo Yuan](https://yuanhaobo.me/), [Xiangtai Li](https://lxtgh.github.io/), [Zhouchen Lin](https://zhouchenlin.github.io/), [Philip Torr](https://www.robots.ox.ac.uk/~phst/), [Dacheng Tao](https://www.sydney.edu.au/engineering/about/our-people/academic-staff/dacheng-tao.html)

**Accepted by ICLR 2023 (top25%), Kigali, Rwanda.**



[[PDF]](https://arxiv.org/pdf/2302.03004) [[CODE]](https://github.com/NeuralCollapseApplications/FSCIL)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neural-collapse-inspired-feature-classifier/few-shot-class-incremental-learning-on-mini)](https://paperswithcode.com/sota/few-shot-class-incremental-learning-on-mini?p=neural-collapse-inspired-feature-classifier)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neural-collapse-inspired-feature-classifier/few-shot-class-incremental-learning-on-cifar)](https://paperswithcode.com/sota/few-shot-class-incremental-learning-on-cifar?p=neural-collapse-inspired-feature-classifier)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/neural-collapse-inspired-feature-classifier/few-shot-class-incremental-learning-on-cub)](https://paperswithcode.com/sota/few-shot-class-incremental-learning-on-cub?p=neural-collapse-inspired-feature-classifier)


## Environment

You do not need to install the environment. What you need is to start a docker container. I already put the docker image online.

```commandline
DATALOC={YOUR DATA LOCATION} LOGLOC={YOUR LOG LOCATION} bash tools/docker.sh
```

If you want to build it by yourself (otherwise **ignore** it). Please run:
```commandline
docker build -t harbory/openmmlab:2206 --network=host .
```

## Data Preparation
You do not need to prepare CIFAR datasets since it is managed by torch.

For other datasets, please refer to [hub](https://huggingface.co/datasets/HarborYuan/Few-Shot-Class-Incremental-Learning)([Link](https://huggingface.co/datasets/HarborYuan/Few-Shot-Class-Incremental-Learning/blob/main/fscil.zip)). It is worth noting that the Mini ImageNet dataset is with various versions. Here we follow [CEC](https://drive.google.com/drive/folders/11LxZCQj2FRCs0JTsf_dafvTHqFn2yGSN), which is widely adopted in FSCIL. Please keep in mind that the usage of datasets is governed by their corresponding agreements. Data sharing here is for research purposes only.

Please put the datasets into the {YOUR DATA LOCATION} you provided above.

## Getting Start
Let's go for 🏃‍♀️running code.

[Update🙋‍♀️] We test the training scripts after the release, please refer to logs.
### CIFAR
Run:
```commandline
bash tools/dist_train.sh configs/cifar/resnet12_etf_bs512_200e_cifar.py 8 --work-dir /opt/logger/cifar_etf --seed 0 --deterministic && bash tools/run_fscil.sh configs/cifar/resnet12_etf_bs512_200e_cifar_eval.py /opt/logger/cifar_etf /opt/logger/cifar_etf/best.pth 8 --seed 0 --deterministic
```
| Session  | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| NC-FSCIL | 82.52 | 76.82 | 73.34 | 69.68 | 66.19 | 62.85 | 60.96 | 59.02 | 56.11 |

[[Base Log]](logs/cifar_base.log) [[Incremental Log]](logs/cifar_inc.log)

### Mini Imagenet
Run:
```commandline
bash tools/dist_train.sh configs/mini_imagenet/resnet12_etf_bs512_500e_miniimagenet.py 8 --work-dir /opt/logger/m_imagenet_etf --seed 0 --deterministic && bash tools/run_fscil.sh configs/mini_imagenet/resnet12_etf_bs512_500e_miniimagenet_eval.py /opt/logger/m_imagenet_etf /opt/logger/m_imagenet_etf/best.pth 8 --seed 0 --deterministic
```

| Session  | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| NC-FSCIL | 84.02 | 76.80 | 72.00 | 67.83 | 66.35 | 64.04 | 61.46 | 59.54 | 58.31 |

[[Base Log]](logs/min_base.log) [[Incremental Log]](logs/min_inc.log)

### CUB
Run:
```commandline
bash tools/dist_train.sh configs/cub/resnet18_etf_bs512_80e_cub.py 8 --work-dir /opt/logger/cub_etf --seed 0 --deterministic && bash tools/run_fscil.sh configs/cub/resnet18_etf_bs512_80e_cub_eval.py /opt/logger/cub_etf /opt/logger/cub_etf/best.pth 8 --seed 0 --deterministic
```

| Session  | 0     | 1     | 2     | 3     | 4     | 5     | 6     | 7     | 8     | 9     | 10    |
|----------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| NC-FSCIL | 80.45 | 75.98 | 72.30 | 70.28 | 68.17 | 65.16 | 64.43 | 63.25 | 60.66 | 60.01 | 59.44 |

[[Base Log]](logs/cub_base.log) [[Incremental Log]](logs/cub_inc.log)

## Citation
If you think the code is useful in your research, please consider to refer:
```bibtex
@inproceedings{yang2023neural,
  title = {Neural Collapse Inspired Feature-Classifier Alignment for Few-Shot Class-Incremental Learning},
  author = {Yang, Yibo and Yuan, Haobo and Li, Xiangtai and Lin, Zhouchen and Torr, Philip and Tao, Dacheng},
  booktitle = {ICLR},
  year = {2023},
}
```
