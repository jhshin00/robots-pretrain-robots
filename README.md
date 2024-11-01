# [🧸 Robots Pre-Train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets](https://robots-pretrain-robots.github.io/)

<a href="https://robots-pretrain-robots.github.io/"><strong>Project Page</strong></a>
  |
  <a href="https://arxiv.org/abs/2410.22325"><strong>arXiv</strong></a>
  |
  <a href="https://x.com/LuccaChiang/status/1851651164187635732"><strong>Twitter</strong></a> 
  | <a href="https://huggingface.co/GqJiang/robots-pretrain-robots"><strong>Dataset & Model</strong></a>

  <a href="https://luccachiang.github.io/">Guangqi Jiang*</a>, 
  <a href="https://guangnianyuji.github.io/">Yifei Sun*</a>, 
  <a href="https://taohuang13.github.io/">Tao Huang*</a>, 
  <a href="https://github.com/xierhill">Huanyu Li</a>, 
  <a href="https://cheryyunl.github.io/">Yongyuan Liang</a>, 
  <a href="http://hxu.rocks/">Huazhe Xu</a>


**In submission, 2024**

<div align="center">
  <img src="resources/overview.png" alt="mcr" width="100%">
</div>

# 🗞️ News

- **2024-10-31** Release code!
- **2024-10-29** Release our paper on ArXiv.


# 🛠️ Installation

Clone this repository and create a conda environment:

    git clone https://github.com/luccachiang/robots-pretrain-robots.git
    cd robots-pretrain-robots
    conda remove -n mcr --all
    conda env create -f mcr/mcr.yaml
    conda activate mcr


Install MCR:

    pip install -e .

# 📚 Data and checkpoints
Our processed DROID subset and pre-trained model checkpoints are availble on our [Huggingface repository](https://huggingface.co/GqJiang/robots-pretrain-robots).

# 💻 Usage
You can use this codebase for the following purposes:

## 1. Use our released pre-trained checkpoints.

    # first, download our model checkpoint from Huggingface
    # then get a torchvision.models.resnet50
    import mcr
    encoder = mcr.load_model(ckpt_path=<path_to_downloaded_ckpt>)

    # please see more details in utils/example.py

## 2. Train MCR from scratch.

    # first, download our pre-trained dataset from Huggingface TODO
    # then run
    cd mcr
    bash train_mcr.sh
    # you can get a full list of parameter helps in train_mcr.sh

## 3. Train MCR with custom dataset.
TODO
We also provide a guidance on how to train MCR on your own dataset. Basically, you need to modify code in xxx, xxx, and xxx.


# 🧭 Code Navigation
    todo

# 🏷️ Licence
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

# ✉️ Acknowledgement & Contact
Our codebase is built upon [R3M](https://github.com/facebookresearch/r3m.git). We thank all these authors for their nicely open sourced code and their great contributions to the community.

Please contact [Guangqi Jiang](https://luccachiang.github.io/) if you are interested in this project. Also feel free to open an issue or raise a pull request :)


# 📝 BibTeX

We will be glad if you find this work helpful. Please consider citing:
```
@article{jiang2024robots,
        title={Robots Pre-Train Robots: Manipulation-Centric Robotic Representation from Large-Scale Robot Datasets},
        author={Jiang, Guangqi and Sun, Yifei and Huang, Tao and Li, Huanyu and Liang, Yongyuan and Xu, Huazhe},
        journal={arXiv preprint arXiv:2410.22325},
        year={2024}
        }
```