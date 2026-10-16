# MatchFormer

### MatchFormer: Interleaving Attention in Transformers for Feature Matching

Qing Wang∗, [Jiaming Zhang](https://jamycheung.github.io/)∗, [Kailun Yang](https://yangkailun.com/)†, Kunyu Peng, [Rainer Stiefelhagen](https://cvhci.anthropomatik.kit.edu/people_596.php)

∗ denotes equal contribution and † denotes corresponding author

### News
- [09/2022] **MatchFormer** [[**PDF**](https://arxiv.org/pdf/2203.09645.pdf)] is accepted to **ACCV2022**.

![matchformer](matchformer.png)

### Introduction

In this work, we propose a novel hierarchical extract-and-match transformer, termed as **MatchFormer**. Inside each stage of the hierarchical encoder, we interleave self-attention for feature extraction and cross-attention for feature matching, enabling a human-intuitive **extract-and-match** scheme. 

More detailed can be found in our [arxiv](https://arxiv.org/pdf/2203.09645.pdf) paper.

### Installation

 The requirements are listed in the `requirement.txt` file. To create your own environment, an example is:

```bash
conda create -n matchformer python=3.7
conda activate matchformer
cd /path/to/matchformer
pip install -r requirement.txt
```

### Datasets

You can prepare the test dataset in the same way as [LoFTR](https://github.com/zju3dv/LoFTR/blob/master/docs/TRAINING.md), place the dataset and index in the data directory.

A structure of dataset should be:

 ```
data
├── scannet
│   ├── index
│   │   ├── intrinsics.npz
│   │   ├── scannet_test.txt
│   │   └── test.npz
│   └── test
│   	├── scene0707_00
│   	├── ...
│   	└── scene0806_00
└── megadepth
    ├── index
    │	  ├── 0015_0.1_0.3.npz
    │	  ├── ...
    │	  ├── 0022_0.5_0.7.npz
    │	  └── megadepth_test_1500.txt
    └── test
    	  ├── Undistorted_SfM
    	  └── phoenix
 ```



### Evaluation

The evaluation configurations can be adjusted at `/config/defaultmf.py`

Pretrained model weights are hosted on Hugging Face at [InSAI-Lab/MatchFormer](https://huggingface.co/InSAI-Lab/MatchFormer). You can download them directly:

| Model | Setting | Size | Hugging Face Download |
| :--- | :--- | :---: | :--- |
| `indoor-large-SEA.ckpt` | Indoor (Large SEA) | 89.5 MB | [Download](https://huggingface.co/InSAI-Lab/MatchFormer/resolve/main/indoor-large-SEA.ckpt) |
| `indoor-lite-LA.ckpt` | Indoor (Lite LA) | 77.4 MB | [Download](https://huggingface.co/InSAI-Lab/MatchFormer/resolve/main/indoor-lite-LA.ckpt) |
| `outdoor-large-LA.ckpt` | Outdoor (Large LA) | 77.4 MB | [Download](https://huggingface.co/InSAI-Lab/MatchFormer/resolve/main/outdoor-large-LA.ckpt) |
| `outdoor-lite-SEA.ckpt` | Outdoor (Lite SEA) | 103.5 MB | [Download](https://huggingface.co/InSAI-Lab/MatchFormer/resolve/main/outdoor-lite-SEA.ckpt) |

Or download all weights via Hugging Face Hub CLI:
```bash
hf download InSAI-Lab/MatchFormer --local-dir model/weights
```

Put the weights at `model/weights`.

#### Indoor:

```
# adjust large SEA model config:
MATCHFORMER.BACKBONE_TYPE = 'largesea'
MATCHFORMER.SCENS = 'indoor'
MATCHFORMER.RESOLUTION = (8,2)
MATCHFORMER.COARSE.D_MODEL = 256
MATCHFORMER.COARSE.D_FFN = 256

python test.py /config/data/scannet_test_1500.py --ckpt_path /model/weights/indoor-large-SEA.ckpt --gpus=1 --accelerator="ddp"
```

```
# adjust lite LA model config:
MATCHFORMER.BACKBONE_TYPE = 'litela'
MATCHFORMER.SCENS = 'indoor'
MATCHFORMER.RESOLUTION = (8,4)
MATCHFORMER.COARSE.D_MODEL = 192
MATCHFORMER.COARSE.D_FFN = 192

python test.py /config/data/scannet_test_1500.py --ckpt_path /model/weights/indoor-lite-LA.ckpt --gpus=1 --accelerator="ddp"
```

#### Outdoor:

```
# adjust large LA model config:
MATCHFORMER.BACKBONE_TYPE = 'largela'
MATCHFORMER.SCENS = 'outdoor'
MATCHFORMER.RESOLUTION = (8,2)
MATCHFORMER.COARSE.D_MODEL = 256
MATCHFORMER.COARSE.D_FFN = 256

python test.py /config/data/megadepth_test_1500.py --ckpt_path /model/weights/outdoor-large-LA.ckpt --gpus=1 --accelerator="ddp"
```

```
# adjust lite SEA model config:
MATCHFORMER.BACKBONE_TYPE = 'litesea'
MATCHFORMER.SCENS = 'outdoor'
MATCHFORMER.RESOLUTION = (8,4)
MATCHFORMER.COARSE.D_MODEL = 192
MATCHFORMER.COARSE.D_FFN = 192

python test.py /config/data/megadepth_test_1500.py --ckpt_path /model/weights/outdoor-lite-SEA.ckpt --gpus=1 --accelerator="ddp"
```

### Training

Based on the LOFTER code to train MatchFormer, replace LoFTR/src/loftr/backbone/ with model/backbone/match_**.py to train.

### Citation

If you are interested in this work, please cite the following work:

```
@inproceedings{wang2022matchformer,
  title={MatchFormer: Interleaving Attention in Transformers for Feature Matching},
  author={Wang, Qing and Zhang, Jiaming and Yang, Kailun and Peng, Kunyu and Stiefelhagen, Rainer},
  booktitle={Asian Conference on Computer Vision},
  year={2022}
}
```

### Acknowledgments

Our work is based on [LoFTR](https://github.com/zju3dv/LoFTR) and we use their code.  We appreciate the previous open-source repository [LoFTR](https://github.com/zju3dv/LoFTR).
