# KEAT-TemporalGNN
Repository for AAAI-2026 paper **Kernelized Edge Attention: Addressing Semantic Attention Blurring in Temporal Graph Neural Networks**

[![arXiv](https://img.shields.io/badge/arXiv-2602.00596-orange.svg)](https://arxiv.org/abs/2602.00596)

---

## Create and activate the Environment

Refer to ```environment.yml``` for package details.


## TGN-KEAT

### Model Training

```KEAT_TGN/train.py```

### Model Testing
```KEAT_TGN/test.py```

Results on tgbl-wiki are in:

```KEAT_TGN/keat_tgn.ipynb```

## TGN-DyGFormer

We follow original DyGFormer steps for training and evaluation : [KEAT_DyGFormer/README.md](./KEAT_DyGFormer/README.md)

## Code Credits

KEAT codebase is built on top of following repos:

- [TGB](https://github.com/shenyangHuang/TGB/tree/main) 
- [DyGFormer](https://github.com/yule-BUAA/DyGLib)
- [LeTE](https://github.com/chenxi1228/LeTE)
