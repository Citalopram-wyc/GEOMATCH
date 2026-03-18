# GEOMATCH

# GeoMatch

Cross-view Spatial Relation Matching between Natural Language and Aerial Imagery.

## Overview

Cross-view geolocalization—the task of matching images from different viewpoints such as drone and satellite—has emerged as a pivotal task in modern geospatial intelligence. However, lacking the capability for fine-grained object perception, current methods are largely confined to the scene-level registration stage, where their granularity and accuracy often fail to meet practical application demands. In terms of granularity, these methods lack the ability to accurately locate specific target objects based on descriptive attributes, especially in scenes with multiple similar objects. Furthermore, their accuracy is often compromised when faced with multiple visually similar candidate scenes (hard negatives), making it difficult to rank the correct match first. To overcome these limitations, we propose GeoMatch, a coarse-to-fine fine grained cross-view geolocalization and matching framework. It leverages hierarchical language descriptions to match scenes and objects, and uses fine-grained object matching results to optimize and re-rank the initial coarse scene retrieval. GeoMatch introduces a Task Adaptive Fusion (TAF) module that decouples scene-level feature alignment from object-level feature alignment. To bridge the viewpoint gap, we introduce a Cross-View Spatial Relation Reasoning (CV-SRR) module to convert the viewpoint-dependent spatial language into a universal geometric guidance signal. They are then fused into Adaptive Multimodal Probe (AMP) for high-precision matching and re-ranking. To promote and validate research in this area, we introduce the GeoMatch-1652 dataset, the first benchmark set that contains 49,726 manually annotated cross-view object-level correspondences. GeoMatch demonstrates highly competitive performance on this new benchmark set. In the satellite-to-drone (S2D) task, the model achieves 96.38% scene retrieval R@1 and 53.65% object matching accuracy (SR@1@0.5). In the reverse drone-to-satellite (D2S) task, R@1 and SR@1@0.5 are 94.02% and 42.53%, respectively. Compared to recent stateof-the-art (SOTA) methods, our model achieves absolute R@1 improvements of 0.79% and 1.39% in the S2D and D2S tasks, respectively. These results, validated by extensive ablation studies and robustness analysis, confirm the significant advantages of our language-guided stratification approach.

![Model Overview](assets/Geomatch_model_overview.png)



## Model Architecture

The model adopts a cross-modal architecture:
- **Visual Encoder**: Swin Transformer / CLIP ViT / ViT
- **Text Encoder**: BERT / RoBERTa
- **Cross-modal Alignment**: X-VLM framework

## Setup

### Environment

```bash
conda create -n geomatch python=3.8
conda activate geomatch
pip install -r requirements.txt
```

### Dependencies

See `requirements.txt` for detailed dependencies.

## Dataset

The GeoText-1652 dataset can be downloaded from baiduwangpan：GeoMatch
链接: https://pan.baidu.com/s/1d1lvuJFVfJSn9GDQW84nng?pwd=e15i 提取码: e15i 
--来自百度网盘超级会员v6的分享.

```
dataset/
├── images/           # Aerial imagery
├── annotations/     # Text descriptions and bounding boxes
└── splits/          # Train/val/test splits
```

## Training

```bash
python run.py --config config.yaml
```

### Training Options

Configure training parameters in `config.yaml` or via command line arguments.

## Evaluation

```bash
python match_eval.py --checkpoint path/to/checkpoint
```

## Project Structure

```
├── models/                    # Model architectures
│   ├── xvlm.py               # X-VLM cross-modal model
│   ├── swin_transformer.py   # Swin Transformer
│   ├── clip_vit.py           # CLIP ViT
│   ├── vit.py                # Vision Transformer
│   ├── xbert.py              # BERT module
│   ├── xroberta.py           # RoBERTa module
│   ├── model_match.py        # Matching model
│   └── model_re_bbox.py      # Relation extraction model
├── dataset/                    # Data loading
│   ├── bbox_match_dataset.py  # Bounding box matching dataset
│   └── re_bbox_dataset.py     # Relation extraction dataset
├── configs/                    # Configuration files
│   ├── text2match.yaml        # Text-to-match config
│   └── re_bbox.yaml           # Relation extraction config
├── utils/                      # Utility functions
│   ├── checkpointer.py        # Model checkpointing
│   └── cider/                 # CIDEr evaluation
├── accelerators/               # Training accelerators
├── run.py                      # Main entry point
├── config.yaml                 # Global config
└── README.md                   # This file
```



## License

Apache License 2.0 - See `LICENSE` file.
