## Preparation
- Install Densepose: 
```
pip install "git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose"
```

- Download checkpoint:
```
mkdir ckpt
curl -o ckpt/final.pkl https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_101_FPN_s1x_legacy/164832182/model_final_10af0e.pkl
```

## Inference
- This command will extract Densepose as `.npy` format for all JPG images in `input` folder and save them to `output` folder. 
```
bash ./run.sh
```
