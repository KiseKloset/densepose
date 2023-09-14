import argparse
import torch
import numpy as np
import cv2

from pathlib import Path
from tqdm import tqdm

WIDTH = 192
HEIGHT = 256

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", help="output folder of densepose")
    args = parser.parse_args()


    with open('output.pkl', 'rb') as f:
        datas = torch.load(f)

        for data in tqdm(datas):
            filename = Path(data['file_name']).stem
            out_path = Path(args.out) / f"{filename}.npy"
            try:
                densepose = np.zeros((HEIGHT, WIDTH)).astype(np.uint8)
                x1, y1, x2, y2 = [int(v) for v in data['pred_boxes_XYXY'][0].cpu().numpy()]
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                mask = data['pred_densepose'].cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (w, h), interpolation = cv2.INTER_NEAREST)

                densepose[y:y+h, x:x+w] = mask
                np.save(out_path, densepose)
            except Exception as e:
                print(filename, e)

