import argparse
from os.path import exists, join, dirname

from PIL import Image

from uio import runner
from uio.configs import CONFIGS
import numpy as np
import numpy.ma as ma

from absl import logging
import warnings

import pickle

import torch
from torchvision.ops import masks_to_boxes

import sys
import re

warnings.simplefilter(action='ignore', category=FutureWarning)
logging.set_verbosity(logging.INFO)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_size", choices=list(CONFIGS))
  parser.add_argument("model_weights")
  parser.add_argument("path")
  args = parser.parse_args()

#  seg_path = join(dirname(args.path), "obs.pkl")
#  assert exists(seg_path)
  img_path = join(dirname(args.path), "rgb_top/0.jpg")
  assert exists(img_path)
  trajectory_path = join(dirname(args.path), "trajectory.pkl")
  assert exists(trajectory_path)

  with open(trajectory_path, "rb") as f:
    trajectory = pickle.load(f)

  prompt = trajectory.pop('prompt')
  prompt_assets = trajectory.pop('prompt_assets')

  obj_names = re.findall(r'\{.*?\}', prompt)

  masks = []

  for i in range(0,len(obj_names)):
    obj_names[i] = obj_names[i].replace('{', '').replace('}', '')
    obj_asset = prompt_assets.pop(obj_names[i])
    obj_mask = obj_asset.get('segm').get('top')
    bool_mask = np.logical_not(ma.make_mask(obj_mask))
    masks.append(bool_mask)

  masks = np.asarray(masks)
  torch_masks = torch.from_numpy(masks)
    
  boxes = masks_to_boxes(torch_masks)
  boxes = boxes.numpy()

  model = runner.ModelRunner(args.model_size, args.model_weights)
  with Image.open(img_path) as scene:
      output = model.simple_manipulation(np.asarray(scene), prompt, obj_names, boxes)
  print(output["text"])


if __name__ == "__main__":
  main()
