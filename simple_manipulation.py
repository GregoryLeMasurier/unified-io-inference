import argparse
from os.path import exists, join

from PIL import Image

from uio import runner
from uio.configs import CONFIGS
import numpy as np

from absl import logging
import warnings

import pickle

# flax kicks up a lot of future warnings at the moment, ignore them
warnings.simplefilter(action='ignore', category=FutureWarning)

# To see INFO messages from `ModelRunner`
logging.set_verbosity(logging.INFO)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_size", choices=list(CONFIGS))
  parser.add_argument("model_weights")
  parser.add_argument("seg_path")
  parser.add_argument("img_path")
  args = parser.parse_args()

  seg_path = args.seg_path
  assert exists(seg_path)
  img_path = args.img_path
  assert exists(img_path)

  with open(seg_path, "rb") as f:
    obs = pickle.load(f)
  segm = obs.pop("segm")
  print("Obs")
  for k, v in segm.items():
    print(f"Segm {k} view : {v.shape}")
    seg1 = Image.fromarray(v[0])
    seg2 = Image.fromarray(v[1])


  model = runner.ModelRunner(args.model_size, args.model_weights)
#  image = np.array(seg1.convert('RGB'))
#  image2 = np.array(seg2.convert('RGB'))
  with Image.open(img_path) as scene:
      output = model.simple_manipulation(np.asarray(scene), [seg1,seg2])
  print(output["text"])


if __name__ == "__main__":
  main()
