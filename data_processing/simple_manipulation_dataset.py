from torch.utils.data import Dataset
import os
from data_processing import process_data
from transformers import T5Tokenizer
import constants
import numpy as np
import csv
import re

def sort_files(lst):
  lst.sort(key=lambda x: int(os.path.basename(x)))
  return lst

class SimpleManipulationDataset(Dataset):
    def __init__(self, path):
      self.path = path
      self.data_samples = os.listdir(path)

      self.tokenizer = T5Tokenizer.from_pretrained(
      "t5-base", model_max_length=256, extra_ids=constants.VOCAB_EXTRA_IDS)
      self.bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token != None else self.tokenizer.convert_ids_to_tokens(0)

      self.data_file = 'raw_data.csv'

    def __len__(self):
      return len(self.data_samples)

    def __getitem__(self, idx):

      element_path = os.path.join(self.path, self.data_samples[idx])
      assert os.path.exists(element_path)

      raw_data_file = os.path.join(element_path, self.data_file)
      assert os.path.exists(raw_data_file)
      raw_data = next(csv.DictReader(open(raw_data_file)))

      img = process_data.getSceneImg(element_path, raw_data['image_path'])
      image_encoder_input = process_data.getImgTensor(img)

      text_encoder_input = self.tokenizer(raw_data['prompt'], max_length=64, truncation=True, padding='max_length')['input_ids']

      action = self.bos_token + raw_data['action']
      action = self.tokenizer(action, max_length=constants.ACTION_SEQUENCE_LENGTH, padding='max_length')['input_ids']
      text_decoder_target = action[1:]
      text_decoder_input = action[:-1]

      image_decoder_target = np.ndarray((1,1,1,),int)#Don't need so have it be size 1 to trigger the flag

      raw_pose_str = raw_data['raw_answer']
      res = re.findall(r'\[.*?\]', raw_pose_str)
      res = ' '.join(res)
      res = res.replace('[','').replace(']','')
      res = [float(x) for x in res.split()]

      item = {
      'image_encoder_input': image_encoder_input,
      'text_encoder_input': text_encoder_input,
      'text_decoder_input': text_decoder_input,
      'text_decoder_target': text_decoder_target,
      'image_decoder_target': image_decoder_target,
      'raw_poses': res
      }

      return item