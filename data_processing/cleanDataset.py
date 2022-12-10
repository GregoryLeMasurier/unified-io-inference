import argparse
from torch.utils.data import Dataset, DataLoader
import os
import csv
from data_processing import process_data
from tqdm.auto import tqdm
import shutil

def sort_files(lst):
  lst.sort(key=lambda x: int(os.path.basename(x)))
  return lst

parser = argparse.ArgumentParser()
parser.add_argument(
    "--overwrite",
    type=bool,
    default=False,
)
parser.add_argument("--path")
parser.add_argument(
    "--new_path",
    default=None
)

args = parser.parse_args()

train_path = os.path.join(args.path, "train")
val_path = os.path.join(args.path, "val")
test_path = os.path.join(args.path, "test")
train = os.listdir(train_path)
train = [os.path.join("train", element) for element in train]
val = os.listdir(val_path)
val = [os.path.join("val", element) for element in val]
test = os.listdir(test_path)
test = [os.path.join("test", element) for element in test]

full_dataset = train + val + test
full_dataset = sort_files(full_dataset)

save_file_name = 'raw_data.csv'
invalid_paths = []
for sample in tqdm(full_dataset, total=len(full_dataset)):
    dest_full_path = os.path.join(args.path, sample)
    #full_path = "/media/NLP/simple_manipulation/train/000143"
    raw_data, isValid = process_data.getRawData(dest_full_path)
    if not isValid:
        invalid_paths.append(sample)
        continue
    src_save_file = os.path.join(dest_full_path, save_file_name)
    if args.overwrite or not os.path.isfile(src_save_file):
        #print(save_file)
        with open(src_save_file, 'w') as f:
            w = csv.DictWriter(f, raw_data.keys())
            w.writeheader()
            w.writerow(raw_data)
print(len(invalid_paths))
invalid_path = os.path.join(args.path, 'invalid.csv')
with open(invalid_path,'w') as f:
    w = csv.writer(f)
    for v in invalid_paths:
        w.writerow([v])
for invalid_sample in invalid_paths:
    full_dataset.remove(invalid_sample)
    
valid_path = os.path.join(args.path, 'valid.csv')
with open(valid_path,'w') as f:
    w = csv.writer(f)
    for v in full_dataset:
        w.writerow([v])
    #return

if args.new_path != None:
    if not os.path.exists(args.new_path):
        os.makedirs(args.new_path)

    new_train_path = os.path.join(args.new_path, "train")
    if not os.path.exists(new_train_path):
        os.makedirs(new_train_path)
    new_val_path = os.path.join(args.new_path, "val")
    if not os.path.exists(new_val_path):
        os.makedirs(new_val_path)
    new_test_path = os.path.join(args.new_path, "test")
    if not os.path.exists(new_test_path):
        os.makedirs(new_test_path)


    for sample in tqdm(full_dataset, total=len(full_dataset)):
        src_full_path = os.path.join(args.path, sample)

        dest_full_path = os.path.join(args.new_path, sample)
        if not os.path.exists(dest_full_path):
            os.makedirs(dest_full_path)
        src_img_path = os.path.join(src_full_path, "rgb_top/0.jpg")
        dest_img_path = os.path.join(dest_full_path, "0.jpg")
        assert os.path.exists(src_img_path)
        shutil.copy(src_img_path, dest_img_path)

        src_save_file = os.path.join(src_full_path, save_file_name)
        dest_save_file = os.path.join(dest_full_path, save_file_name)
        assert os.path.exists(src_save_file)
        shutil.copy(src_save_file, dest_save_file)