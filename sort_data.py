import os
import argparse
import shutil

from sklearn.model_selection import train_test_split


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    data = os.listdir(args.path)

    #50065 samples
    #.8 ; .1 ; .1

    X_train, X_test, _tr, _te = train_test_split(data, data, test_size=0.2, random_state=42)
    X_val, X_test, _tr, _te = train_test_split(X_test, X_test, test_size=0.5, random_state=42)

    print(len(data))
    print(len(X_train))
    print(len(X_val))
    print(len(X_test))

    train_path = os.path.join(args.path, "train")
    val_path = os.path.join(args.path, "val")
    test_path = os.path.join(args.path, "test")


    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for data_dir in X_train:
        data_path = os.path.join(args.path, data_dir)
        shutil.move(data_path, train_path)

    for data_dir in X_val:
        data_path = os.path.join(args.path, data_dir)
        shutil.move(data_path, val_path)
    
    for data_dir in X_test:
        data_path = os.path.join(args.path, data_dir)
        shutil.move(data_path, test_path)
