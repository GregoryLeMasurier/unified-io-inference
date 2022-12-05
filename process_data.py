import os
import numpy as np
import numpy.ma as ma

import pickle
import json
from PIL import Image
import re

import torch
from torchvision.ops import masks_to_boxes
from transformers import T5Tokenizer

from uio import utils
from pose_quantizer import PoseQuantizer

from datasets import Dataset
from tqdm.auto import tqdm
from numpy_encoder import NumpyEncoder

def getSceneImg(element_path):
    img_path = os.path.join(element_path, "rgb_top/0.jpg")
    assert os.path.exists(img_path)
    img = None
    with Image.open(img_path) as scene:
        img = np.asarray(scene)
        if img is not None:
            assert len(img.shape) == 3 and img.shape[-1] == 3
    return img

def getImgTensor(img):
    tensor, _mask = utils.preprocess_image(img, None)
    return tensor

def getPrompt(trajectory, img):
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

    assert len(obj_names) == len(boxes)

    for obj_name, bbox in zip(obj_names, boxes):
        region = utils.region_to_tokens(bbox, img.shape[1], img.shape[0])
        orig_prompt_str = "{" + obj_name + "}"
        region_tokens = "\" " + "".join(region) + " \""
        prompt = prompt.replace(orig_prompt_str, region_tokens)
    prompt = prompt.replace('.', ' . ')
    return prompt

def getTrajectory(element_path):
    trajectory_path = os.path.join(element_path, "trajectory.pkl")
    assert os.path.exists(trajectory_path)

    with open(trajectory_path, "rb") as f:
        trajectory = pickle.load(f)
    return trajectory

def getPoseQuantizer(trajectory):
    bounds = trajectory['action_bounds']
    return PoseQuantizer(min(bounds['low']), max(bounds['high']), 100)

def getAction(element_path, pose_quantizer):
    prefix_action = "action: "
    prefix_pose = "pose"
    prefix_rotation = "rotation"

    action_path = os.path.join(element_path, "action.pkl")
    assert os.path.exists(action_path)

    with open(action_path, "rb") as f:
        action = pickle.load(f)

    #Hard-coded for this task
    pp0 = prefix_pose + ": " + ''.join(pose_quantizer.encode_array(action['pose0_position'][0])) + " "
    pr0 = prefix_rotation + ": " + ''.join(pose_quantizer.encode_array(action['pose0_rotation'][0])) + " "
    pp1 = prefix_pose + ": " + ''.join(pose_quantizer.encode_array(action['pose1_position'][0])) + " "
    pr1 = prefix_rotation + ": " + ''.join(pose_quantizer.encode_array(action['pose1_rotation'][0])) + " "

    action_str = prefix_action + pp0 + pr0 + prefix_action + pp1 + pr1
    return action_str

def processDataPoint(element_path):
        point_dict = {}

        tokenizer = T5Tokenizer.from_pretrained(
        "t5-base", model_max_length=256, extra_ids=1200) 

        cached_path = os.path.join(element_path, "processed_data.json")
        #backup_cached_path = os.path.join(element_path, "backup_final_processed_data.pkl")
        if not os.path.isfile(cached_path):
            img = getSceneImg(element_path)
            image_encoder_input = getImgTensor(img)
            #print("IMAGE SHAPE: " + str(img.shape))
            #print("IMAGE TENSOR SHAPE: " + str(image_encoder_input.shape))
            #print(image_encoder_input)
            #TODO: convert float32 to bfloat16
 
            trajectory = getTrajectory(element_path)

            prompt = getPrompt(trajectory, img)
            text_encoder_input = tokenizer(prompt, max_length=64, truncation=True, padding='max_length')['input_ids']

            pose_quantizer = getPoseQuantizer(trajectory)
            action = getAction(element_path, pose_quantizer)
            action = tokenizer(action, max_length=64, truncation=True, padding='max_length')['input_ids']
            text_decoder_target = action
            seq_length = np.array(text_decoder_target).shape[-1]
            #text_decoder_masks = np.ones((seq_length, seq_length))
            #text_decoder_masks = np.triu(text_decoder_masks, 1)
            action.insert(0,0)
            text_decoder_input = action
            image_decoder_target = np.ndarray((1,1,1,),int)#Don't need so have it be size 1 to trigger the flag

            point_dict = {
            'image_encoder_input': image_encoder_input,
            'text_encoder_input': text_encoder_input,
            'text_decoder_input': text_decoder_input,
            #'text_decoder_masks': text_decoder_masks,
            'text_decoder_target': text_decoder_target,
            'image_decoder_target': image_decoder_target
            }

            serialized_point_dict = json.dumps(point_dict, cls=NumpyEncoder)
            with open(cached_path, 'w') as f:
                json.dump(serialized_point_dict, f)
        else:
            with open(cached_path, 'r') as f:
                point_dict = json.load(f)
                point_dict = json.loads(point_dict)
                #with open(backup_cached_path, 'wb') as bu:
                #    pickle.dump(dict, bu, protocol=pickle.HIGHEST_PROTOCOL)
            #print("USING CACHED DATA: " + cached_path)
        return point_dict



def getDataDict(path, data):
    image_encoder_inputs = []
    text_encoder_inputs = []
    text_decoder_inputs = []
    text_decoder_targets = []
    image_decoder_targets = []  
    #text_decoder_masks = []

    for data_point in tqdm(data):
        element_path = os.path.join(path, data_point)
        point_dict = processDataPoint(element_path)
        image_encoder_inputs.append(point_dict['image_encoder_input'])
        text_encoder_inputs.append(point_dict['text_encoder_input'])
        text_decoder_inputs.append(point_dict['text_decoder_input'])
        text_decoder_targets.append(point_dict['text_decoder_target'])
        image_decoder_targets.append(point_dict['image_decoder_target'])
        #text_decoder_masks.append(point_dict['text_decoder_masks'])

    data_dict = {
    'image_encoder_inputs': np.array(image_encoder_inputs),
    'text_encoder_inputs': np.array(text_encoder_inputs),
    'text_decoder_inputs': np.array(text_decoder_inputs),
    #'text_decoder_masks': np.array(text_decoder_masks),
    'text_decoder_targets': np.array(text_decoder_targets),
    'image_decoder_targets': np.array(image_decoder_targets)
    }

    return data_dict

def getDataset(path): 
    dict = {}

    cached_dataset_path = os.path.join(path, "final_dataset.json")
    #backup_cached_dataset_path = os.path.join(path, "backup_dataset.pkl")

    if not os.path.isfile(cached_dataset_path):
        train_path = os.path.join(path, "train")
        val_path = os.path.join(path, "val")
        test_path = os.path.join(path, "test")

        train = os.listdir(train_path)
        train = train[0:2000]#4000]
        val = os.listdir(val_path)
        val = val[0:100]
        test = os.listdir(test_path)
        test = test[0:100]

        print("Train Instances: " + str(len(train)))
        print("Val Instances: " + str(len(val)))
        print("Test Instances: " + str(len(test)))

        train = Dataset.from_dict(getDataDict(train_path, train))
        val = Dataset.from_dict(getDataDict(val_path, val))
        test = Dataset.from_dict(getDataDict(test_path, test))

        dict = {
            'train': train,
            'val': val,
            'test': test
        }
        serialized_dict = json.dumps(dict, cls=NumpyEncoder)
        with open(cached_dataset_path, 'w') as f:
                json.dump(serialized_dict, f)
    else:
        with open(cached_dataset_path, 'r') as f:
            dict = json.load(f)
            dict = json.loads(dict)
            #with open(backup_cached_dataset_path, 'wb') as bu:
            #    pickle.dump(dict, bu, protocol=pickle.HIGHEST_PROTOCOL)
            

    return dict