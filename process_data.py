import os
import numpy as np
import numpy.ma as ma

import pickle
from PIL import Image
import re

import torch
from torchvision.ops import masks_to_boxes
from transformers import T5Tokenizer

from uio import utils
from pose_quantizer import PoseQuantizer

from datasets import Dataset


def getScenes(path, data):
    image_tensors = []
    for data_point in data:
        element_path = os.path.join(path, data_point)
        img_path = os.path.join(element_path, "rgb_top/0.jpg")
        assert os.path.exists(img_path)
        img = None
        with Image.open(img_path) as scene:
            img = np.asarray(scene)
            if img is not None:
                assert len(img.shape) == 3 and img.shape[-1] == 3
            tensor, _mask = utils.preprocess_image(img, None)
            image_tensors.append(tensor)
    return np.stack(image_tensors)

def getPrompts(path, data):
    #Compute Prompt Strings
    prompts = []
    for data_point in data:
        element_path = os.path.join(path, data_point)

        img_path = os.path.join(element_path, "rgb_top/0.jpg")
        assert os.path.exists(img_path)
        img = None
        with Image.open(img_path) as scene:
            img = np.asarray(scene)

        trajectory_path = os.path.join(element_path, "trajectory.pkl")
        assert os.path.exists(trajectory_path)

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

        assert len(obj_names) == len(boxes)

        for obj_name, bbox in zip(obj_names, boxes):
            region = utils.region_to_tokens(bbox, img.shape[1], img.shape[0])
            orig_prompt_str = "{" + obj_name + "}"
            region_tokens = "\" " + "".join(region) + " \""
            prompt = prompt.replace(orig_prompt_str, region_tokens)
        prompt = prompt.replace('.', ' . ')
        prompts.append(prompt)

    #Tokenize Prompts     
    tokenizer = T5Tokenizer.from_pretrained(
        "t5-base", model_max_length=256, extra_ids=1200)    
    tokenized_prompts = tokenizer(prompts, max_length=64, truncation=True, padding='max_length')
    return tokenized_prompts['input_ids']

def getTrajectoryBounds(path, data):
    element_path = os.path.join(path, data[0]) # Assume all trajectories have the same bounds (safe for this task at least)
    trajectory_path = os.path.join(element_path, "trajectory.pkl")
    assert os.path.exists(trajectory_path)

    with open(trajectory_path, "rb") as f:
        trajectory = pickle.load(f)
    bounds = trajectory['action_bounds']

    return (min(bounds['low']), max(bounds['high']))

def getActions(path, data):
    #actions =["pick","place","push","wipe"]

    prefix_action = "action: "
    prefix_pose = "pose"
    prefix_rotation = "rotation"

    action_seq = []
    pose_quantizer = None
    for data_point in data:
        element_path = os.path.join(path, data_point)
        action_path = os.path.join(element_path, "action.pkl")
        assert os.path.exists(action_path)

        with open(action_path, "rb") as f:
            action = pickle.load(f)

        if pose_quantizer == None:
            low,high = getTrajectoryBounds(path, data)
            pose_quantizer = PoseQuantizer(low, high, 100)

        #Hard-coded for this task
        pp0 = prefix_pose + ": " + ''.join(pose_quantizer.encode_array(action['pose0_position'][0])) + " "
        pr0 = prefix_rotation + ": " + ''.join(pose_quantizer.encode_array(action['pose0_rotation'][0])) + " "
        pp1 = prefix_pose + ": " + ''.join(pose_quantizer.encode_array(action['pose1_position'][0])) + " "
        pr1 = prefix_rotation + ": " + ''.join(pose_quantizer.encode_array(action['pose1_rotation'][0])) + " "

        action_str = prefix_action + pp0 + pr0 + prefix_action + pp1 + pr1
        action_seq.append(action_str)
    #Tokenize Prompts     
    tokenizer = T5Tokenizer.from_pretrained(
        "t5-base", model_max_length=256, extra_ids=1200)    
    tokenized_actions = tokenizer(action_seq, max_length=64, truncation=True, padding='max_length')
    return tokenized_actions['input_ids']

def getDataDict(path, data):
    image_encoder_inputs = getScenes(path, data)

    text_encoder_inputs = getPrompts(path, data)

    actions = getActions(path, data)
    text_decoder_inputs = []
    text_decoder_targets = []
    for action in actions:
        text_decoder_targets.append(action)
        action.insert(0,0)
        text_decoder_inputs.append(action)

    data_dict = {
      'image_encoder_inputs': np.array(image_encoder_inputs),
      'text_encoder_inputs': np.array(text_encoder_inputs),
      'text_decoder_inputs': np.array(text_decoder_inputs),
      'text_decoder_targets': np.array(text_decoder_targets),
      'image_decoder_targets': np.ndarray((len(data),1,1,3,),int) #Don't need so have it be size 1 to trigger the flag
    }
    return data_dict

def getDataset(path): 
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")
    test_path = os.path.join(path, "test")

    train = os.listdir(train_path)
    train = train[0:10]
    val = os.listdir(val_path)
    val = val[0:10]
    test = os.listdir(test_path)
    test = test[0:10]

    print("Train Instances: " + str(len(train)))
    print("Val Instances: " + str(len(val)))
    print("Test Instances: " + str(len(test)))

    dict = {
        'train': Dataset.from_dict(getDataDict(train_path, train)),
        'val': Dataset.from_dict(getDataDict(val_path, val)),
        'test': Dataset.from_dict(getDataDict(test_path, test))
    }

    return dict