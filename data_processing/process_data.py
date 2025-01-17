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
from data_processing.pose_quantizer import PoseQuantizer
import constants

from datasets import Dataset
from tqdm.auto import tqdm

def getSceneImg(element_path, file_name):
    img_path = os.path.join(element_path, file_name)
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

def getRawAction(element_path):
    action_path = os.path.join(element_path, "action.pkl")
    assert os.path.exists(action_path)

    with open(action_path, "rb") as f:
        action = pickle.load(f)

    isValid = True
    if action['pose0_position'].shape[0] != 1:
        isValid = False
        #print("WARNING MORE THAN ONE ACTION PER POSE! " + str(action['pose0_position'].shape[0]))
        #print(action)
    #print(action['pose0_position'])

    pose0 = (action['pose0_position'][0], action['pose0_rotation'][0])
    pose1 = (action['pose1_position'][0], action['pose1_rotation'][0])

    raw_action = [pose0, pose1]

    return raw_action, isValid

def getRawActionString(element_path):
    raw_action, isValid = getRawAction(element_path)
    raw_string = 'position0:'+ str(raw_action[0][0]) + ';rotation0:' +  str(raw_action[0][1]) + ';position1:'+ str(raw_action[1][0]) + ';rotation1:' +  str(raw_action[1][1])
    return raw_string, isValid

def getAction(element_path, pose_quantizer, bos=""):
    prefix_action = "action: "
    prefix_pose = "pose"
    prefix_rotation = "rotation"

    raw_action, _ = getRawAction(element_path)

    #Hard-coded for this task
    pp0 = prefix_pose + ": " + ''.join(pose_quantizer.encode_array(raw_action[0][0])) + " "
    pr0 = prefix_rotation + ": " + ''.join(pose_quantizer.encode_array(raw_action[0][1])) + " "
    pp1 = prefix_pose + ": " + ''.join(pose_quantizer.encode_array(raw_action[1][0])) + " "
    pr1 = prefix_rotation + ": " + ''.join(pose_quantizer.encode_array(raw_action[1][1])) + " "

    action_str = bos + prefix_action + pp0 + pr0 + prefix_action + pp1 + pr1
    return action_str

def getRawData(element_path):
    raw_dict = {}

    img_file_name = "0.jpg"

    img = getSceneImg(element_path, "rgb_top/0.jpg")
    trajectory = getTrajectory(element_path)
    #print(trajectory)
    prompt = getPrompt(trajectory, img)
    #print(prompt)

    pose_quantizer = getPoseQuantizer(trajectory)
    action = getAction(element_path, pose_quantizer)

    raw_action, isValid = getRawActionString(element_path)

    raw_dict = {
    'image_path': img_file_name,
    'prompt': prompt,
    'action': action,
    'raw_answer': raw_action
    }

    return raw_dict, isValid

def processDataPoint(element_path):
        point_dict = {}

        tokenizer = T5Tokenizer.from_pretrained(
        "t5-base", model_max_length=256, extra_ids=constants.VOCAB_EXTRA_IDS) 
        bos_token = tokenizer.bos_token if tokenizer.bos_token != None else tokenizer.convert_ids_to_tokens(0)

        #cached_path = os.path.join(element_path, "processed_data.pkl")
        #backup_cached_path = os.path.join(element_path, "backup_final_processed_data.pkl")
        #if not os.path.isfile(cached_path):
        img = getSceneImg(element_path, "rgb_top/0.jpg")
        image_encoder_input = getImgTensor(img)
        #print("IMAGE SHAPE: " + str(img.shape))
        #print("IMAGE TENSOR SHAPE: " + str(image_encoder_input.shape))
        #print(image_encoder_input)
        #TODO: convert float32 to bfloat16

        trajectory = getTrajectory(element_path)

        prompt = getPrompt(trajectory, img)
        text_encoder_input = tokenizer(prompt, max_length=64, truncation=True, padding='max_length')['input_ids']

        pose_quantizer = getPoseQuantizer(trajectory)
        action = getAction(element_path, pose_quantizer, bos_token)
        action = tokenizer(action, max_length=constants.ACTION_SEQUENCE_LENGTH, padding='max_length')['input_ids']
        text_decoder_target = action[1:]
        text_decoder_input = action[:-1]
        image_decoder_target = np.ndarray((1,1,1,),int)#Don't need so have it be size 1 to trigger the flag

        point_dict = {
        'image_encoder_input': image_encoder_input,
        'text_encoder_input': text_encoder_input,
        'text_decoder_input': text_decoder_input,
        #'text_decoder_masks': text_decoder_masks,
        'text_decoder_target': text_decoder_target,
        'image_decoder_target': image_decoder_target
        }
            #with open(cached_path, 'wb') as f:
            #    serialized = pickle.dumps(point_dict)
            #    pickle.dump(serialized, f, protocol=pickle.HIGHEST_PROTOCOL)
        #else:
            #with open(cached_path, 'rb') as f:
            #    point_dict = pickle.loads(pickle.load(f))
                #with open(backup_cached_path, 'wb') as bu:
                #    pickle.dumps(dict, bu, protocol=pickle.HIGHEST_PROTOCOL)
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

    #cached_dataset_path = os.path.join(path, "7_final_dataset.pkl")
    #backup_cached_dataset_path = os.path.join(path, "backup_dataset.pkl")

    #if not os.path.isfile(cached_dataset_path):
    train_path = os.path.join(path, "train")
    val_path = os.path.join(path, "val")
    test_path = os.path.join(path, "test")

    train = os.listdir(train_path)
    val = os.listdir(val_path)
    test = os.listdir(test_path)

    num_train = constants.DATASET_SAMPLE["train"] if constants.DATASET_SAMPLE["train"] < len(train) else len(train)
    num_val = constants.DATASET_SAMPLE["val"] if constants.DATASET_SAMPLE["val"] < len(val) else len(val)
    num_test = constants.DATASET_SAMPLE["test"] if constants.DATASET_SAMPLE["test"] < len(test) else len(test)

    train = train[0:num_train]
    val = val[0:num_val]
    test = test[0:num_test]

    print("Train Instances: " + str(num_train))
    print("Val Instances: " + str(num_val))
    print("Test Instances: " + str(num_test))

    train = Dataset.from_dict(getDataDict(train_path, train))
    val = Dataset.from_dict(getDataDict(val_path, val))
    test = Dataset.from_dict(getDataDict(test_path, test))

    dict = {
        'train': train,
        'val': val,
        'test': test
    }
        #with open(cached_dataset_path, 'wb') as f:
        #        serialized = pickle.dumps(dict)
        #        pickle.dump(serialized, f, protocol=pickle.HIGHEST_PROTOCOL)
    #else:
    #    with open(cached_dataset_path, 'rb') as f:
    #            dict = pickle.loads(pickle.load(f))

    return dict