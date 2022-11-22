import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer
import numpy as np
import numpy.ma as ma
import os
import argparse
from tqdm.auto import tqdm
import wandb
import pickle
import torch
from torchvision.ops import masks_to_boxes
from PIL import Image
import re
from uio import utils

# dataset splits should be numpy array
def train_and_evaluate(train_dataset, eval_dataset, test_dataset, state, epochs):
    #TODO: Get the cardinality from dataset
    num_train_batches = 0
    num_eval_batches = 0
    num_test_batches = 0
   
    for epoch in tqdm(range(1, epochs + 1)):
        best_eval_loss = 1e6

        # ============== Training ============== #
        train_batch_metrics = []
        train_datagen = iter(train_dataset)
        for batch_idx in range(num_train_batches):
            batch = next(train_datagen)
            state, metrics = train_step(state, batch)
            train_batch_metrics.append(metrics)
        train_batch_metrics = accumulate_metrics(train_batch_metrics)


#Skip Val for now
        # ============== Validation ============= #
        #eval_batch_metrics = []
        #eval_datagen = iter(eval_dataset)
        #for batch_idx in range(num_eval_batches):
        #    batch = next(eval_datagen)
        #    metrics = eval_step(state, batch)
        #    eval_batch_metrics.append(metrics)
        #eval_batch_metrics = accumulate_metrics(eval_batch_metrics)


        # Log Metrics to Weights & Biases
        wandb.log({
            "Train Loss": train_batch_metrics['loss'],
            "Train Accuracy": train_batch_metrics['accuracy'],
            #"Validation Loss": eval_batch_metrics['loss'],
            #"Validation Accuracy": eval_batch_metrics['accuracy']
        }, step=epoch)


    return state

def accumulate_metrics(metrics):
    metrics = jax.device_get(metrics)
    return {
        k: np.mean([metric[k] for metric in metrics])
        for k in metrics[0]
    }

@jax.jit
def train_step(
    state: train_state.TrainState, batch: jnp.ndarray
):
    image, label = batch


    def loss_fn(params):
        logits = state.apply_fn({'params': params}, image)
        loss = cross_entropy_loss(logits=logits, labels=label)
        return loss, logits


    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=label)
    return state, metrics

#@jax.jit
#def eval_step(state, batch):
#    image, label = batch
#    logits = state.apply_fn({'params': state.params}, image)
#    return compute_metrics(logits=logits, labels=label)

def cross_entropy_loss(*, logits, labels):
    one_hot_encoded_labels = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(
        logits=logits, labels=one_hot_encoded_labels
    ).mean()

def compute_metrics(*, logits, labels):
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics

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
            #tensor = np.stack(tensor)#They did this? Do I really need it?
            image_tensors.append(tensor)
    return image_tensors

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
        "t5-base", model_max_length=256, extra_ids=1100)    
    tokenized_prompts = tokenizer(prompts, max_length=64, truncation=True, padding='max_length')
    return tokenized_prompts['input_ids']

def getActions(path, data):
    actions =["pick","place","push","wipe"]

    hasOne = False

    for data_point in data:
        element_path = os.path.join(path, data_point)
        trajectory_path = os.path.join(element_path, "trajectory.pkl")
        assert os.path.exists(trajectory_path)

        with open(trajectory_path, "rb") as f:
            trajectory = pickle.load(f)

        if not hasOne:
            print(trajectory)

        prompt = trajectory.pop('prompt')
        prompt_assets = trajectory.pop('prompt_assets')

        obj_names = re.findall(r'\{.*?\}', prompt)

        masks = []

        for i in range(0,len(obj_names)):
            obj_names[i] = obj_names[i].replace('{', '').replace('}', '')
            obj_asset = prompt_assets.pop(obj_names[i])
            obj_mask = obj_asset.get('segm').get('top')
            bool_mask = np.logical_not(ma.make_mask(obj_mask))
            if not hasOne:
                print(obj_mask)
                hasOne = True
            masks.append(bool_mask)

        masks = np.asarray(masks)
        torch_masks = torch.from_numpy(masks)
    
        boxes = masks_to_boxes(torch_masks)
        boxes = boxes.numpy()

def getBatch(path, data): 
    image_encoder_inputs = getScenes(path, data)
    #print(image_encoder_inputs)

    text_encoder_inputs = getPrompts(path, data)
    #print(text_encoder_inputs)

    text_decoder_inputs = getActions(path, data)
    text_decoder_targets = []

    #print([tokenizer.bos_token_id] + res)
    #print(res + [tokenizer.eos_token_id])

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    train_path = os.path.join(args.path, "train")
    val_path = os.path.join(args.path, "val")
    test_path = os.path.join(args.path, "test")

    train = os.listdir(train_path)
    train = train[0:100]
    val = os.listdir(val_path)
    val = val[0:10]
    test = os.listdir(test_path)
    test = test[0:10]

    print(len(train))
    print(len(val))
    print(len(test))

    getBatch(train_path, train)
    