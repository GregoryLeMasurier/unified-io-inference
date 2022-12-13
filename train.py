#Gregory LeMasurier

# Most Jax functions were slightly / completely rewritten following the guide here: https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-FLAX--VmlldzoyMzA4ODEy
import argparse
import wandb

import numpy as np

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
from flax.training.common_utils import shard

from uio import utils
from uio import network
from uio.configs import CONFIGS, VAE_CONFIG
from uio.model import UnifiedIOModel

from data_processing import simple_manipulation_dataset, custom_dataloader
from data_processing.pose_quantizer import PoseQuantizer
import constants
from datasets import Dataset
import time
import os
import re
import math

from transformers import T5Tokenizer

#TODO: Warmup?
def init_train_state(
    model, params, learning_rate
) -> train_state.TrainState:
    optimizer = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn = model.module.apply,
        tx=optimizer,
        params=params
    )

#This function was taken from here: https://programtalk.com/vs4/python/huggingface/transformers/examples/flax/token-classification/run_flax_ner.py/
def train_data_collator(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int):
    """Returns shuffled batches of size `batch_size` from truncated `train dataset`, sharded over all local devices."""
    steps_per_epoch = len(dataset) // batch_size
    perms = jax.random.permutation(rng, len(dataset))
    perms = perms[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    perms = perms.reshape((steps_per_epoch, batch_size))

    for perm in perms:
        batch = dataset[perm]
        batch = {k: np.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch

#This function was taken from here: https://programtalk.com/vs4/python/huggingface/transformers/examples/flax/token-classification/run_flax_ner.py/
def eval_data_collator(dataset: Dataset, batch_size: int):
    """Returns batches of size `batch_size` from `eval dataset`, sharded over all local devices."""
    for i in range(len(dataset) // batch_size):
        batch = dataset[i * batch_size : (i + 1) * batch_size]
        batch = {k: np.array(v) for k, v in batch.items()}
        batch = shard(batch)

        yield batch

def train_and_evaluate(dataloaders, state, rng, config, run_evaluation): 
    train_dataloader = dataloaders['train']  
    val_dataloader = dataloaders['val']  
    test_dataloader = dataloaders['test']  

    #bs = config['batch_size']
    epochs = config['epochs']
    checkpoint_path = config['checkpoint_path']
    enable_wandb = config['enable_wandb']

    step_per_epoch = len(train_dataloader)# // bs
    total_steps = step_per_epoch * epochs

    for epoch in tqdm(range(1, epochs + 1)):
        rng, input_rng = jax.random.split(rng)

        # ============== Training ============== #
        train_batch_metrics = []
        for step, batch in enumerate(
            tqdm(
                train_dataloader,
                total=step_per_epoch,
                desc="Training...",
                position=1,
            )
        ):
            state, metrics = train_step(state, batch)
            train_batch_metrics.append(metrics)
            if checkpoint_path != None and step == (step_per_epoch - 1):
                checkpoint_prefix = "checkpoint_{}_epoch_".format(time.strftime("%Y%m%d-%H%M%S"))
                checkpoints.save_checkpoint(ckpt_dir=checkpoint_path, target=state.params, prefix=checkpoint_prefix,step=epoch)
        train_batch_metrics = accumulate_metrics(train_batch_metrics)
        # Log Metrics to Weights & Biases
        if enable_wandb:
            wandb.log({
                "Train Loss": train_batch_metrics['loss'],
                "Train Accuracy": train_batch_metrics['accuracy']
            }, step=epoch)

        # ============== Validation ============= #
        val_batch_metrics = []
        for v_step, batch in enumerate(
            tqdm(
                val_dataloader,
                total=len(val_dataloader),# // bs,
                desc="Validating ...",
                position=2,
            )
        ):
            metrics = eval_step(state, batch)
            val_batch_metrics.append(metrics)
        val_batch_metrics = accumulate_metrics(val_batch_metrics)


        # Log Metrics to Weights & Biases
        if enable_wandb:
            wandb.log({
                "Validation Accuracy": val_batch_metrics['accuracy']
            }, step=epoch)

    # ============== Test ============= #
    # to only run test, run with epoch = 0
    if run_evaluation:
        test_batch_metrics = []
        for t_step, batch in enumerate(
            tqdm(
                test_dataloader,
                total=len(test_dataloader),
                desc="Evaluating ...",
                position=3,
            )
        ):
            metrics = test_step(state, batch)
            test_batch_metrics.append(metrics)
        test_batch_metrics = accumulate_metrics(test_batch_metrics)

        # Log Metrics to Weights & Biases
        if enable_wandb:
            wandb.log({
                "Test Accuracy": test_batch_metrics['accuracy'],
                "Test Position Accuracy": test_batch_metrics['position_accuracy'],
                "Test Euclidean Dist 3D": test_batch_metrics['euclidean_distance_3d'],
                "Test Euclidean Dist 2D": test_batch_metrics['euclidean_distance_2d'],
                "Test Quantizer_Error": test_batch_metrics['quantizer_error']
            }, step=epochs)

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
#    import ipdb
#    ipdb.set_trace()
    image=batch['image_encoder_inputs'].squeeze(0) # (1,1,384,384,3) -> (1,384,384,3))
    prompt=batch['text_encoder_inputs'].squeeze(0)
    actions_in=batch['text_decoder_inputs'].squeeze(0)
    actions_out=batch['text_decoder_targets'].squeeze(0)
    image_out=batch['image_decoder_targets'].squeeze(0)

    def loss_fn(params):
        logits = state.apply_fn({'params': params},
                                enable_dropout=True, 
                                image_encoder_inputs=image, 
                                text_encoder_inputs=prompt, 
                                text_decoder_inputs=actions_in, 
                                text_decoder_targets=actions_out, 
                                image_decoder_targets=image_out)
        logits = logits[0] #only use text logits
        loss = cross_entropy_loss(logits=logits, labels=actions_out)
        return loss, logits


    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=actions_out)
    return state, metrics

@jax.jit
def eval_step(
    state: train_state.TrainState, batch: jnp.ndarray
):
    image=batch['image_encoder_inputs'].squeeze(0) # (1,1,384,384,3) -> (1,384,384,3))
    prompt=batch['text_encoder_inputs'].squeeze(0)
    actions_in=batch['text_decoder_inputs'].squeeze(0)
    actions_out=batch['text_decoder_targets'].squeeze(0)
    image_out=batch['image_decoder_targets'].squeeze(0)
    logits = state.apply_fn({'params': state.params}, 
                            enable_dropout=False, 
                            image_encoder_inputs=image, 
                            text_encoder_inputs=prompt, 
                            text_decoder_inputs=actions_in, 
                            text_decoder_targets=actions_out, 
                            image_decoder_targets=image_out)
    logits = logits[0] #only use text logits    
    return compute_metrics(logits=logits, labels=actions_out)

#@jax.jit
def test_step(
    state: train_state.TrainState, batch: jnp.ndarray
):
    image=batch['image_encoder_inputs'].squeeze(0) # (1,1,384,384,3) -> (1,384,384,3))
    prompt=batch['text_encoder_inputs'].squeeze(0)
    actions_in=batch['text_decoder_inputs'].squeeze(0)
    actions_out=batch['text_decoder_targets'].squeeze(0)
    image_out=batch['image_decoder_targets'].squeeze(0)
    raw_poses=batch['raw_poses'].squeeze(0)

    logits = state.apply_fn({'params': state.params}, 
                            enable_dropout=False, 
                            image_encoder_inputs=image, 
                            text_encoder_inputs=prompt, 
                            text_decoder_inputs=actions_in, 
                            text_decoder_targets=actions_out, 
                            image_decoder_targets=image_out)
    logits = logits[0] #only use text logits    
    return compute_test_metrics(logits=logits, labels=actions_out, raw_poses=raw_poses)

def cross_entropy_loss(*, logits, labels):
    one_hot_encoded_labels = jax.nn.one_hot(labels, logits.shape[-1])
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

def compute_test_metrics(*, logits, labels, raw_poses):
  tokenizer = T5Tokenizer.from_pretrained(
      "t5-base", model_max_length=256, extra_ids=constants.VOCAB_EXTRA_IDS)
  #hardcoded max and min of dataset to make things easier for now
  MIN = -0.5
  MAX = 0.75
  pq = PoseQuantizer(MIN, MAX, 100)
  loss = cross_entropy_loss(logits=logits, labels=labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

  poses = jnp.argmax(logits, -1)
  positions = np.concatenate((poses[:, 4:7], poses[:, 17:20]), -1)
  exp_positions = np.concatenate((labels[:, 4:7], labels[:, 17:20]), -1)
  position_accuracy = jnp.mean(positions == exp_positions)

  dist_3d = 0
  dist_2d = 0
  quantize_error = 0
  for quantized_actual, actual in zip(labels, raw_poses):
    dec_actual = tokenizer.decode(quantized_actual[:-1]) #doesn't work with jit, unsurprisingly
    dec_vals = re.findall(r'\<.*?\>', dec_actual)
    quantized_dec_vals = pq.decode_array(dec_vals)
    quantize_error += abs(np.linalg.norm(np.array((quantized_dec_vals[0],quantized_dec_vals[1],quantized_dec_vals[2])) - np.array((actual[0], actual[1], actual[2]))))
    quantize_error += abs(np.linalg.norm(np.array((quantized_dec_vals[7],quantized_dec_vals[8],quantized_dec_vals[9])) - np.array((actual[7], actual[8], actual[9]))))
  for pred, actual in zip(jnp.argmax(logits, -1), raw_poses):
    dec_pred = tokenizer.decode(pred[:-1]) #doesn't work with jit, unsurprisingly
    pred_vals = re.findall(r'\<.*?\>', dec_pred)
    if len(pred_vals) != 14:
        dist_3d += 2 * abs(np.linalg.norm(np.array((MIN,MIN,MIN)) - np.array((MAX,MAX,MAX)))) #default to 2 * max distance (since we have 2 poses)
        dist_2d += 2 * abs(math.dist((MIN,MIN), (MAX,MAX))) #default to 2 * max distance (since we have 2 poses)
    else:
        pred_dec_vals = pq.decode_array(pred_vals)
        #ignore rotation for now. Hardcoded position indices
        dist_3d += abs(np.linalg.norm(np.array((pred_dec_vals[0],pred_dec_vals[1],pred_dec_vals[2])) - np.array((actual[0], actual[1], actual[2]))))
        dist_3d += abs(np.linalg.norm(np.array((pred_dec_vals[7],pred_dec_vals[8],pred_dec_vals[9])) - np.array((actual[7], actual[8], actual[9]))))
        dist_2d += abs(math.dist((pred_dec_vals[0],pred_dec_vals[1]), (actual[0], actual[1])))
        dist_2d += abs(math.dist((pred_dec_vals[7],pred_dec_vals[8]), (actual[7], actual[8])))

  num_instances = (2*len(raw_poses))

  avg_dist_3d = dist_3d / num_instances
  avg_dist_2d = dist_2d / num_instances
  quantize_error = quantize_error / num_instances

  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'position_accuracy': position_accuracy,
      'euclidean_distance_3d': avg_dist_3d,
      'euclidean_distance_2d': avg_dist_2d,
      'quantizer_error': quantize_error
  }
  return metrics

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--checkpoint_path",
        default=None
    )
    parser.add_argument(
        "--enable_wandb",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--evaluate",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--model_size",
        choices=["small", "base", "large", "xl"],
        default="small",
    )
    parser.add_argument("--data_path")
    parser.add_argument("--params_path")
    args = parser.parse_args()

    train_path = os.path.join(args.data_path, "train")
    val_path = os.path.join(args.data_path, "val")
    test_path = os.path.join(args.data_path, "test")

    train_data = simple_manipulation_dataset.SimpleManipulationDataset(train_path)
    val_data = simple_manipulation_dataset.SimpleManipulationDataset(val_path)
    test_data = simple_manipulation_dataset.SimpleManipulationDataset(test_path)

    assert min(len(train_data), len(val_data), len(test_data)) > args.batch_size
    train_dataloader = custom_dataloader.CustomDataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dataloader = custom_dataloader.CustomDataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = custom_dataloader.CustomDataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    rng = jax.random.PRNGKey(constants.RANDOM_KEY)

    conf = CONFIGS[args.model_size]
    module = network.Transformer(config=conf, vae_config=VAE_CONFIG)
    model = UnifiedIOModel(module, text_decoder_length=constants.DECODER_LENGTH, image_decoder_length=1)
    params = utils.load_checkpoint(args.params_path)
    state = init_train_state(model, params, learning_rate=args.learning_rate)

    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "checkpoint_path": args.checkpoint_path,
        "enable_wandb": args.enable_wandb
    }

    dataloaders = {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }

    if args.enable_wandb:
        wandb.init()
    train_and_evaluate(dataloaders=dataloaders, state=state, rng=rng, config=config, run_evaluation=args.evaluate)
    if args.enable_wandb:
        wandb.run.save()
    
