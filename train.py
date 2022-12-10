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
import constants
from datasets import Dataset
import time
import os

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
                checkpoint_prefix = "checkpoint_{}_step_".format(time.strftime("%Y%m%d-%H%M%S"))
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
            metrics = eval_step(state, batch)
            test_batch_metrics.append(metrics)
        test_batch_metrics = accumulate_metrics(test_batch_metrics)

        # Log Metrics to Weights & Biases
        if enable_wandb:
            wandb.log({
                "Test Accuracy": test_batch_metrics['accuracy']
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
    
