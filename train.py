# Most Jax functions were slightly / completely rewritten following the guide here: https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-FLAX--VmlldzoyMzA4ODEy

import os
import argparse
import wandb

import numpy as np

from tqdm.auto import tqdm

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax.training.common_utils import shard

from uio import utils
from uio import network
from uio.configs import CONFIGS, VAE_CONFIG
from uio.model import UnifiedIOModel

import process_data
from datasets import Dataset


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

def train_and_evaluate(train_dataset, eval_dataset, test_dataset, state, rng, epochs, bs):   
    step_per_epoch = len(train_dataset)
    total_steps = step_per_epoch * epochs

    for epoch in tqdm(range(1, epochs + 1)):
        rng, input_rng = jax.random.split(rng)

        # ============== Training ============== #
        train_batch_metrics = []
        for step, batch in enumerate(
            tqdm(
                train_data_collator(input_rng, train_dataset, bs),
                total=step_per_epoch,
                desc="Training...",
                position=1,
            )
        ):
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

    image=batch['image_encoder_inputs'][0]
    prompt=batch['text_encoder_inputs'][0]
    actions_in=batch['text_decoder_inputs'][0]
    actions_out=batch['text_decoder_targets'][0]
    image_out=batch['image_decoder_targets'][0]

    def loss_fn(params):
        logits = state.apply_fn({'params': params}, image_encoder_inputs=image, text_encoder_inputs=prompt, text_decoder_inputs=actions_in, text_decoder_targets=actions_out, image_decoder_targets=image_out)
        logits = logits[0] #only use text logits
        loss = cross_entropy_loss(logits=logits, labels=actions_out)
        return loss, logits


    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=actions_out)
    return state, metrics

#@jax.jit
#def eval_step(state, batch):
#    image, label = batch
#    logits = state.apply_fn({'params': state.params}, image)
#    return compute_metrics(logits=logits, labels=label)

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
    parser.add_argument("path")
    parser.add_argument("params_path")
    args = parser.parse_args()

    train_path = os.path.join(args.path, "train")
    val_path = os.path.join(args.path, "val")
    test_path = os.path.join(args.path, "test")

    train = os.listdir(train_path)
    train = train[0:10]
    val = os.listdir(val_path)
    val = val[0:10]
    test = os.listdir(test_path)
    test = test[0:10]

    print("Train Instances: " + str(len(train)))
    print("Val Instances: " + str(len(val)))
    print("Test Instances: " + str(len(test)))

    train_data = process_data.getDataset(train_path, train)

    rng = jax.random.PRNGKey(42)#hard-coded for now

    conf = CONFIGS["small"]
    module = network.Transformer(config=conf, vae_config=VAE_CONFIG)
    model = UnifiedIOModel(module, text_decoder_length=32, image_decoder_length=1)
    params = utils.load_checkpoint(args.params_path)
    state = init_train_state(model, params, learning_rate=0.01)

    wandb.init()
    train_and_evaluate(train_dataset=train_data, eval_dataset=None, test_dataset=None, state=state, rng=rng, epochs=1, bs=1)
    wandb.run.save()
    