import jax
import jax.numpy as jnp

import optax

from flax.training import train_state

import numpy as np
import os

import argparse
from tqdm.auto import tqdm
import wandb

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


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    data = os.listdir(args.path)
    #50065 samples
    print(len(data))