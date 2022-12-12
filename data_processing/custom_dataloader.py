import numpy as np
from torch.utils import data
from flax.training.common_utils import shard

def collate(batch):
    image_encoder_inputs = []
    text_encoder_inputs = []
    text_decoder_inputs = []
    text_decoder_targets = []
    image_decoder_targets = []
    raw_poses = []

    for data_point in batch:
        image_encoder_inputs.append(data_point['image_encoder_input'])
        text_encoder_inputs.append(data_point['text_encoder_input'])
        text_decoder_inputs.append(data_point['text_decoder_input'])
        text_decoder_targets.append(data_point['text_decoder_target'])
        image_decoder_targets.append(data_point['image_decoder_target'])
        raw_poses.append(data_point['raw_poses'])

    data_dict = {
    'image_encoder_inputs': np.array(image_encoder_inputs),
    'text_encoder_inputs': np.array(text_encoder_inputs),
    'text_decoder_inputs': np.array(text_decoder_inputs),
    'text_decoder_targets': np.array(text_decoder_targets),
    'image_decoder_targets': np.array(image_decoder_targets),
    'raw_poses': np.array(raw_poses)
    }

    final_batch = {k: np.array(v) for k, v in data_dict.items()}
    final_batch = shard(final_batch)
    return final_batch

class CustomDataLoader(data.DataLoader):
  def __init__(self, dataset, batch_size=1,
                shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0,
                pin_memory=False, drop_last=False,
                timeout=0, worker_init_fn=None):
    super(self.__class__, self).__init__(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn)