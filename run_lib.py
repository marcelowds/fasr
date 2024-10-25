# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import datasets_train
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from datetime import datetime
from aux import manipule
#from torchinfo import summary
FLAGS = flags.FLAGS

import net
#from face_alignment import align
import PIL
from PIL import Image
import torchvision
import torchvision.transforms as T
import random



def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  #sample_dir = os.path.join(workdir, "samples")
  #tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  f = open(workdir+"/config_log.txt", "a")
  now = datetime.now()
  t_string = now.strftime("%d/%m/%Y %H:%M:%S\n")
  f.write(t_string)
  f.write(str(config))
  f.close()

  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets_train.get_dataset(config,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets_train.get_data_scaler(config)
  inverse_scaler = datasets_train.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)


  num_train_steps = config.training.n_iters
  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  
  ## marcelo
  adaface_models = {
    'ir_18':config.data.adaface_model_path
    }

  def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model
  ##
  model = load_pretrained_model('ir_18')
  model.to(config.device)
  batch_size = config.training.batch_size
  tensor = torch.randn(batch_size,3,112,112)
  feat = torch.randn(batch_size,512)
  resize_image = T.Resize((112,112))
  ## fim marcelo
  cont = 0
  for step in range(initial_step, num_train_steps + 1):

    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)    
    

    for i in range(batch_size):
      ## input HR image
      tensor[i,:,:,:] = (torch.flip((resize_image(batch[i,:3,:,:])),dims=(0,))-0.5)*2

    feat, _ = model(tensor.to(config.device))    

    loss = train_step_fn(state, batch, feat.to(config.device))
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Report the loss on an evaluation dataset periodically    
    if step % config.training.eval_freq == 0:
      eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
      eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      
      for i in range(batch_size):
        tensor[i,:,:,:] = (torch.flip((resize_image(batch[i,:3,:,:])),dims=(0,))-0.5)*2
      feat, _ = model(tensor.to(config.device))      
      ##
      eval_loss = eval_step_fn(state, eval_batch, feat.to(config.device))
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)    
      
    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)    
      
    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)
      #break

      
     

def evaluate(config,
             workdir,
             eval_folder="sr_results"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "sr_results".
  """
  adaface_models = {
    'ir_18':config.data.adaface_model_path
    }

  def load_pretrained_model(architecture='ir_18'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model
  model = load_pretrained_model('ir_18')
  batch_size = config.eval.batch_size
  tensor = torch.randn(batch_size,3,112,112)
  resize_image = T.Resize((112,112))
  ##
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline
  samples_ds, __, _ = datasets.get_dataset(config,uniform_dequantization=config.data.uniform_dequantization,evaluation=True)
  sample_iter = samples_ds


  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
  ckpt_filename = os.path.join(checkpoint_dir, "checkpoint.pth")

  score_model = mutils.create_model(config)
  
  optimizer = losses.get_optimizer(config, score_model.parameters())

  ema = ExponentialMovingAverage(score_model.parameters(),decay=config.model.ema_rate)
  state = dict(step=0, optimizer=optimizer, model=score_model, ema=ema)
  state = restore_checkpoint(ckpt_filename, state, config.device)
  ema.copy_to(score_model.parameters())


  # Setup SDEs
  if config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  ## upload features celeba
  #path_features = '/home/msantos/sr-sde/experimentos_celeba/media_features_500ids'
  #path_features = config.data.features_path
 
  total_path_features = config.data.features_path #os.path.join(path_features,'celeba_500ids_feature_ir18_media.t')
  with open(total_path_features,'rb') as f:
    features_media = torch.load(f)


  # num_iter = number of images / batch size
  num_iter = 5 
  batch_size = config.eval.batch_size

  # Build the sampling function when sampling is enabled
  sampling_shape = (batch_size,config.data.num_channels,config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
  
  sample_dir = eval_dir
  feat = torch.randn(batch_size,512)
  for i, batch in enumerate(sample_iter):
    print(f"Starting batch {i+1} de {num_iter}.")
    batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
    batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)

    for j in range(batch_size):
      ## mean feat
      feat[j,:] = features_media[i*batch_size+j]
    #feat, _ = model(tensor.to(config.device))    
    #feat = ((feat-torch.min(feat))/(torch.max(feat)-torch.min(feat)))
    
    # High resolution image
    original_hr = manipule.HR(batch)

    # Low resolution image
    input_lr = manipule.LR(batch)

    # Compute Super-Resolution image
    output_sr, n = sampling_fn(score_model,input_lr, feat)
    
    # Save images
    manipule.save_img(inverse_scaler(original_hr),sample_dir,'-hr.png',batch_size,i)
    manipule.save_img(inverse_scaler(input_lr),sample_dir,'-lr.png',batch_size,i)
    manipule.save_img(output_sr,sample_dir,'-sr.png',batch_size,i)

    print(f"Finished processing on images {i*batch_size} to {(i+1)*batch_size-1}")
    if i == num_iter-1:
      break
  print(f"Total photos processed: {batch_size*num_iter}")

