import numpy as np
import os
import torch
import random
from argparse import ArgumentParser
import pdb
import scipy.io as sio
import torch.utils.tensorboard
from params import AttrDict, params as base_params
from model import DiffuSE
from os import path
from glob import glob
from tqdm import tqdm

random.seed(23)
models = {}

def load_model(model_dir=None, args=None, params=None, device=torch.device('cuda')):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    
    model = DiffuSE(args, AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model
  model = models[model_dir]
  model.params.override(params)
      
  return model
      

def inference_schedule(model, fast_sampling=False):
    training_noise_schedule = np.array(model.params.noise_schedule)
    inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

    talpha = 1 - training_noise_schedule
    talpha_cum = np.cumprod(talpha)

    beta = inference_noise_schedule
    alpha = 1 - beta
    alpha_cum = np.cumprod(alpha)
    sigmas = [0 for i in alpha]
    for n in range(len(alpha) - 1, -1, -1): 
      sigmas[n] = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])

    T = []
    for s in range(len(inference_noise_schedule)):
      for t in range(len(training_noise_schedule) - 1):
        if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
          twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
          T.append(t + twiddle)
          break
    T = np.array(T, dtype=np.float32)

    m = [0 for i in alpha] 
    gamma = [0 for i in alpha] 
    delta = [0 for i in alpha]  
    d_x = [0 for i in alpha]  
    d_y = [0 for i in alpha]  
    delta_cond = [0 for i in alpha]  
    delta_bar = [0 for i in alpha] 
    c1 = [0 for i in alpha] 
    c2 = [0 for i in alpha] 
    c3 = [0 for i in alpha] 
    oc1 = [0 for i in alpha] 
    oc3 = [0 for i in alpha] 
    
    for n in range(len(alpha)):
      m[n] = min(((1- alpha_cum[n])/(alpha_cum[n]**0.5)),1)**0.5
    m[-1] = 1    

    for n in range(len(alpha)):
      delta[n] = max(1-(1+m[n]**2)*alpha_cum[n],0)
      gamma[n] = sigmas[n]

    for n in range(len(alpha)):
      if n >0: 
        d_x[n] = (1-m[n])/(1-m[n-1]) * (alpha[n]**0.5)
        d_y[n] = (m[n]-(1-m[n])/(1-m[n-1])*m[n-1])*(alpha_cum[n]**0.5)
        delta_cond[n] = delta[n] - (((1-m[n])/(1-m[n-1])))**2 * alpha[n] * delta[n-1]
        delta_bar[n] = (delta_cond[n])* delta[n-1]/ delta[n]
      else:
        d_x[n] = (1-m[n])* (alpha[n]**0.5)
        d_y[n]= (m[n])*(alpha_cum[n]**0.5)
        delta_cond[n] = 0
        delta_bar[n] = 0

    for n in range(len(alpha)):
      oc1[n] = 1 / alpha[n]**0.5
      oc3[n] = oc1[n] * beta[n] / (1 - alpha_cum[n])**0.5
      if n >0:
        c1[n] = (1-m[n])/(1-m[n-1])*(delta[n-1]/delta[n])*alpha[n]**0.5 + (1-m[n-1])*(delta_cond[n]/delta[n])/alpha[n]**0.5
        c2[n] = (m[n-1] * delta[n] - (m[n] *(1-m[n]))/(1-m[n-1])*alpha[n]*delta[n-1])*(alpha_cum[n-1]**0.5/delta[n])
        c3[n] = (1-m[n-1])*(delta_cond[n]/delta[n])*(1-alpha_cum[n])**0.5/(alpha[n])**0.5
      else:
        c1[n] = 1 / alpha[n]**0.5
        c3[n] = c1[n] * beta[n] / (1 - alpha_cum[n])**0.5
    return alpha, beta, alpha_cum,sigmas, T, c1, c2, c3, delta, delta_bar
      

def predict(spectrogram, model, chrom, alpha, beta, alpha_cum, sigmas, T, c1, c2, c3, delta, delta_bar, device=torch.device('cuda')):
  with torch.no_grad():
    # Expand rank 2 tensors by adding a batch dimension.
    spectrogram = spectrogram.to(dtype=torch.float32)
    if len(spectrogram.shape) == 2:
      spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)
    
    
    bvp = torch.randn(spectrogram.shape[0], spectrogram.shape[-1], device=device)
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)
    noisy_bvp = torch.zeros(spectrogram.shape[0], spectrogram.shape[-1], device=device)
    noisy_bvp[:, :chrom.shape[0]] = torch.from_numpy(chrom).to(device)
    bvp = noisy_bvp
    gamma = [0.2]
    for n in range(len(alpha) - 1, -1, -1):
      if n > 0:
        predicted_noise =  model(bvp, spectrogram, torch.tensor([T[n]], device=bvp.device)).squeeze(1)
        bvp = c1[n] * bvp + c2[n] * noisy_bvp - c3[n] * predicted_noise
        noise = torch.randn_like(bvp)
        newsigma= delta_bar[n]**0.5 
        bvp += newsigma * noise
      else:
        predicted_noise =  model(bvp, spectrogram, torch.tensor([T[n]], device=bvp.device)).squeeze(1)
        bvp = c1[n] * bvp - c3[n] * predicted_noise
        bvp = (1-gamma[n])*bvp+gamma[n]*noisy_bvp
      bvp = torch.clamp(bvp, -1.0, 1.0)
  return bvp


def main(args):
  specnames = []
  print("spectrum:",args.spectrogram_path)
  print("chrom:",args.chrom_path)
  for path in args.spectrogram_path:
    specnames += glob(f'{path}/*.spec.npy', recursive=True)
  
  model = load_model(model_dir=args.model_dir ,args=args)
  alpha, beta, alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar = inference_schedule(model, fast_sampling=args.fast)

  output_path = args.output
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  for spec in tqdm(specnames):
    spectrogram = torch.from_numpy(np.load(spec))
    chrom = np.load(os.path.join(args.chrom_path,spec.split("/")[-1].replace(".mat.spec","")))
    chrom = chrom.squeeze(0) #[300]
    wlen = chrom.shape[0]
    enhance_bvp = predict(spectrogram, model, chrom, alpha, beta, alpha_cum, sigmas, T,c1, c2, c3, delta, delta_bar)
    enhance_bvp = enhance_bvp[:,:wlen]
    output_name = os.path.join(output_path,spec.split("/")[-1].replace(".mat.spec.npy",""))
    np.save(output_name, enhance_bvp.cpu())


if __name__ == '__main__':
    parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
    parser.add_argument('--model_dir', default='model_dir/weights.pt',
                        help='directory containing a trained model (or full path to weights.pt file)')
    parser.add_argument('--spectrogram_path', nargs='+', default=['data/test/spectrogram/'],
                        help='space separated list of directories from spectrogram file generated by diffwave.preprocess')
    parser.add_argument('--chrom_path', default='data/test/noisy/',
                        help='input noisy wav directory')
    parser.add_argument('--output', '-o', default='inference_results/output1/',
                        help='output path name')
    parser.add_argument('--fast', dest='fast', action='store_true',
                        help='fast sampling procedure')
    parser.add_argument('--full', dest='fast', action='store_false',
                        help='fast sampling procedure')
    parser.add_argument('--se', dest='se', action='store_true')
    parser.add_argument('--se_pre', dest='se', action='store_false')
    # parser.add_argument('--voicebank', dest='voicebank', action='store_true')
    # parser.set_defaults(se=True)
    parser.set_defaults(fast=True)
    # parser.set_defaults(voicebank=False)
    main(parser.parse_args())

