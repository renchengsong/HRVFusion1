import os
import numpy as np
import torch
from glob import glob
from torch.utils import data

from torch.utils.data.distributed import DistributedSampler
class NumpyDataset(data.Dataset):
    def __init__(self, gt_path, chrom_path ,npy_paths):
        super().__init__()

        self.chrom_path = chrom_path
        self.gt_path = gt_path
        self.npy_paths = npy_paths
        self.specnames = []
        print(chrom_path,gt_path,npy_paths)
        for path in npy_paths:
            self.specnames += glob(f'{path}/*.spec.npy', recursive=True)


    def __getitem__(self, idx):
        spec_filename = self.specnames[idx]
        spec_path = "/".join(spec_filename.split("/")[:-1])
        chrom_filename = spec_filename.replace(spec_path, self.chrom_path).replace(".mat.spec.npy", ".npy")
        base_filename = os.path.basename(spec_filename).replace(".mat.spec.npy", "")
        gt_filename = f"{self.gt_path}{base_filename.replace('CHROM', 'GT')}.npy"
        chrom_signal = np.load(chrom_filename)
        chrom_signal = chrom_signal[0, :]
        gt_signal = np.load(gt_filename)
        gt_signal = gt_signal[0, :]
        spectrogram = np.load(spec_filename)
        return {
            'gt':  gt_signal,
            'chrom': chrom_signal,
            'spectrogram':spectrogram,
        }
    def __len__(self):
        return len(self.specnames)

class Collator:
    def __init__(self, params):
        self.params = params
    def collate(self,minibatch):
        gt = np.stack([record['gt'] for record in minibatch])
        chrom = np.stack([record['chrom'] for record in minibatch])
        spectrogram = np.stack([record['spectrogram'] for record in minibatch])
        gt = torch.from_numpy(gt).to(dtype=torch.float32)
        chrom = torch.from_numpy(chrom).to(dtype=torch.float32)
        spectrogram = torch.from_numpy(spectrogram).to(dtype=torch.float32)
        return {
            'gt': gt,
            'chrom':chrom ,
            'spectrogram': spectrogram,
        }

def from_path(gt_dir, chrom_dir,data_dirs, params,is_distributed=False):
    dataset = NumpyDataset(gt_dir, chrom_dir,data_dirs)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=Collator(params).collate,
        shuffle=not is_distributed,
        num_workers=os.cpu_count(),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True)
