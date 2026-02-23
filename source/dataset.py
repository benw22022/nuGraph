
import logging
import random
import torch
import uproot
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import numpy as np
import awkward as ak
from torch_geometric.loader import DataLoader as GeoDataLoader


import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os


class GraphDataset(Dataset):
    def __init__(self, file_name):
        super().__init__()

        self.file_event_counts = []
        self.offsets = []

        total = 0
        
        root_file = uproot.open(file_name)
        
        truth = root_file["event"].arrays(library="np")
        primaries = root_file["primaries"].arrays(library="np")
        geom = root_file["geometry"].arrays(library="np")
        hits = root_file["Hits/pixelHits"]
        scints = root_file["Hits/scintHits"]

        event_id   = hits["event_id"].array(library="np")
        hit_layer  = hits["hit_layerID"].array(library="np")
        hit_col    = hits["hit_colID"].array(library="np")
        hit_row    = hits["hit_rowID"].array(library="np")

        self.data = []
        self.targets = []

        truth_map = {
            evt: (E, pdgc, proc, x, y, z)
            for evt, E, pdgc, proc, x, y, z in zip(
                truth["evtID"],
                truth["initE"],
                truth["initPDG"],
                truth["processName"],
                truth["initX"],
                truth["initY"],
                truth["initZ"],
            )
        }


        unique_evt, evt_index = np.unique(event_id, return_inverse=True)

        for i, evt in tqdm(enumerate(unique_evt), total=len(unique_evt)):
            idx = np.where(evt_index == i)[0]
            if evt not in truth_map:
                continue
            
            # Skip events with < 100 hits
            if len(np.ravel(hit_layer[idx][0])) < 100: continue

            row = hit_row[idx][0]
            col = hit_col[idx][0]
            layer = hit_layer[idx][0]

            col = col / (np.amax(col) + 1e-6)
            row = row / (np.amax(row) + 1e-6)
            layer = layer / (np.amax(layer) + 1e-6)

            x = np.stack([row, col, layer], axis=1).astype(np.float32)
            
            energy, pdg, process, vx, vy, vz = truth_map[evt]
            energy = np.log1p(energy / 1000) # Convert Mev -> GeV and take log

            if "NC" in process:
                interaction = 3
            else:
                interaction = {12: 0, 14: 1, 16: 2}.get(abs(pdg))

            
            data = Data( 
                x=torch.from_numpy(x),
                y_class=torch.tensor(interaction),
                y_energy=torch.tensor(energy, dtype=torch.float32),
                )

            self.data.append(data)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]
    

# class CombinedDataset(Dataset):
#     def __init__(self, pt_files):
#         self.datasets = [torch.load(f, map_location="cpu", weights_only=False, mmap=True) for f in pt_files]
#         self.cum_lengths = []
#         total = 0
#         for ds in self.datasets:
#             total += len(ds)
#             self.cum_lengths.append(total)

#     def __len__(self):
#         return self.cum_lengths[-1]

#     def __getitem__(self, idx):
#         for ds_idx, end in enumerate(self.cum_lengths):
#             if idx < end:
#                 start = 0 if ds_idx == 0 else self.cum_lengths[ds_idx - 1]
#                 return self.datasets[ds_idx][idx - start]
#         raise IndexError

class CombinedDataset(Dataset):
    def __init__(self, pt_files):
        self.datasets = [
            torch.load(f, map_location="cpu", weights_only=False, mmap=True)
            for f in pt_files
        ]

        self.index_map = []
        for ds_idx, ds in enumerate(self.datasets):
            for i in range(len(ds)):
                self.index_map.append((ds_idx, i))

        # shuffle events globally
        random.shuffle(self.index_map)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.index_map[idx]
        return self.datasets[ds_idx][sample_idx]


class GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        pt_files,
        batch_size=2,
        num_workers=4,
        train_test_val_split=(0.7, 0.2, 0.1),
    ):
        super().__init__()
        self.pt_files = pt_files
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_test_val_split[0]
        self.test_split = train_test_val_split[1]
        self.val_split = train_test_val_split[2]

    def setup(self, stage=None):
        files = list(self.pt_files)
        random.Random(42).shuffle(files)

        n = len(files)
        n_test = int(self.test_split * n)
        n_val = int(self.val_split * n)

        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]

        self.train_ds = CombinedDataset(train_files)
        self.val_ds = CombinedDataset(val_files)
        self.test_ds = CombinedDataset(test_files)

    def train_dataloader(self):
        return GeoDataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return GeoDataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return GeoDataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
