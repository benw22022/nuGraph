
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
from torch_geometric.nn import radius_graph
import vector

import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os


class GraphDataset(Dataset):
    def __init__(self, file_name, compute_edges=False):
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

        # primaries_map = {
        #     evt: (pdgc, q, px, py, pz, E)
        #     for evt, pdgc, q, px, py, pz, E in zip(
        #         primaries["evtID"],
        #         primaries["PDG"],
        #         primaries["charge"],
        #         primaries["Px"],
        #         primaries["Py"],
        #         primaries["Pz"],
        #         primaries["E"],
        #     )
        # }

        primaries_map = {
            evt: ([], [], [], [], [], []) for evt in primaries["evtID"]
        }
        for evt, pdgc, q, px, py, pz, E in zip(
            primaries["evtID"],
            primaries["PDG"],
            primaries["charge"],
            primaries["Px"],
            primaries["Py"],
            primaries["Pz"],
            primaries["E"],
        ):
            primaries_map[evt][0].append(pdgc)
            primaries_map[evt][1].append(q)
            primaries_map[evt][2].append(px)
            primaries_map[evt][3].append(py)
            primaries_map[evt][4].append(pz)
            primaries_map[evt][5].append(E)

        

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
            
            nu_energy, nu_pdg, process, vx, vy, vz = truth_map[evt]
            nu_energy = np.log10(nu_energy / 1000) # Convert Mev -> GeV and take log

            E_vis = 0
            E_lep = 0
            pT_miss = 0
            p_jet = 0
            pT_jet = 0
            p_lep = 0
            pT_lep = 0
            
            p4_lep = vector.obj(px=0, py=0, pz=0, E=0)
            p4_jet = vector.obj(px=0, py=0, pz=0, E=0)
            p4_jet_vis = vector.obj(px=0, py=0, pz=0, E=0)
            p4_miss = vector.obj(px=0, py=0, pz=0, E=0)

            prim_pdg, prim_charge, prim_px, prim_py, prim_pz, prim_E = primaries_map.get(evt, [])
            for pdg, q, px, py, pz, E in zip(prim_pdg, prim_charge, prim_px, prim_py, prim_pz, prim_E):
                # if abs(pdg) in [12, 14, 16]: # neutrinos
                #     p4_miss += vector.obj(px=px, py=py, pz=pz, E=E)
                if abs(pdg) in [11, 13, 15]: # leptons
                    p4_lep += vector.obj(px=px, py=py, pz=pz, E=E)
                    p4_miss += vector.obj(px=-px, py=-py, pz=-pz, E=E)
                else: # hadrons
                    p4_jet += vector.obj(px=px, py=py, pz=pz, E=E)
                    p4_miss += vector.obj(px=-px, py=-py, pz=-pz, E=E)
                    if q != 0:
                        p4_jet_vis += vector.obj(px=px, py=py, pz=pz, E=E)
            
            E_vis = p4_jet_vis.E + p4_lep.E
            pT_miss = p4_miss.pt
            pT_lep = p4_lep.pt
            pT_jet = p4_jet.pt
            p_jet = p4_jet.p
            p_lep = p4_lep.p
            E_lep = p4_lep.E

            pT_miss = np.log10(pT_miss/1000 + 1e-6)
            pT_lep = np.log10(pT_lep/1000 + 1e-6)
            pT_jet = np.log10(pT_jet/1000 + 1e-6)
            p_jet = np.log10(p_jet/1000 + 1e-6)
            p_lep = np.log10(p_lep/1000 + 1e-6)
            E_lep = np.log10(E_lep/1000 + 1e-6)
            E_vis = np.log10(E_vis/1000 + 1e-6)


            if "NC" in process:
                interaction = 3
            else:
                interaction = {12: 0, 14: 1, 16: 2}.get(abs(nu_pdg))

            
            data = Data( 
                x=torch.from_numpy(x),
                y_class=torch.tensor(interaction),
                E_nu=torch.tensor(nu_energy, dtype=torch.float32),
                E_vis=torch.tensor(E_vis, dtype=torch.float32),
                pT_miss=torch.tensor(pT_miss, dtype=torch.float32),
                pT_lep=torch.tensor(pT_lep, dtype=torch.float32),
                pT_jet=torch.tensor(pT_jet, dtype=torch.float32),
                p_jet=torch.tensor(p_jet, dtype=torch.float32),
                p_lep=torch.tensor(p_lep, dtype=torch.float32),
                E_lep=torch.tensor(E_lep, dtype=torch.float32),
                )
            
            if compute_edges:
                data.to(device="cuda")
                data.edge_index = radius_graph(data.x, r=2.5, loop=False, num_workers=31)
                data.to(device="cpu")
        
            self.data.append(data)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]
    

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
