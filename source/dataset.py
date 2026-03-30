
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
import particle


def norm_data(data, mean, std):
    return (data - mean) / std

def unnorm_data(data, mean ,std):
    return (std * data) + mean


class GraphDataset(Dataset):
    def __init__(self, file_name, compute_edges=False, stats=None):
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
        tau_data = root_file["tau"].arrays(library="pd")
        charm_data = root_file["charm"].arrays(library="pd")

        event_id   = hits["event_id"].array(library="np")
        hit_layer  = hits["hit_layerID"].array(library="np")
        hit_col    = hits["hit_colID"].array(library="np")
        hit_row    = hits["hit_rowID"].array(library="np")

        self.data = []
        self.targets = []

        truth_map = {
            evt: (E, pdgc, proc, x, y, z, px, py, pz)
            for evt, E, pdgc, proc, x, y, z, px, py, pz in zip(
                truth["evtID"],
                truth["initE"],
                truth["initPDG"],
                truth["processName"],
                truth["initX"],
                truth["initY"],
                truth["initZ"],
                truth["initPx"],
                truth["initPy"],
                truth["initPz"],
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
            
            E_nu, nu_pdg, process, vx, vy, vz, nu_px, nu_py, nu_pz = truth_map[evt]
            # E_nu = np.log10(E_nu) # Convert Mev -> GeV and take log

            E_vis = 0
            E_lep = 0
            pT_miss = 0
            p_jet = 0
            pT_jet = 0
            p_lep = 0
            pT_lep = 0
            
            p4_nu = vector.obj(px=nu_px, py=nu_py, pz=nu_pz, E=E_nu)
            p4_lep = vector.obj(px=0, py=0, pz=0, E=0)
            p4_jet = vector.obj(px=0, py=0, pz=0, E=0)
            p4_jet_vis = vector.obj(px=0, py=0, pz=0, E=0)
            p4_miss = vector.obj(px=0, py=0, pz=0, E=0)

            tau_decay_mode = 0 # 0 = no tau, 1 = electron, 2 = muon, 3 = hadronic

            prim_pdg, prim_charge, prim_px, prim_py, prim_pz, prim_E = primaries_map.get(evt, [])
            for pdg, q, px, py, pz, E in zip(prim_pdg, prim_charge, prim_px, prim_py, prim_pz, prim_E):
                if abs(pdg) in [12, 14, 16]: # neutrinos
                    p4_miss += vector.obj(px=px, py=py, pz=pz, E=E)
                if abs(pdg) in [11, 13, 15]: # leptons
                    p4_lep += vector.obj(px=px, py=py, pz=pz, E=E)
                    # p4_miss += vector.obj(px=-px, py=-py, pz=-pz, E=E)
                else: # hadrons - the jet
                    p4_jet += vector.obj(px=px, py=py, pz=pz, E=E)
                    # p4_miss += vector.obj(px=-px, py=-py, pz=-pz, E=E)
                    if q != 0:
                        p4_jet_vis += vector.obj(px=px, py=py, pz=pz, E=E)

            # Tau data
            n_prongs = 0
            n_neutrals = 0
            n_neutral_pions = 0
            if len(tau_data["evtID"]) > 0:
                tau_decay_mode = 3
                tau_decay_prods = tau_data[tau_data["evtID"] == evt]
                for pdg, px, py, pz in zip(
                    tau_decay_prods["PDG"],
                    tau_decay_prods["Px"],
                    tau_decay_prods["Py"],
                    tau_decay_prods["Pz"],
                ):  
                    try:
                        p = particle.Particle.from_pdgid(pdg)
                        q = p.charge
                    except particle.ParticleNotFound:
                        q = 0
                        continue

                    if abs(pdg) in [12, 14, 16]: # neutrinos
                        p4_miss += vector.obj(px=px, py=py, pz=pz, mass=0)
                    
                    if abs(pdg) == 12:
                        tau_decay_mode = 1
                    elif abs(pdg) == 14:
                        tau_decay_mode = 2
                    elif q != 0 and abs(pdg) > 18:
                        n_prongs += 1
                    elif q == 0 and abs(pdg) > 18:
                        n_neutrals += 1
                    elif abs(pdg) in [111]: # neutral pions
                        n_neutral_pions += 1


            # Charm data
            if len(charm_data["evtID"]) > 0:
                charm_decay_prods = charm_data[charm_data["evtID"] == evt]
                for pdg, px, py, pz in zip(
                    charm_decay_prods["PDG"],
                    charm_decay_prods["Px"],
                    charm_decay_prods["Py"],
                    charm_decay_prods["Pz"],
                ):
                    if abs(pdg) in [12, 14, 16]: # neutrinos
                        p4_miss += vector.obj(px=px, py=py, pz=pz, mass=0)
            
            E_vis = p4_jet_vis.E + p4_lep.E
            pT_miss = p4_miss.pt
            pT_lep = p4_lep.pt
            pT_jet = p4_jet.pt
            p_jet = p4_jet.p
            p_lep = p4_lep.p
            E_lep = p4_lep.E

            # px_nu = p4_nu.px / 1000
            # py_nu = p4_nu.py / 1000
            # pz_nu = p4_nu.pz / 1000
            # E_nu = p4_nu.E / 1000

            # px_lep = p4_lep.px / 1000
            # py_lep = p4_lep.py / 1000
            # pz_lep = p4_lep.pz / 1000
            # E_lep = p4_lep.E / 1000

            # px_jet = p4_jet.px / 1000
            # py_jet = p4_jet.py / 1000
            # pz_jet = p4_jet.pz / 1000
            # E_jet = p4_jet.E / 1000

            # px_miss = p4_miss.px / 1000
            # py_miss = p4_miss.py / 1000
            # pz_miss = p4_miss.pz / 1000
            # E_miss = p4_miss.E / 1000


            
            # print(p4_lep, pT_lep/1000, p_lep/1000, E_lep/1000)

            # pT_miss = np.log10(pT_miss/1000 + 1e-6)
            # pT_lep = np.log10(pT_lep/1000 + 1e-6)
            # pT_jet = np.log10(pT_jet/1000 + 1e-6)
            # p_jet = np.log10(p_jet/1000 + 1e-6)
            # p_lep = np.log10(p_lep/1000 + 1e-6)
            # E_lep = np.log10(E_lep/1000 + 1e-6)
            # E_vis = np.log10(E_vis/1000 + 1e-6)

            # Normalise
            pT_miss /= 1000
            pT_lep  /= 1000
            pT_jet  /= 1000
            p_lep   /= 1000
            p_jet   /= 1000
            E_vis   /= 1000
            E_nu    /= 1000
            E_lep   /= 1000

            if stats is not None:
                pT_miss = norm_data(pT_miss, stats.pT_miss.mean, stats.pT_miss.std)
                pT_lep = norm_data(pT_lep, stats.pT_lep.mean, stats.pT_lep.std)
                pT_jet = norm_data(pT_jet, stats.pT_jet.mean, stats.pT_jet.std)
                p_lep = norm_data(p_lep, stats.p_lep.mean, stats.p_lep.std)
                p_jet = norm_data(p_jet, stats.p_jet.mean, stats.p_jet.std)
                E_lep = norm_data(E_lep, stats.E_lep.mean, stats.E_lep.std)
                E_vis = norm_data(E_vis, stats.E_vis.mean, stats.E_vis.std)
                E_nu = norm_data(E_nu, stats.E_nu.mean, stats.E_nu.std)
                
                
            # Work out the interaction type for classification - CC nu_mu, CC nu_e, CC nu_tau or NC
            if "NC" in process:
                interaction = 3
                pT_lep = np.nan
                p_lep = np.nan
                E_lep = np.nan
            else:
                interaction = {12: 0, 14: 1, 16: 2}.get(abs(nu_pdg))
            
            data = Data( 
                x=torch.from_numpy(x),
                interaction=torch.tensor(interaction),
                pT_nu = torch.tensor(p4_nu.pt / 1000, dtype=torch.float32),
                eta_nu = torch.tensor(p4_nu.eta, dtype=torch.float32),
                phi_nu = torch.tensor(p4_nu.phi, dtype=torch.float32),
                E_nu=torch.tensor(p4_nu.E / 1000, dtype=torch.float32),

                pT_lep=torch.tensor(p4_lep.pt / 1000, dtype=torch.float32),
                eta_lep=torch.tensor(p4_lep.eta, dtype=torch.float32),
                phi_lep=torch.tensor(p4_lep.phi, dtype=torch.float32),
                E_lep=torch.tensor(p4_lep.E / 1000, dtype=torch.float32),

                pT_jet=torch.tensor(p4_jet.pt / 1000, dtype=torch.float32),
                eta_jet=torch.tensor(p4_jet.eta, dtype=torch.float32),
                phi_jet=torch.tensor(p4_jet.phi, dtype=torch.float32),
                E_jet=torch.tensor(p4_jet.E / 1000, dtype=torch.float32),

                pT_miss=torch.tensor(pT_miss / 1000, dtype=torch.float32),
                eta_miss=torch.tensor(p4_miss.eta, dtype=torch.float32),
                phi_miss=torch.tensor(p4_miss.phi, dtype=torch.float32),
                E_miss=torch.tensor(p4_miss.E / 1000, dtype=torch.float32),

                # E_vis=torch.tensor(E_vis, dtype=torch.float32),
                # pT_miss=torch.tensor(pT_miss, dtype=torch.float32),
                # pT_lep=torch.tensor(pT_lep, dtype=torch.float32),
                # pT_jet=torch.tensor(pT_jet, dtype=torch.float32),
                # p_jet=torch.tensor(p_jet, dtype=torch.float32),
                # p_lep=torch.tensor(p_lep, dtype=torch.float32),
                # E_lep=torch.tensor(E_lep, dtype=torch.float32),
                tau_decay_mode=torch.tensor(tau_decay_mode, dtype=torch.long),
                tau_prongs=torch.tensor(n_prongs, dtype=torch.long),
                tau_neutrals=torch.tensor(n_neutrals, dtype=torch.long),
                n_neutral_pions=torch.tensor(n_neutral_pions, dtype=torch.long),
                )
            
            # Compute edges if requested (requires a GPU)
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
