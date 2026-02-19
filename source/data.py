import torch
from torch_geometric.data import Dataset, Data
import uproot
import numpy as np
import awkward as ak
from tqdm import tqdm
import logging

# class NeutrinoRootDataset(Dataset):
#     def __init__(self, root_files):
#         super().__init__()

#         self.root_files = root_files
#         self.file_event_counts = []
#         self.offsets = []

#         total = 0
#         for f in root_files:
#             file = uproot.open(f)
#             n = file["event"].num_entries
#             self.file_event_counts.append(n)
#             self.offsets.append(total)
#             total += n

#         self.total_events = total

#         # cache
#         self.current_file_index = None
#         self.hits = None
#         self.truth = None

#     def len(self):
#         return self.total_events

#     def _load_file(self, file_idx):
#         f = uproot.open(self.root_files[file_idx])

#         self.hits = f["Hits/pixelHits"].arrays(
#             ["hit_layerID", "hit_rowID", "hit_colID"],
#             library="ak",
#         )

#         self.truth = f["event"].arrays(
#             ["initE", "initPDG", "processName"],
#             library="ak",
#         )

#         self.current_file_index = file_idx

#     def _locate(self, idx):
#         for i in range(len(self.offsets) - 1, -1, -1):
#             if idx >= self.offsets[i]:
#                 return i, idx - self.offsets[i]

#     def get(self, idx):
#         file_idx, local_idx = self._locate(idx)

#         if file_idx != self.current_file_index:
#             self._load_file(file_idx)

#         row = ak.to_numpy(self.hits["hit_rowID"][local_idx])
#         col = ak.to_numpy(self.hits["hit_colID"][local_idx])
#         layer = ak.to_numpy(self.hits["hit_layerID"][local_idx])

#         col = col / (np.amax(col) + 1e-6)
#         row = row / (np.amax(row) + 1e-6)
#         layer = layer / (np.amax(layer) + 1e-6)

#         x = np.stack([row, col, layer], axis=1).astype(np.float32)

#         energy = float(self.truth["initE"][local_idx])
#         pdg = int(self.truth["initPDG"][local_idx])
#         process = str(self.truth["processName"][local_idx])

#         energy = np.log1p(energy)

#         if "NC" in process:
#             interaction = 3
#         else:
#             interaction = {12: 0, 14: 1, 16: 2}.get(abs(pdg), 3)

#         return Data(
#             x=torch.from_numpy(x),
#             y_class=torch.tensor(interaction),
#             y_energy=torch.tensor(energy, dtype=torch.float32),
#         )

import uproot
import awkward as ak
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def preprocess_root_to_parquet(root_files, output_file, output_dir="data/parq"):
    output_dir = Path(output_dir)
    output_file = Path(output_file)
    output_dir.mkdir(exist_ok=True)

    if not isinstance(root_files, list): root_files = [root_files]

    all_events = []
    nskipped_events = 0

    for file_idx, f in enumerate(root_files):
        print(f"Processing file {file_idx+1}/{len(root_files)}: {f}")
        file = uproot.open(f)

        hits = file["Hits/pixelHits"].arrays(
            ["hit_layerID", "hit_rowID", "hit_colID"],
            library="ak",
        )
        truth = file["event"].arrays(
            ["initE", "initPDG", "processName"],
            library="ak",
        )

        num_events = len(truth["initE"])

        for i in tqdm(range(num_events)):
            row = ak.to_numpy(hits["hit_rowID"][i])
            col = ak.to_numpy(hits["hit_colID"][i])
            layer = ak.to_numpy(hits["hit_layerID"][i])

            if len(row) < 100:
                nskipped_events += 1
                continue  # skip small events

            # normalize
            row = row / (np.amax(row) + 1e-6)
            col = col / (np.amax(col) + 1e-6)
            layer = layer / (np.amax(layer) + 1e-6)

            x = np.stack([row, col, layer], axis=1).astype(np.float32)

            # log energy
            energy = float(truth["initE"][i])
            energy = np.log1p(energy)

            pdg = int(truth["initPDG"][i])
            process = str(truth["processName"][i])
            interaction = 3 if "NC" in process else {12: 0, 14: 1, 16: 2}.get(abs(pdg), 3)

            # store as dict (Parquet doesn't support ragged arrays directly)
            all_events.append({
                "x": x.tolist(),  # save as nested list
                "y_class": interaction,
                "y_energy": energy
            })

    # Convert to DataFrame
    df = pd.DataFrame(all_events)
    output_file = output_dir / output_file
    df.to_parquet(output_file, engine="pyarrow", index=False)
    logging.info(f"Saved {len(df)} events to {output_file}. Skipped {nskipped_events} events with < 100 hits")


import torch
from torch_geometric.data import Dataset, Data
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

class NeutrinoParquetDataset(Dataset):
    def __init__(self, parquet_files):
        super().__init__()
        self.parquet_files = [Path(f) for f in parquet_files]

        # Precompute cumulative number of rows for indexing
        self.file_row_counts = []
        total = 0
        for f in self.parquet_files:
            parquet_file = pq.ParquetFile(f)
            n = parquet_file.metadata.num_rows
            self.file_row_counts.append((total, n))  # (offset, num_rows)
            total += n

        self.total_events = total

    def len(self):
        return self.total_events

    def _locate(self, idx):
        for file_idx, (offset, n_rows) in enumerate(self.file_row_counts):
            if idx < offset + n_rows:
                local_idx = idx - offset
                return file_idx, local_idx
        raise IndexError(f"Index {idx} out of bounds")

    def get(self, idx):
        file_idx, local_idx = self._locate(idx)
        f = self.parquet_files[file_idx]

        # Read the full table for simplicity
        table = pq.read_table(f, columns=["x", "y_class", "y_energy"])

        # Extract the row
        x = np.array(table["x"][local_idx].as_py(), dtype=np.float32)   # <-- use as_py()
        y_class = int(table["y_class"][local_idx].as_py())
        y_energy = float(table["y_energy"][local_idx].as_py())

        return Data(
            x=torch.from_numpy(x),
            y_class=torch.tensor(y_class, dtype=torch.long),
            y_energy=torch.tensor(y_energy, dtype=torch.float32)
        )
