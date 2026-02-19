import uproot
import awkward as ak
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import logging

import glob
import os
import logging
import argparse
import torch
from omegaconf import DictConfig
from source.dataset import GraphDataset


def preprocess_data(input_files, output_dir):

    assert len(input_files) > 0, f"No input files found in {input_files}"
    logging.info(f"Found {len(input_files)} input files to process.")

    logging.info(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for fpath in input_files:

        output_torch_path = os.path.basename(fpath).replace(".root", ".pt")
        output_torch_path = os.path.join(output_dir, output_torch_path)

        dataset = GraphDataset(fpath)

        logging.info(f"Saving dataset to {output_torch_path}")
        torch.save(dataset, output_torch_path)



# def preprocess_data(cfg: DictConfig):

#     input_files = glob.glob(cfg.preprocessing.input_dirpath)

#     assert len(input_files) > 0, f"No input files found in {cfg.preprocessing.input_dirpath}"
#     logging.info(f"Found {len(input_files)} input files to process.")

#     output_dir = cfg.preprocessing.output_dirpath
#     logging.info(f"Output directory: {output_dir}")
#     os.makedirs(output_dir, exist_ok=True)


#     for fpath in input_files:

#         output_torch_path = os.path.basename(fpath).replace(".root", ".pt")
#         output_torch_path = os.path.join(output_dir, output_torch_path)

#         dataset = GraphDataset(fpath)

#         logging.info(f"Saving dataset to {output_torch_path}")
#         torch.save(dataset, output_torch_path)



# if __name__ == "__main__":

#     input_files = glob.glob("/home/benwilson/data/pinpointG4_data/root/10000/*.root")

#     num_bins = 2048
#     num_layers = 100
    
#     for fpath in input_files:

#         output_torch_path = os.path.basename(fpath).replace(".root", ".pt")

#         dataset = CNNProjectionDataset(
#             fpath, bins=(num_bins, num_bins, num_layers)
#         )

#         logging.info(f"Saving dataset to {output_torch_path}")
#         torch.save(dataset, output_torch_path)


# def preprocess_root_to_parquet(root_files, output_file, output_dir="data/parq"):
#     output_dir = Path(output_dir)
#     output_file = Path(output_file)
#     output_dir.mkdir(exist_ok=True)

#     if not isinstance(root_files, list): root_files = [root_files]

#     all_events = []
#     nskipped_events = 0

#     for file_idx, f in enumerate(root_files):
#         print(f"Processing file {file_idx+1}/{len(root_files)}: {f}")
#         file = uproot.open(f)

#         hits = file["Hits/pixelHits"].arrays(
#             ["hit_layerID", "hit_rowID", "hit_colID"],
#             library="ak",
#         )
#         truth = file["event"].arrays(
#             ["initE", "initPDG", "processName"],
#             library="ak",
#         )

#         num_events = len(truth["initE"])

#         for i in tqdm(range(num_events)):
#             row = ak.to_numpy(hits["hit_rowID"][i])
#             col = ak.to_numpy(hits["hit_colID"][i])
#             layer = ak.to_numpy(hits["hit_layerID"][i])

#             if len(row) < 100:
#                 nskipped_events += 1
#                 continue  # skip small events

            
#             row = ak.to_numpy(self.hits["hit_rowID"][local_idx])
#             col = ak.to_numpy(self.hits["hit_colID"][local_idx])
#             layer = ak.to_numpy(self.hits["hit_layerID"][local_idx])

#             col = col / (np.amax(col) + 1e-6)
#             row = row / (np.amax(row) + 1e-6)
#             layer = layer / (np.amax(layer) + 1e-6)

#             x = np.stack([row, col, layer], axis=1).astype(np.float32)

#             energy = float(self.truth["initE"][local_idx])
#             pdg = int(self.truth["initPDG"][local_idx])
#             process = str(self.truth["processName"][local_idx])

#             energy = np.log1p(energy)

#             if "NC" in process:
#                 interaction = 3
#             else:
#                 interaction = {12: 0, 14: 1, 16: 2}.get(abs(pdg), 3)

#             # log energy
#             energy = float(truth["initE"][i])
#             energy = np.log1p(energy)

#             pdg = int(truth["initPDG"][i])
#             process = str(truth["processName"][i])
#             interaction = 3 if "NC" in process else {12: 0, 14: 1, 16: 2}.get(abs(pdg), 3)

#             # store as dict (Parquet doesn't support ragged arrays directly)
#             all_events.append({
#                 "x": x.tolist(),  # save as nested list
#                 "y_class": interaction,
#                 "y_energy": energy
#             })

#     # Convert to DataFrame
#     df = pd.DataFrame(all_events)
#     output_file = output_dir / output_file
#     df.to_parquet(output_file, engine="pyarrow", index=False)
#     logging.info(f"Saved {len(df)} events to {output_file}. Skipped {nskipped_events} events with < 100 hits")
