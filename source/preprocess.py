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


def preprocess_data(cfg):

    for run in cfg.preprocessing.runs:
        input_files = glob.glob(cfg.preprocessing.input_dirpath + f"/{run}/*.root")
        output_dir = cfg.preprocessing.output_dirpath + f"/{run}/"

        assert len(input_files) > 0, f"No input files found in {input_files}"
        logging.info(f"Found {len(input_files)} input files to process.")

        logging.info(f"Output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        for fpath in input_files:

            output_torch_path = os.path.basename(fpath).replace(".root", ".pt")
            output_torch_path = os.path.join(output_dir, output_torch_path)

            stats = None if not cfg.preprocessing.normalise else cfg.stats

            dataset = GraphDataset(fpath, compute_edges=cfg.preprocessing.compute_edges, stats=stats)

            logging.info(f"Saving dataset to {output_torch_path}")
            torch.save(dataset, output_torch_path)
