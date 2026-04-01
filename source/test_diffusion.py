import argparse
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
import os
import logging
from typing import List
from scipy.optimize import curve_fit
import uproot
from copy import deepcopy

from source.diffusion_model import DiffusionSchedule, SparseDiffusionModel, q_sample
from source.train_diffusion import GravNetLightning
from source.dataset import GraphDataModule, unnorm_data

from dataclasses import dataclass
from tqdm import tqdm
from pathlib import Path


@dataclass
class ResolutionHistogram:
    mean: float
    std_dev: float
    param_range: List
    values: np.ndarray
    edges: np.ndarray

    def get_hname(self):
        return f"{self.param_range[0]} - {self.param_range[1]}"
    
    def as_numpy(self):
        return self.values, self.edges


# Fit a line to the bias points
def line(x, m, c):
    return m*x + c

def load_run_config(run_dir: str):

    if not os.path.isdir(run_dir):
        logging.error(f"Run directory {run_dir} does not exist")
        raise ValueError(f"Run directory {run_dir} does not exist")

    cfg_path = Path(run_dir) / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    logging.info(f"Loaded config from {cfg_path}")
    return cfg


def get_checkpoint_path(cfg: DictConfig):

    # Load a specific checkpoint if provided
    if cfg.testing.checkpoint_filepath is not None:
        if not os.path.isfile(cfg.testing.checkpoint_filepath):
            logging.error(f"Checkpoint file {cfg.testing.checkpoint_filepath} does not exist")
            raise ValueError(f"Checkpoint file {cfg.testing.checkpoint_filepath} does not exist")
        
        logging.info(f"Using checkpoint: {cfg.testing.checkpoint_filepath}")
        return cfg.testing.checkpoint_filepath
    
    # Otherwise, find the latest checkpoint in the run directory
    ckpt_dir = os.path.join(cfg.testing.run_dir, cfg.testing.checkpoint_dir, "*.ckpt")
    logging.info(f"Searching for checkpoints in directory: {ckpt_dir}")
    checkpoint_paths = glob.glob(ckpt_dir)
    
    if len(checkpoint_paths) == 0:
        logging.error("No checkpoint found in run directory")
        raise ValueError("No checkpoint found in run directory")
    elif len(checkpoint_paths) > 1:
        logging.warning("Multiple checkpoints found, using the first one")
    
    checkpoint_path = checkpoint_paths[0]

    logging.info(f"Using checkpoint: {checkpoint_path}")
    return checkpoint_path


@torch.no_grad()
def run_inference(cfg, model, dataloader, device):
    model.eval()
    model.freeze()
    
    targets_true = {t : [] for t in cfg.training.targets}
    targets_pred = {t : [] for t in cfg.training.targets}

    for batch in tqdm(dataloader, desc="Running Inference..."):
        batch = batch.to(device)

        target_truth = {t: batch[t] for t in cfg.training.targets if hasattr(batch, t)}
        
        y_true = torch.stack(
        [
            target_truth[t].float()
            for t in cfg.training.targets
        ],
        dim=1,  # (batch, n targets)
        )

        y_pred = model.sample(batch)

        # print(y_pred.shape)
        # print(y_true.shape)
        
        for i, t in enumerate(cfg.training.targets):
            targets_true[t].append(y_true.cpu()[:,i].numpy())
            targets_pred[t].append(y_pred.cpu()[:,i].numpy())
        
        # if nb > 500: break

        # break

    for t in cfg.training.targets:
        targets_true[t] = np.concatenate(targets_true[t])
        targets_pred[t] = np.concatenate(targets_pred[t])
    
    
    # Apply mean, std scaling
    if cfg.preprocessing.normalise:
        stats = cfg.stats
        for t in cfg.training.targets:
            targets_true[t] = unnorm_data(targets_true[t], stats[t].mean, stats[t].std)
            targets_pred[t] = unnorm_data(targets_pred[t], stats[t].mean, stats[t].std) 

    # Apply any necessary inverse transformations here
    # for t in cfg.training.targets:
    #     if cfg.variables.get(t, None):
    #         if cfg.variables[t].get("transformation", None) == "log10":
    #             targets_true[t] = 10 ** targets_true[t]
    #             targets_pred[t] = 10 ** targets_pred[t]
    #         elif cfg.variables[t].get("transformation", None) == "eta":
    #             targets_true[t] = 2 * np.arctan(np.exp(targets_true[t])) - np.pi/2
    #             targets_pred[t] = 2 * np.arctan(np.exp(targets_pred[t])) - np.pi/2
    
    

    return targets_true, targets_pred


def plot_resolution_hists(cfg, varname, targets_true, targets_pred, bias_params=None):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"
    assert "bins" in var_cfg, f"No binning defined for variable {varname} in config"

    y_true = deepcopy(targets_true[varname])
    y_pred = deepcopy(targets_pred[varname])

    if bias_params is not None:
        a, b = bias_params
        correction = 1.0 + a * y_pred + b
        y_pred = y_pred / correction

    resolution = (y_pred - y_true) / y_true


    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()

    histograms = []

    bin_labels = []
    for edgemin, edgemax in var_cfg.bins:
        if edgemax == np.inf:
            bin_labels.append(f"> {edgemin} {var_cfg.get('unit', '')}")
        else:
            bin_labels.append(f"{edgemin}–{edgemax} {var_cfg.get('unit', '')}")

    for i, ((ymin, ymax), label) in enumerate(zip(var_cfg.bins, bin_labels)):
        try:
            ax = axes[i]
        except IndexError:
            logging.warning("More bins defined than subplots available")
            break

        mask = (y_true >= ymin) & (y_true < ymax)
        res_bin = resolution[mask]

        # if len(res_bin) < 50:
        #     ax.text(0.5, 0.5, "Too few events", ha="center", va="center")
        #     continue

        values, edges, _ = ax.hist(
            res_bin,
            bins=60,
            range=(-1, 1),
            histtype="step",
            linewidth=1.5,
        )

        mean = np.mean(res_bin)
        sigma = np.std(res_bin)

        histograms.append(ResolutionHistogram(mean=mean, std_dev=sigma, values=values, edges=edges, param_range=[ymin, ymax]))

        ax.set_title(label)
        latex_var = var_cfg.get('latex', varname)
        ax.set_xlabel(rf"$({latex_var}^{{\rm reco}} - {latex_var}^{{\rm true}}) / {latex_var}^{{\rm true}}$")
        ax.set_ylabel("Events")

        ax.text(
            0.05, 0.95,
            f"Mean = {mean:.3f}\nStd. = {sigma:.3f}",
            transform=ax.transAxes,
            va="top",
        )

    fig.delaxes(axes[-1])

    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_ResolutionPerBin")
    if bias_params is not None: outfile += "_BiasCorrected"

    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")

    return histograms


def plot_true_vs_reco(cfg, varname, targets_true, targets_pred, logscale=False, bias_params=None):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"

    y_true = deepcopy(targets_true[varname])
    y_pred = deepcopy(targets_pred[varname])

    if bias_params is not None:
        a, b = bias_params
        correction = 1.0 + a * y_pred + b
        # correction = a * y_pred + b
        y_pred = y_pred / correction


    fig, ax = plt.subplots(figsize=(6, 6))

    # Log-spaced bins work best here
    if logscale:
        bins = np.logspace(
            np.log10(min(y_true.min(), y_pred.min()) + 1e-6),
            np.log10(max(y_true.max(), y_pred.max()) + 1e-6),
            100,
        )
    else:
        bins = np.linspace(
            min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max()),
            100,
        )

    # print(y_true)
    # print(y_pred)
    # print(bins)

    h , xedges, yedges, image= ax.hist2d(
        y_true,
        y_pred,
        bins=[bins, bins],
        norm="log",
        cmap="viridis",
    )

    plt.colorbar(image, label="Events", ax=ax)

    # y = x reference line
    if logscale:
        x = np.logspace(
            np.log10(y_true.min()+1e-6),
            np.log10(y_true.max()+1e-6),
            100,
        )
    else:
        x = np.linspace(
            y_true.min(),
            y_true.max(),
            100,
        )
            
    ax.plot(x, x, "r--", linewidth=1, label=rf"${var_cfg.get('latex', varname)}^{{\rm reco}} = {var_cfg.get('latex', varname)}^{{\rm true}}$")
    ax.axvline(np.mean(y_true), color="gray", linestyle="--", linewidth=0.5)

    if logscale:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xlabel(rf"True ${var_cfg.get('latex', varname)}$ {var_cfg.get('unit', '')}")
    ax.set_ylabel(rf"Reconstructed ${var_cfg.get('latex', varname)}$ {var_cfg.get('unit', '')}")
    ax.legend()
    # ax.set_xlim(0, 4500)
    # ax.set_ylim(0, 4500)
    plt.tight_layout()

    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_TrueVsReco")
    if bias_params is not None: outfile += "_BiasCorrected"
    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")

    return h, xedges, yedges
    

def plot_resolution_vs_target(cfg, varname, targets_true, targets_pred):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"
    assert "bins" in var_cfg, f"No binning defined for variable {varname} in config"

    y_true = targets_true[varname]
    y_pred = targets_pred[varname]

    resolution = (y_pred - y_true) / y_true

    bin_centers = []
    sigmas = []
    sigma_errs = []

    for emin, emax in var_cfg.bins:
        mask = (y_true >= emin) & (y_true < emax)
        res_bin = resolution[mask]

        if len(res_bin) < 50:
            continue

        sigma = np.std(res_bin)
        sigma_err = sigma / np.sqrt(2 * len(res_bin))
        center = np.mean(y_true[mask])

        bin_centers.append(center)
        sigmas.append(sigma)
        sigma_errs.append(sigma_err)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        bin_centers,
        sigmas,
        yerr=sigma_errs,
        fmt="o",
    )

    for x, s in zip(bin_centers, sigmas):
        if s < 1.0: continue
        ax.annotate(
            "",                    # no text
            xy=(x, 1.0),            # arrow head
            xytext=(x, 0.9),        # arrow tail
            arrowprops=dict(
                arrowstyle="->",
                linewidth=1.5,
            ),
        )

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    latex_var = var_cfg.get('latex', varname)
    unit = var_cfg.get('unit', '')
    ax.set_xlabel(rf"True ${latex_var}$ {unit}")
    ax.set_ylabel(rf"${latex_var}$ Resolution (σ)")
    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_ResolutionVsTrue")
    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")


def plot_bias(cfg, varname, targets_true, targets_pred, bias_params=None):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"
    assert "bins" in var_cfg, f"No binning defined for variable {varname} in config"

    y_true = targets_true[varname]
    y_pred = targets_pred[varname]

    resolution = (y_pred - y_true) / y_true

    bin_centers = []
    sigmas = []
    sigma_errs = []
    means = []

    for emin, emax in var_cfg.bins:
        mask = (y_true >= emin) & (y_true < emax)
        res_bin = resolution[mask]

        if len(res_bin) < 50:
            continue

        sigma = np.std(res_bin)
        sigma_err = sigma / np.sqrt(2 * len(res_bin))
        center = np.mean(y_true[mask])
        mean = np.mean(res_bin)
        means.append(mean)

        bin_centers.append(center)
        sigmas.append(sigma)
        sigma_errs.append(sigma_err)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        bin_centers,
        means,
        yerr=sigmas,
        fmt="o",
    )

    for x, s in zip(bin_centers, sigmas):
        if -1 < s < 1.0: continue
        if s > 1.0:
            ax.annotate(
                "",                    # no text
                xy=(x, 1.0),            # arrow head
                xytext=(x, 0.45),        # arrow tail
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=1.5,
                ),
            )
        elif s < -1.0:
            ax.annotate(
                "",                    # no text
                xy=(x, -1.0),            # arrow head
                xytext=(x, -0.45),        # arrow tail
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=1.5,
                ),
            )
    
    popt, pcov = curve_fit(
        line, bin_centers, means,
        sigma=sigmas,
        absolute_sigma=True
    )

    x_fit = np.linspace(min(bin_centers), max(bin_centers), 100)
    y_fit = line(x_fit, *popt)
    ax.plot(x_fit, y_fit, "r--", label=f"Fit: y = {popt[0]:.3e} x + {popt[1]:.3e}")
    ax.legend()


    # ax.set_xscale("log")
    ax.set_ylim(-1, 1)
    latex_var = var_cfg.get('latex', varname)
    unit = var_cfg.get('unit', '')
    ax.set_xlabel(rf"True ${latex_var}$ {unit}")
    ax.set_ylabel(rf"${latex_var}$ Bias (Mean Scale $\pm$ Std Dev)")
    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_Bias")
    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")

    return popt

def plot_true(cfg, varname, targets_true, logscale=False, bias_params=None):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"

    y_true = deepcopy(targets_true[varname])

    fig, ax = plt.subplots(figsize=(6, 6))

    if logscale:
        bins = np.logspace(
            np.log10(y_true.min()),
            np.log10(y_true.max()),
            100,
        )
        ax.set_xscale("log")
    else:
        bins = np.linspace(
            y_true.min(),
            y_true.max(),
            100,
        )

    ax.hist(
        y_true,
        bins=bins,
        histtype="step",
        linewidth=1.5,
    )
    latex_var = var_cfg.get('latex', varname)
    unit = var_cfg.get('unit', '')
    ax.set_xlabel(rf"True ${latex_var}$ {unit}")
    ax.set_ylabel("Events")
    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_TrueDist")
    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")



def plot_pairwise_2dhists(targets, bins=50, cmap="viridis"):
    keys = list(targets.keys())
    n = len(keys)

    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
    
    # Convert to numpy arrays
    data = {k: np.asarray(v) for k, v in targets.items()}

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            if i < j:
                # Upper triangle: turn off
                ax.axis("off")
                continue

            x = data[keys[j]]
            y = data[keys[i]]

            if i == j:
                # Diagonal: 1D histogram
                ax.hist(x, bins=bins, histtype="stepfilled", alpha=0.7)
            else:
                # Lower triangle: 2D histogram
                h = ax.hist2d(x, y, bins=bins, cmap=cmap)

            # Labels only on outer axes
            if i == n - 1:
                ax.set_xlabel(f"{keys[j]}")
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(f"{keys[i]}")
            else:
                ax.set_yticklabels([])

    plt.tight_layout()
    return fig, axes

def plot_true_and_reco(cfg, varname, targets_true, targets_pred, logscale=False, bias_params=None):

    var_cfg = cfg.variables.get(varname, {})

    assert var_cfg is not None, f"No config found for variable {varname}"

    y_true = deepcopy(targets_true[varname])
    y_pred = deepcopy(targets_pred[varname])

    fig, ax = plt.subplots(figsize=(6, 6))

    if logscale:
        bins = np.logspace(
            np.log10(y_true.min()),
            np.log10(y_true.max()),
            100,
        )
        ax.set_xscale("log")
    else:
        bins = np.linspace(
            y_true.min(),
            y_true.max(),
            100,
        )

    ax.hist(
        y_true,
        bins=bins,
        histtype="stepfilled",
        linewidth=1.5,
        color="blue",
        label="True",
    )
    ax.hist(
        y_pred,
        bins=bins,
        histtype="step",
        linewidth=1.5,
        color="red",
        label="Reco",
    )
    
    latex_var = var_cfg.get('latex', varname)
    unit = var_cfg.get('unit', '')
    ax.set_xlabel(rf"True ${latex_var}$ {unit}")
    ax.set_ylabel("Events")
    outfile = os.path.join(cfg.testing.run_dir, f"{varname}_TrueDist")
    for fmt in cfg.plotting.formats:
        extn = fmt.lower().replace(".", "")
        plt.savefig(f"{outfile}.{extn}", dpi=300)
    logging.info(f"Plotted {outfile}")


def run_diffusion_testing(cfg: DictConfig):

    logging.info("Starting testing...")

    # Load the config for this run
    if cfg.testing.run_dir is None:
        logging.info("Finding most recent run directory...")
        base_dir = Path(os.path.join("logs", cfg.logging.name.replace("test", "train")))
        dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        latest_dir = max(dirs, key=lambda d: d.stat().st_ctime)
        cfg.testing.run_dir = str(latest_dir)
        logging.info(f"Using latest run directory: {cfg.testing.run_dir}")


    cfg_this_run = load_run_config(cfg.testing.run_dir)
    
    device = torch.device("cuda" if cfg.device == "gpu" else "cpu")
    
    # Get the checkpoint path
    checkpoint_path = get_checkpoint_path(cfg)

    # Load the dataset
    pt_files = []
    for run in cfg_this_run.data.runs:
        run_files = glob.glob(os.path.join(cfg_this_run.data.datapath, str(run), "*.pt"))
        logging.info(f"Found {len(run_files)} for run {run}")
        pt_files += run_files
    assert len(pt_files) > 0, "No .pt files found" 
    logging.info(f"Found {len(pt_files)} files total")

     # Reconstruct model exactly as in training
    backbone = SparseDiffusionModel(cfg_this_run.model.n_targets, input_channels=1).to(device)
    model = GravNetLightning.load_from_checkpoint(
            checkpoint_path,
            model=backbone,
            map_location="cpu",
            # strict=False,
        )
    model.to(device)

    datamodule = GraphDataModule(
        pt_files=pt_files,
        batch_size=cfg.testing.batch_size,
        num_workers=cfg.testing.num_workers,
    )
    datamodule.setup()


    targets_true, targets_pred = run_inference(
        cfg_this_run,
        model,
        datamodule.test_dataloader(),
        device=device
    )

    output_file = uproot.recreate(os.path.join(cfg.testing.run_dir, cfg.testing.output_file))

    plot_pairwise_2dhists(targets_true, bins=40)
    plt.savefig(os.path.join(cfg.testing.run_dir, "TruePairwise2DHists.png"), dpi=300)

    plot_pairwise_2dhists(targets_pred, bins=40)
    plt.savefig(os.path.join(cfg.testing.run_dir, "PredPairwise2DHists.png"), dpi=300)


    for varname in cfg_this_run.training.targets:

        res_hists = plot_resolution_hists(cfg, varname, targets_true, targets_pred)
        # plot_resolution_vs_target(cfg, varname, targets_true, targets_pred)
        true_v_reco = plot_true_vs_reco(cfg, varname, targets_true, targets_pred, logscale=False)
        plot_true_and_reco(cfg, varname, targets_true, targets_pred)

        continue
        bias_fit = plot_bias(cfg, varname, targets_true, targets_pred)

        res_hists_bias_corrected = plot_resolution_hists(cfg, varname, targets_true, targets_pred, bias_params=bias_fit)
        # plot_resolution_vs_target(cfg, varname, targets_true, targets_pred)
        true_v_rec_bias_corrected = plot_true_vs_reco(cfg, varname, targets_true, targets_pred, bias_params=bias_fit)
        # _ = plot_bias(cfg, varname, targets_true, targets_pred, bias_params=bias_fit)

        # Dump the results to a file
        means = np.array([h.mean for h in res_hists])
        std_devs = np.array([h.std_dev for h in res_hists])
        param_ranges = np.array([h.param_range for h in res_hists])
        bias_fit_params = np.array(bias_fit)
        
        output_file[f"{varname}/resHists/stats"] = {"means": means, "std_devs": std_devs, "param_ranges": param_ranges}
        output_file[f"{varname}/resHists/bias"] = {"bias_fit_params": bias_fit_params}
        
        output_file[f"{varname}/true_vs_reco"] = true_v_reco
        output_file[f"{varname}/true_vs_reco_BiasCorr"] = true_v_rec_bias_corrected

        means_bias_corr = np.array([h.mean for h in res_hists_bias_corrected])
        std_devs_bias_corr  = np.array([h.std_dev for h in res_hists_bias_corrected])
        param_ranges_bias_corr  = np.array([h.param_range for h in res_hists_bias_corrected])
        
        output_file[f"{varname}/resHistsBiasCorr/stats"] = {"means": means_bias_corr , "std_devs": std_devs_bias_corr , "param_ranges": param_ranges_bias_corr }

        for hist in res_hists:
            output_file[f"{varname}/resHists/{hist.get_hname()}"] = hist.as_numpy()
        
        for hist in res_hists_bias_corrected:
            output_file[f"{varname}/resHistsBiasCorr/{hist.get_hname()}"] = hist.as_numpy()
            