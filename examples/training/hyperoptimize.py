import json
import argparse

import h5py
import torch
import numpy as np

from BIBgen.losses import GaussianNLLLoss, DecoupledGaussianNLLLoss
from BIBgen.training import BatchedDataLoader, hyperoptimize

def main(args):
    inpath = args.inpath
    nepochs = args.epochs
    schedule_path = args.noise_schedule
    model_template_config_path = args.model_template_config
    batch_size = args.batch_size
    outpath = args.out
    assert inpath.endswith(".hdf5")
    assert schedule_path.endswith(".csv")
    assert model_template_config_path.endswith(".json")
    assert outpath.endswith(".json")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")

    with open(model_template_config_path, "r") as fin:
        model_template = json.load(fin)

    if "predict_variances" in model_template["hyperparameters"] and model_template["hyperparameters"]["predict_variances"]:
        gaussian_nll = DecoupledGaussianNLLLoss(variance_loss_weight=args.variance_loss_weight)
        loss_fn = lambda pred, y, tau: gaussian_nll(pred[0], pred[1], y)
    else:
        gaussian_nll = GaussianNLLLoss()
        loss_fn = lambda pred, y, tau: gaussian_nll(pred, schedule[tau], y)

    schedule = torch.from_numpy(np.loadtxt(schedule_path)).to(device)

    with h5py.File(inpath, "r") as infile:
        training_loader = BatchedDataLoader(infile, "training", batch_size=batch_size)
        validation_loader = BatchedDataLoader(infile, "validation", batch_size=batch_size)

        best_val_loss, best_params = hyperoptimize(
            model_template,
            training_loader, 
            validation_loader, 
            loss_fn,
            device,
            n_timesteps=len(schedule),
            nepochs=nepochs
        )
        
    print("Best validation loss:", best_val_loss)
    with open(outpath, "w") as fout:
        json.dump(best_params, fout)
        
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to hyperoptimize an equivariant denoising model")
    parser.add_argument("inpath", help="Diffused training data")
    parser.add_argument("noise_schedule", help="Noise schedule of forward diffusion process")
    parser.add_argument("model_template_config", help="json file specifying model name and hyperparameter ranges")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("-b", "--batch-size", type=int, default=5, help="Batch size")
    parser.add_argument("--variance-loss-weight", type=float, default=1.0, help="Weight on the variance-training loss term (only used when predict_variances=True)")
    parser.add_argument("-o", "--out", default="best.json", help="Output optimal model config")
    print("\nFinished with exit code:", main(parser.parse_args()))