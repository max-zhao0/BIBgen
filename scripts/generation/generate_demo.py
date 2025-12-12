import argparse

import h5py
import torch

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("-c", "--condor", action="store_true")
parser.add_argument("-n", "--nhits", type=int)
args = parser.parse_args()

if args.condor:
    from diffusion import EquivariantDenoiser
    from generation import generate_sphered
else:
    from BIBgen.models import EquivariantDenoiser
    from BIBgen.generation import generate_sphered

def main(args):
    model_path = args.model_path
    nhits = args.nhits
    assert model_path.endswith(".pth")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using {device} device")
    
    model = EquivariantDenoiser(
        n_timesteps = 100,
        tau_encoding_dimension = 32,
        position_encoding_dimension = 64,
        hidden_layer_size = 256,
        n_hidden_layers = 4
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))

    sphered = generate_sphered(model, nhits, device, demo=True, verbosity=1)
    
    with h5py.File("demo.hdf5", "w") as fout:
        fout.create_dataset("evt_demo", data=sphered.cpu())

    return 0

if __name__ == "__main__":
    print("\nFinished with exit code:", main(args))