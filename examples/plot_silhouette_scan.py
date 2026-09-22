import argparse
import os

import h5py
import numpy as np

from BIBgen.analysis import silhouette_scan, MaiaAxis

def main(args):
    mcpath = args.mc_file
    gen_input = {}
    for entry in args.gen_files:
        entry_split = entry.split(",")
        assert len(entry_split) == 2, "Each gen-files entry should be a tuple of path and name"
        assert entry_split[0].endswith(".hdf5")
        gen_input[entry_split[1]] = (entry_split[0], entry_split[1].lower())

    outpath = args.out
    os.makedirs(outpath, exist_ok=True)

    with h5py.File(mcpath, "r") as mcfile:
        mu = np.array(mcfile["transformation/mu"])
        std = np.array(mcfile["transformation/std"])
        stored_log_energy = bool(mcfile["transformation"].attrs.get("log_energy", False))

        mcdata = {event_id : np.array(mcfile["test/" + event_id + "/tau0"]) for event_id in mcfile["test"].keys()}

    gendata = {}
    for name in gen_input:
        with h5py.File(gen_input[name][0], "r") as genfile:
            gendata[name] = {event_id : np.array(genfile[event_id]) for event_id in genfile.keys()}

    for event_id in ["evt_903", "evt_901", "evt_902", "evt_904", "evt_905"]:
        nhits = len(mcdata[event_id])
        ks = np.unique(np.round(np.linspace(0.01 * nhits, 0.60 * nhits, 50)).astype(int))
        
        mc_scores = silhouette_scan(mcdata[event_id], ks)
        gen_scores = {name : silhouette_scan(gendata[name][event_id], ks) for name in gendata}

        with MaiaAxis(outpath + event_id + "_scan.png") as ax:
            ax.plot(ks, mc_scores, label="MC")
            for name in gen_scores:
                ax.plot(ks, gen_scores[name], label=name)

            ax.set_xlabel("Number of clusters")
            ax.set_ylabel("Average silhouette score")
            ax.legend()

    return 0

if __name__ == "__main__":
    # uv run plot_silhouette_scan.py /ospool/uc-shared/project/futurecolliders/rosep8/raw_cyl_phipi4_large_logE.hdf5 generation/v11_like.hdf5,Deepsets generation/v12_like.hdf5,MLP
    parser = argparse.ArgumentParser()
    parser.add_argument("mc_file")
    parser.add_argument("gen_files", nargs="+", help="One or more <path.hdf5>,<name> entries; name is the legend label")
    parser.add_argument("-o", "--out", default="plots/clustering/")
    print("\nFinished with exit code:", main(parser.parse_args()))