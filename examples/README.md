# Preprocessing and Readying Data

## File Conversion
Run ``python convert_slcio_files.py -i [inputfiles] -o [outputfile]`` on a machine with pyLCIO configured. ``inputfiles`` should be LCIO format, ``outputfile`` should end with ``.hdf5``.

## Make Training Data

```bash
python make_training_data.py [mm_file] [mp_file] -o [outfile] -s [train,val,test] -c -p [phi_window]
```

By default, the energy feature is stored as ``ln(E)`` rather than raw ``E``. Pass ``--raw-energy``
to store raw ``E`` instead. The chosen setting is stamped into the output file
(``f["transformation"].attrs["log_energy"]``) so downstream scripts can detect it automatically.

## Make Diffused Data
Run ``diffuse.py [raw_data] [noise_schedule] -o [outfile]``.

## Noise Schedule

``noise_schedule.csv`` files are generated with ``make_noise_schedule.py``:

```bash
python make_noise_schedule.py quadratic [out.csv] -T [n_timesteps] --scale [scale]
python make_noise_schedule.py cosine [out.csv] -T [n_timesteps] --target-alpha-bar-t [alpha_bar_T]
```

``quadratic`` reproduces the original ``beta(tau) = scale * tau^2`` schedule (defaults match
``config/noise_schedules/noise_schedule.csv`` exactly). ``cosine`` follows Nichol & Dhariwal's "Improved DDPM"
schedule, solved so that the cumulative ``alpha_bar`` at the final timestep matches
``--target-alpha-bar-t`` exactly. Compared to the quadratic schedule, cosine spreads noise more
evenly through the middle of the chain, but concentrates a sharp jump in the final step or two;
raise ``--target-alpha-bar-t`` to soften that jump (``--s`` has little effect on it).

Compare schedules before committing to a full diffuse+train run with:

```bash
python plot_noise_schedule.py [schedule1.csv] [schedule2.csv] ... --labels [name1] [name2] ... -o [comparison.png]
```

This prints ``beta_min``/``beta_max``/``alpha_bar_T`` for each schedule and, if ``-o`` is given,
saves a plot of ``beta(tau)`` and ``alpha_bar(tau)``.

## Model Configuration

Model architecture is specified via a JSON config, e.g. ``config/models/equivariant_denoiser.json``:

```json
{
  "name": "EquivariantDenoiser",
  "hyperparameters": {
    "tau_encoding_dimension": 32,
    "position_encoding_dimension": 64,
    "hidden_layer_size": 256,
    "n_hidden_layers": 4,
    "use_position_encoding": false
  }
}
```

``use_position_encoding`` defaults to ``false`` — hit coordinates (φ, s, z) are fed to the model
directly rather than through an additional Fourier encoding.

When ``use_position_encoding`` is ``true``, ``position_encoding_kind`` selects the mapping
(following Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low
Dimensional Domains"):

- ``"learned"`` (default) — the original per-axis encoding, with frequencies trained by gradient
  descent alongside the rest of the network (``FourierEncoding`` in ``models/common.py``, one
  independent encoder each for φ, s, z). Kept as a baseline; the paper (Appendix A.3) finds
  gradient descent does not meaningfully move — or improve on — well-scaled fixed frequencies.
- ``"positional"`` — fixed, deterministic, log-linearly spaced frequencies per axis
  (``PositionalEncoding``), axis-aligned by construction (the paper's "positional encoding").
- ``"gaussian"`` — fixed random frequencies sampled once from a Gaussian and shared across a
  *single joint* encoder over (φ, s, z) (``GaussianFourierFeatures``), rather than three
  independent per-axis encoders. This mixes coordinates the way the paper's random Fourier
  feature mapping does, and the paper finds it outperforms axis-aligned positional encoding,
  especially off-axis (Appendix A.5).

``"positional"`` and ``"gaussian"`` both require ``position_encoding_scale``: either a single
float shared across φ, s, z, or a ``[sigma_phi, sigma_s, sigma_z]`` triple. **This codebase starts
with independent per-axis scales** rather than one shared scale, because φ is restricted to a
narrow window (``--phi-window``, e.g. π/4) while s and z span the full detector — the "densely
clustered" φ coordinate likely needs a different frequency scale than s/z. If the φ window is
later widened toward the full detector extent, φ's frequency content may end up comparable to s/z,
at which point collapsing back to one shared scale is worth revisiting.
The best scale(s) are tuned per dataset on held-out validation loss; see
``training/make_sweep_configs.py`` for generating a scale-sweep grid of configs.

``"gaussian"`` additionally accepts ``position_encoding_seed`` (int, optional): seeds the random
draw of the frequency matrix ``B`` (see ``GaussianFourierFeatures`` in ``models/common.py``) so
it's reproducible across runs. Omitted by default, in which case every run draws an independent
random ``B`` — fine when scale differences are large and clear, but when a sweep turns up two
scales whose validation loss is nearly tied, that tie may just reflect which random ``B`` each one
happened to draw rather than a real difference. Rerunning the tied candidates with a few different
seeds each (see ``--seed`` on ``make_sweep_configs.py`` below) checks whether the ranking actually
holds up. Ignored for ``"learned"``/``"positional"``, which have no randomness to seed.

``predict_variances`` defaults to ``false``. When ``true``, the model predicts its own per-hit,
per-feature variance alongside the mean, turning the loss into a genuine Gaussian NLL rather than
a ``β_τ``-weighted MSE (the fixed noise-schedule value is used as the variance when this is
``false``). See ``config/models/equivariant_denoiser_learned_variance.json`` for an example. ``train.py``
and ``generate_like.py`` handle the wiring automatically based on ``model.predict_variances`` — no
other CLI arguments change; ``noise_schedule`` is still required by both scripts since it also
fixes ``n_timesteps``.

When ``predict_variances`` is ``true``, training automatically uses ``DecoupledGaussianNLLLoss``
(mean and variance share every hidden layer, only diverging at the final output slice — without
decoupling, the variance head tends to collapse onto a trivial solution that just reproduces
``β_τ`` instead of learning anything input-dependent). ``--variance-loss-weight`` (default ``1.0``)
weights the variance-training term relative to the mean-training term.

## Training

```bash
python train.py [diffused_data] [noise_schedule] [model_config] -e [epochs] -b [batch_size] -t [tag]
```

``--loss`` selects the training objective:

- ``simple`` (default): direct NLL regression of the model's prediction against the one observed
  ``x_τ`` from that event's specific forward-diffusion sample, using ``β_τ`` (or the model's own
  learned variance, if ``predict_variances``) as the Gaussian's variance.
- ``nelbo``: the actual negative ELBO (Ho et al. 2020) — for ``τ ≥ 1``, KL-divergence between the
  model's predicted reverse distribution and the *true* forward-process posterior
  ``q(x_τ|x_{τ+1},x_0)`` (a closed form depending on the clean sample ``x_0``, not just the one
  noisy draw); at ``τ = 0`` this degenerates to the same reconstruction term ``simple`` already
  uses. See ``NELBOLoss`` in ``losses.py`` for the full derivation. Applies the same
  ``--variance-loss-weight`` decoupling as ``simple`` when ``predict_variances`` is ``true``.

``-t/--tag`` names the output files for this run (``denoiser_<tag>.pth``, ``history_<tag>.csv``);
defaults to the config filename stem if omitted. ``-o/--out`` overrides the output path directly.

``submit_train.sub`` reads ``(data, config, tag, schedule, loss)`` rows from ``experiments.txt``
and queues one condor job per row, so multiple runs (including runs against different noise
schedules or loss functions) can be submitted at once without output files clobbering each
other. ``loss`` is passed straight through as ``--loss`` (``simple`` or ``nelbo``); every row
must specify it explicitly now that the field exists. ``schedule`` is a filename within
``config/`` (e.g. ``noise_schedule.csv`` or ``noise_schedule_cosine.csv``) —
the whole ``config/`` directory is already transferred to the job, so no other change is needed
to use a new schedule. Likewise ``config`` is a filename within ``config/models/``.

On Della (Slurm), ``./submit_train.sh experiments.txt`` is the equivalent: one ``train.slurm`` job
per row. Run it as a program (``./submit_train.sh`` or ``bash submit_train.sh``), never with
``source``, which would close your shell when the script exits. ``*.txt`` is gitignored, so each
person keeps their own experiments file. A complete one looks like this:

```
REPO=/scratch/gpfs/<GROUP>/<netid>/BIBgen   # required: your checkout; outputs and logs go here
MAIL_USER=<netid>@princeton.edu             # optional: email when each job ends or fails
RUNTIME=12:00:00                            # optional: wall-time limit (default 12:00:00)
DATA_DIR=/scratch/gpfs/IOJALVO/mz6011/BIBgen/data   # optional: default $REPO/data

# data                                config                      tag       schedule
# schedule must be the one `data` was diffused with
diffused_cyl_phipi4_large_logE.hdf5   equivariant_denoiser.json   v1        noise_schedule.csv
diffused_cyl_phipi4_large_logE.hdf5   mlp_denoiser.json           mlp_v1    noise_schedule.csv
```

Settings are ``KEY=value`` lines; anything else that is not a comment is a row, in the same
``data config tag schedule`` order as the Condor files (commas also work). A bare ``data`` name is
looked up in ``DATA_DIR``; a path is used as given. ``config`` and ``schedule`` are bare filenames in
``config/models/`` and ``config/noise_schedules/``. Pointing ``DATA_DIR`` at the shared copy above
saves duplicating the 26 GB file. ``diffused_cyl_phipi4_large_logE.hdf5`` was diffused with
``noise_schedule.csv``.

```bash
./submit_train.sh -n experiments.txt                     # print the sbatch commands only
./submit_train.sh experiments.txt --test-only            # ask the scheduler, create no jobs
./submit_train.sh experiments.txt                        # submit
EPOCHS=51 ./submit_train.sh experiments.txt --time=04:00:00   # env knobs; extra sbatch options last
squeue -u $USER --start                                  # estimated start times
```

The defaults in ``train.slurm`` (1 GPU on a 40 GB A100 or a 40 GB MIG slice, 2 cores, 4 GB, 12 h)
are sized from a measured run: 151 epochs of ``equivariant_denoiser.json`` on the file above took
8 h 57 min and used 1.3 cores, 1.6 GB of CPU memory and 6.3 GB of GPU memory. Scale ``RUNTIME`` with
epochs, model size and dataset; a shorter limit never queues longer and sometimes much shorter.
A tag whose ``denoiser_<tag>.pth`` already exists is skipped unless you pass ``-f``.

On Perlmutter (NERSC, also Slurm), ``./submit_train_perlmutter.sh experiments.txt`` is the
equivalent: one ``train_perlmutter.slurm`` job per row. It takes the same ``data config tag
schedule`` rows as above, plus one extra required setting:

```
REPO=$PSCRATCH/BIBgen                       # required: your checkout; outputs and logs go here
ACCOUNT=mxyz                                # required: NERSC charge account (sbatch -A; see `iris -a`)
MAIL_USER=<user>@princeton.edu              # optional: email when each job ends or fails
RUNTIME=12:00:00                            # optional: wall-time limit (default 12:00:00)
DATA_DIR=/path/to/shared/data               # optional: default $REPO/data
```

``train_perlmutter.slurm`` defaults to the ``shared`` QOS, which bills per-GPU rather than
per-node (each Perlmutter GPU node has 4 A100s) -- the right choice for a single-GPU job like
this one. Pass ``--qos=regular --nodes=N --ntasks=<N*4>`` (and edit the ``srun`` line) to spread
a run across full nodes instead. Usage otherwise matches ``submit_train.sh``:

```bash
./submit_train_perlmutter.sh -n experiments.txt
./submit_train_perlmutter.sh experiments.txt
```

## Hyperparameter Optimization

```bash
python hyperoptimize.py [diffused_data] [noise_schedule] [model_template] -e [epochs] -b [batch_size] -n [trials] --timeout [hours] -o [best.json]
```

``model_template`` is a model config in ``config/model_templates/`` in which a hyperparameter may be a
search range instead of a value, e.g. ``{"type": "int", "exponentiate": true, "range": [7, 9]}``
searches ``2**7`` to ``2**9``. Each optuna trial trains for ``-e`` epochs and is scored by its best
validation loss.

The study is saved as it runs to a ``.db`` next to ``-o`` with the same name (``best.json`` ->
``best.db``), and running the same command again resumes it. ``-n`` (default 40) is the total number
of trials over all runs: completed, pruned and errored (e.g. out of memory) trials count towards it,
while a trial cut off because the process was killed is run again. ``--timeout`` (default 20 hours) stops
new trials from starting after that time; the trial in progress finishes. The best config, in the format
``train.py`` reads, is written to ``-o`` only once all trials are done.

``submit_hyperoptimize.sub`` reads ``(data, config, out, schedule)`` rows from ``experiments_hyper.txt``,
where ``config`` is a template in ``config/model_templates/``. Each row's study lives in a folder named
after ``out``, e.g. ``best_equivariant_denoiser/`` holding ``best_equivariant_denoiser.db`` and, once
finished, ``best_equivariant_denoiser.json``. The folder is transferred to the job and back, so
``condor_submit submit_hyperoptimize.sub`` resumes each unfinished study where the last job left off;
resubmit until the ``.json`` appears. Delete the folder to start a study over.

### Fourier feature scale sweep

``training/make_sweep_configs.py`` generates a grid of ``position_encoding_scale`` configs for
the ``"positional"``/``"gaussian"`` encoders (see Model Configuration above), and appends matching
rows to a manifest in the same format ``experiments.txt`` uses, so the grid submits through the
existing ``submit_train.sub``/Condor workflow with no other changes:

```bash
python make_sweep_configs.py gaussian --phi 0.5 1 2 4 8 16 32 64 --s 1 --z 1 \
    --data diffused_cyl_phipi4_large_logE.hdf5 --schedule noise_schedule.csv \
    --manifest experiments_fourier_sweep.txt
```

For ``gaussian``, ``--seed`` reruns each ``(phi, s, z)`` triple once per given seed (via
``position_encoding_seed``) instead of a single uncontrolled draw — use this to check whether a
close/tied result from a scale sweep actually holds up across different random frequency draws,
e.g. rerunning just the tied candidates:

```bash
python make_sweep_configs.py gaussian --phi 4 --s 2 4 --z 1 --seed 0 1 2 \
    --data diffused_cyl_phipi4_large_logE.hdf5 --schedule noise_schedule.csv \
    --manifest experiments_fourier_sweep_phaseB_reseed.txt
```

Because a full 3-axis grid is combinatorially expensive, sweep one axis at a time (holding the
other two fixed at a reasonable default), picking the best scale by validation loss before moving
to the next axis — φ first (since it's the axis whose "densely clustered" window most likely
needs a different scale than s/z), then s, then z — and only train the combined best-per-axis
config to full length once all three are chosen.

## Generation

```bash
python generate_like.py [model.pth] [model_config] [noise_schedule] [size_file] -t [tag]
```

Produces ``<tag>_like.hdf5``. Use the same ``-t`` you trained with. ``submit_generate_like.sub``
fans this out the same way, reading ``(config, tag, schedule)`` rows from
``generate_experiments.txt``. ``schedule`` must be the same one the model with that ``tag`` was
trained with.

On Della, ``./submit_generate_like.sh generate_experiments.txt`` does the same on a MIG GPU. Its
file takes the same ``REPO=`` (required), ``MAIL_USER=`` and ``RUNTIME=`` (default ``06:00:00``)
settings, followed by ``config tag schedule`` rows:

```
REPO=/scratch/gpfs/<GROUP>/<netid>/BIBgen
MAIL_USER=<netid>@princeton.edu

# config                      tag       schedule (same as training)
equivariant_denoiser.json     v1        noise_schedule.csv
mlp_denoiser.json             mlp_v1    noise_schedule.csv
```

It reads ``$REPO/examples/training/denoiser_<tag>.pth`` and needs ``test_sizes_large.csv`` in
``$REPO/examples/generation/`` (from ``write_test_sizes.py``; the diffused file has no ``test``
group, so it must come from the matching raw file). To queue generation behind a training job:

```bash
./submit_generate_like.sh generate_experiments.txt --dependency=afterok:<train_jobid>
```

On Perlmutter, ``./submit_generate_like_perlmutter.sh generate_experiments.txt`` is the
equivalent, submitting ``generate_like_perlmutter.slurm`` jobs on the ``shared`` QOS. It takes the
same ``REPO=`` (required), ``ACCOUNT=`` (required, NERSC charge account), ``MAIL_USER=`` and
``RUNTIME=`` (default ``06:00:00``) settings and ``config tag schedule`` rows as the Della version
above.

## Analysis

```bash
python plot_comparison.py [mc_file] [gen_file.hdf5],[name] ...
```

Each generated file is given as a ``path,name`` pair; ``name`` is the legend label used for that
dataset in the overlay, 2D and clustering plots. Pass more than one pair to compare several models
against the same MC on the same axes.

Output directory defaults to ``plots/<tag>``, inferred from the first generated file's name;
``-t/--tag`` overrides it. Whether the energy feature needs exponentiating back from ``ln(E)`` is
auto-detected from the MC file; override with ``-l {auto,yes,no}`` if needed.

``wasserstein_distances.csv`` is written with one row per (generated dataset, variable). Its
``tag`` column is the experiment tag inferred from each generated file's name, so the rows still
join against ``history_<tag>.csv`` when several models are plotted in one run; ``-t/--tag`` sets
that column too when there is exactly one generated file.

## Ablation Analysis

Once you've run `plot_comparison.py` for each config/energy-parameterization variant (producing
a `wasserstein_distances.csv` per tag under `plots/<tag>/`), run:

```bash
python analyze_ablation.py --plots-dir plots --history-dir training/history -o plots/analysis
```

This reads every `wasserstein_distances.csv` found under `--plots-dir` (tag comes from the file's
own `tag` column — no hardcoded list to maintain) and every `history_<tag>.csv` found under
`--history-dir`, then writes four comparison plots to `-o`:

- ``wasserstein_by_variable.png`` — one bar chart per physical variable (energy, φ, η, s, z),
  comparing all configs
- ``wasserstein_grouped.png`` — all variables and configs together, log-scale
- ``pareto_aggregate_vs_valloss.png`` — normalized aggregate Wasserstein score vs. best validation
  loss. **Faceted by energy parameterization (raw-E vs. log-E) when both are present** — validation
  loss is not comparable across the two, since ``ln(E)`` and ``E`` correspond to different
  likelihood functions (a Jacobian term separates them). Wasserstein distance itself doesn't have
  this problem, since it's always computed on the untransformed physical variable.
- ``pareto_pairwise_all_combos.png`` — pairwise Pareto fronts across all five variables; dominance
  here is unaffected by axis scale, so no normalization is needed for this plot specifically (unlike
  the aggregate score above, which does normalize before summing).

Use ``--tag-order`` to fix the tag ordering/subset shown (comma-separated); otherwise tags are
auto-sorted from whatever's present in the data.
