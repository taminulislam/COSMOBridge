# COSMOBridge

**Bridging Two-Dimensional Molecular Graphs and Charge-Decorated Molecular Surfaces for Ionic Liquid Thermodynamic Property Prediction**

COSMOBridge is a multimodal architecture that combines Chemprop's directed
message-passing neural network (2D molecular graph) with a PointNet encoder over
a 3D charge-decorated molecular surface, blended through a temperature-conditioned
bilinear fusion and learned **per-property routing gates**. The minimal model (v3)
matches a strong graph baseline on average across 7 thermodynamic properties while
delivering gains on the surface-dependent activity coefficients (γ₁, γ₂). An
extended variant (v4) adds a per-molecule router over three frozen paths.

## What the surface actually is

The 3D input is a 1024-point cloud with 7 features per point: Cartesian
coordinates, surface normal, and one scalar. That scalar is produced as follows
(`scripts/pipeline/step4b_surface_descriptors.py`, `step4c_extract_point_clouds.py`):

1. Conformer generation with RDKit **ETKDGv3**, refined with the **MMFF94** force field.
2. Atomic partial charges assigned by the **Gasteiger** method.
3. A surface built as a superposition of atom-centered Gaussians of width
   0.60 × van der Waals radius, extracted by **marching cubes at density level 0.18**
   on a 0.20 Å grid.
4. The per-point scalar is a Gaussian-weighted average, over the atoms of the
   nearest ionic fragment, of those partial charges. **Units: elementary charge.**

**This scalar is neither an electrostatic potential nor the COSMO screening charge
density σ.** It is a smooth interpolation of the atomic charge distribution onto
the surface. No quantum-chemical or continuum-solvation calculation enters the
representation on which any reported model was trained.

Two things in this repository are easy to mistake for the training route:

- `scripts/pipeline/step3_psi4_cosmo.py` computes DFT surfaces with Psi4
  (B3LYP/def2-SVP + CPCM, ε = 15), and `step4c_extract_point_clouds.py` accepts
  `--prefer-dft` to consume them. **This path was not used for any reported
  result** — none of the 28 reference ionic liquids has a DFT pair surface.
- `step2_geometry_optimization.py` calls GFN2-xTB, but only `singlepoint()`; it
  returns the MMFF94 coordinates together with an xTB energy. **No xTB geometry
  optimization is performed.**

DFT enters the work at exactly one place: an independent check on the twenty
highest-ranked screening candidates (B3LYP/def2-SVP + COSMO, ε = 78.4, NWChem,
on MMFF94 conformers), reported in Supporting Information Table S2.

The virtual screen (`scripts/predict_cosmobridge_v3.py`) builds its surfaces
differently again — points on vdW + 0.42 Å probe spheres carrying a Coulomb sum
Σqⱼ/dⱼ over Gasteiger charges. Training and screening surfaces are related but
not identical quantities; see Section 2.12 of the manuscript.

## Parameter counts

The gate stage trains 7 scalar parameters, but the model as a whole is not a
7-parameter model. Stage 1 trains ≈465K parameters (graph/surface projections,
bilinear fusion, prediction head), Stage 2 trains ≈358K (Chemprop D-MPNN and
readout), and Stage 3 trains the 7 sigmoid gates with Stages 1 and 2 frozen.

## Repository layout

```
src/
  data/         Dataset assembly, graph builder, surface preprocessing
  models/
    graph/      Chemprop D-MPNN, GNN/GAT, atom-surface D-MPNN
    pointcloud/ PointNet, DGCNN surface encoders
    fusion/     COSMOBridge (dual-path gates), GBH bilinear fusion,
                cross-attention, FiLM, MoE, fusion variants
    tabular/    Descriptor baselines
    vision/     ViT (surface-image experiments)
    generative/ Candidate-generation experiments (VAE/CVAE/GAN)
  training/     Trainer, losses, metrics
  utils/        Config / seeding
scripts/
  pipeline/     Surface-generation pipeline
                (geometry → surface descriptors → point clouds;
                 optional Psi4/NWChem DFT, not used for reported models)
  cosmobridge_multiseed.py     v3 multi-seed training/eval
  predict_cosmobridge_v3.py    virtual screening with the v3 model
  rank_ils_desirability.py     Derringer–Suich desirability ranking
  cosmo_sac_baseline.py        simplified COSMO-SAC baseline
  revision/                    statistics for the revised manuscript
                               (b6_cluster_bootstrap.py produces the
                                Bonferroni-corrected p-values of Section 2.11)
  revision_*.py                figure regeneration for the revised manuscript
  ...           architecture-search variants, figure generation
cosmobridge_v4/
  models/ scripts/ configs/    per-molecule routing extension (v4)
configs/        YAML experiment configs
requirements.txt
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Key dependencies: PyTorch ≥ 2.0, PyTorch Geometric, RDKit, Chemprop, scikit-image,
scikit-learn, NumPy/Pandas, Matplotlib. The default surface pipeline needs only
RDKit and scikit-image. Psi4, NWChem and xtb are required **only** for the
optional DFT paths described above.

## Data & model availability

The data behind every result — the 28-ionic-liquid reference dataset with its
19/4/5 split, the 110-candidate desirability ranking, the held-out test-set
predictions of all five seeds, the NWChem COSMO outputs for the twenty
highest-ranked candidates, the trained checkpoints, and the experimental
characterization files — are deposited at:

**https://huggingface.co/datasets/tislamrafi/COSMOBridge** — DOI [10.57967/hf/11424](https://doi.org/10.57967/hf/11424)

This repository holds the source code for that work.

## Reproducing the main result

1. Generate point clouds: `scripts/pipeline/step4b_surface_descriptors.py`,
   `step4c_extract_point_clouds.py` (no DFT required).
2. Train the frozen specialist paths, then the v3 gates:
   `scripts/cosmobridge_multiseed.py`.
3. (Optional) Train the v4 per-molecule router: `cosmobridge_v4/scripts/`.
4. Virtual screen + rank candidates: `scripts/predict_cosmobridge_v3.py`,
   `scripts/rank_ils_desirability.py`.
5. Statistics for the reported comparisons: `scripts/revision/b6_cluster_bootstrap.py`.

## License

MIT — see [LICENSE](LICENSE).
