"""Generate SI Figure S1: 2D structures + COSMO ESP surfaces for representative ILs."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
from PIL import Image

# ILs matching Table S1: (name, SMILES, split, EP_image_path)
ILS = [
    # Train set
    ("AMIMCl", "C=CC[n+]1ccn(C)c1.[Cl-]", "Train",
     "Image/Electrostatic Potential/AMIMCL_EP.png"),
    ("BMIMCl", "CCCCn1cc[n+](C)c1.[Cl-]", "Train",
     "Image/Electrostatic Potential/BMIMCl .png"),
    ("EMIMHSO4", "CCn1cc[n+](C)c1.OS(=O)(=O)[O-]", "Train",
     "Image/Electrostatic Potential/EMIMHSO4 .png"),
    ("ChOAc", "C[N+](C)(C)CCO.CC(=O)[O-]", "Train",
     "Image/Electrostatic Potential/CHOAC_EP.png"),
    ("TEALAC", "CC[NH+](CC)CC.CC(O)C(=O)[O-]", "Train",
     "Image/Electrostatic Potential/TEALAC_EP.png"),
    # Test set
    ("BMIMOAc", "CCCCn1cc[n+](C)c1.CC(=O)[O-]", "Test",
     "Image/Electrostatic Potential/BMIMOAc_EP.png"),
    ("ChCl", "C[N+](C)(C)CCO.[Cl-]", "Test",
     "Image/Electrostatic Potential/CHCL_EP.png"),
    ("EMIMBr", "CCn1cc[n+](C)c1.[Br-]", "Test",
     "Image/Electrostatic Potential/EMIMBr_EP.png"),
    ("EMIMOAc", "CCn1cc[n+](C)c1.CC(=O)[O-]", "Test",
     "Image/Electrostatic Potential/EMIMOAc_EP.png"),
    ("ChLys", "C[N+](C)(C)CCO.NCCCC(N)C(=O)[O-]", "Test",
     "Image/Electrostatic Potential/CHLYS_EP.png"),
]

# Full chemical names for labels
FULL_NAMES = {
    "AMIMCl": "Allylmethylimidazolium Chloride",
    "BMIMCl": "Butylmethylimidazolium Chloride",
    "EMIMHSO4": "Ethylmethylimidazolium\nHydrogen Sulfate",
    "ChOAc": "Cholinium Acetate",
    "TEALAC": "Triethylammonium Lactate",
    "BMIMOAc": "Butylmethylimidazolium Acetate",
    "ChCl": "Cholinium Chloride",
    "EMIMBr": "Ethylmethylimidazolium Bromide",
    "EMIMOAc": "Ethylmethylimidazolium Acetate",
    "ChLys": "Cholinium Lysinate",
}


def smiles_to_image(smiles, size=(450, 300)):
    """Convert SMILES to 2D structure image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Image.new('RGB', size, 'white')
    drawer = Draw.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.5
    opts.minFontSize = 16
    opts.additionalAtomLabelPadding = 0.15
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    bio = BytesIO(drawer.GetDrawingText())
    return Image.open(bio)


def make_panel(ils_subset, output_path, panel_title, color_key):
    """Generate one panel (train or test) with given ILs."""
    n = len(ils_subset)

    fig = plt.figure(figsize=(16, 3.2 * n + 1.2))
    gs = GridSpec(n + 1, 2, width_ratios=[1, 1],
                  hspace=0.15, wspace=0.05,
                  left=0.03, right=0.97, top=0.94, bottom=0.04,
                  height_ratios=[1]*n + [0.08])

    for i, (name, smiles, split, ep_path) in enumerate(ils_subset):
        color = color_key
        full_name = FULL_NAMES[name]

        # Left: 2D structure
        ax_2d = fig.add_subplot(gs[i, 0])
        img_2d = smiles_to_image(smiles, size=(500, 330))
        ax_2d.imshow(img_2d)
        ax_2d.axis('off')
        ax_2d.set_title(f"{name}\n{full_name}", fontsize=14, fontweight='bold',
                       color=color, pad=6, linespacing=1.2)

        # Right: COSMO EP surface
        ax_ep = fig.add_subplot(gs[i, 1])
        try:
            img_ep = Image.open(ep_path)
            img_arr = np.array(img_ep)
            if img_arr.ndim == 3:
                mask = img_arr.min(axis=2) < 240
            else:
                mask = img_arr < 240
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                pad = 10
                rmin = max(0, rmin - pad)
                rmax = min(img_arr.shape[0], rmax + pad)
                cmin = max(0, cmin - pad)
                cmax = min(img_arr.shape[1], cmax + pad)
                img_arr = img_arr[rmin:rmax, cmin:cmax]
            ax_ep.imshow(img_arr)
        except Exception as e:
            ax_ep.text(0.5, 0.5, f"Image not found",
                      ha='center', va='center', fontsize=10,
                      transform=ax_ep.transAxes)
        ax_ep.axis('off')
        ax_ep.set_title(f"COSMO Electrostatic Surface",
                       fontsize=14, fontweight='bold', color=color, pad=6)

    # Column headers
    fig.text(0.27, 0.98, '2D Molecular Structure', fontsize=18, fontweight='bold',
            ha='center', va='top', transform=fig.transFigure)
    fig.text(0.73, 0.98, 'COSMO Electrostatic Potential Surface', fontsize=18,
            fontweight='bold', ha='center', va='top', transform=fig.transFigure)

    # Panel title
    fig.text(0.5, 0.995, panel_title, fontsize=20, fontweight='bold',
            ha='center', va='top', transform=fig.transFigure, color=color)

    # Legend at bottom
    ax_legend = fig.add_subplot(gs[n, :])
    ax_legend.axis('off')
    ax_legend.text(0.5, 0.5,
                  'Red = Positive ESP (H-bond donor / electrophilic)    |    '
                  'Blue = Negative ESP (H-bond acceptor / nucleophilic)',
                  fontsize=13, ha='center', va='center',
                  transform=ax_legend.transAxes,
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0',
                           edgecolor='#cccccc'))

    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    train_ils = [il for il in ILS if il[2] == "Train"]
    test_ils = [il for il in ILS if il[2] == "Test"]

    make_panel(train_ils,
               'paper/jcim/figures/si_structure_esp_train.png',
               'Figure S1a: Training Set (5 of 19 ILs)',
               '#1a5276')

    make_panel(test_ils,
               'paper/jcim/figures/si_structure_esp_test.png',
               'Figure S1b: Test Set (all 5 ILs)',
               '#922b21')


if __name__ == "__main__":
    main()
