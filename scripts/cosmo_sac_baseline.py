"""COSMO-SAC baseline for activity coefficient prediction.

Computes activity coefficients using the COSMO-SAC model directly from
the ESP sigma-profiles extracted from COSMO surface point clouds.

COSMO-SAC (Segment Activity Coefficient) is the standard physics-based
method for predicting activity coefficients from COSMO surfaces.

Reference: Lin & Sandler, Ind. Eng. Chem. Res., 2002.
           Hsieh et al., Fluid Phase Equilibria, 2014 (COSMO-SAC-dsp).

Implementation: Simplified COSMO-SAC using sigma-profiles from our
COSMO surface point clouds (ESP values on the molecular surface).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.training.metrics import compute_metrics, format_metrics

TARGET_COLUMNS = ["gamma1", "gamma2", "G_E", "H_E", "G_mix", "H_vap", "P"]

# COSMO-SAC parameters (from Lin & Sandler, 2002)
R_CAL = 1.987  # cal/(mol·K)
R_KCAL = 1.987e-3  # kcal/(mol·K)
SIGMA_BINS = np.linspace(-0.025, 0.025, 51)  # e/Å², 51 bins
SIGMA_CENTERS = 0.5 * (SIGMA_BINS[:-1] + SIGMA_BINS[1:])  # 50 bin centers
D_SIGMA = SIGMA_BINS[1] - SIGMA_BINS[0]

# COSMO-SAC interaction parameters
ALPHA_PRIME = 16466.72  # kcal·Å⁴/(mol·e²) — misfit energy parameter
R_EFF = 1.07  # Å — effective segment radius
F_DECAY = 1.0  # decay factor for averaging
A_EFF = np.pi * R_EFF**2  # effective segment area


def extract_sigma_profile(pc_path, n_bins=50):
    """Extract sigma-profile from COSMO surface point cloud.

    The sigma-profile p(σ) is the probability distribution of surface
    charge density on the molecular surface.

    Args:
        pc_path: path to .npz point cloud file
        n_bins: number of sigma bins

    Returns:
        sigma_profile: (n_bins,) normalized histogram of ESP values
        surface_area: total surface area estimate
    """
    data = np.load(pc_path)
    points = data["points"]  # (N, 7): xyz + normals + ESP

    esp_values = points[:, 6]  # ESP in e/Å² (COSMO convention)

    # Compute sigma-profile as histogram
    profile, _ = np.histogram(esp_values, bins=SIGMA_BINS, density=True)
    profile = profile * D_SIGMA  # normalize so sum ≈ 1

    # Estimate surface area from point density
    # (rough estimate: convex hull or just use point count as proxy)
    surface_area = len(points) * A_EFF / 10  # rough scaling

    return profile, surface_area


def cosmo_sac_gamma(sigma_profile_1, area_1, sigma_profile_2, area_2,
                     x1, T, n_iter=50):
    """Compute activity coefficients using COSMO-SAC.

    Simplified implementation following Lin & Sandler (2002).

    Args:
        sigma_profile_1: solute sigma-profile (n_bins,)
        area_1: solute surface area
        sigma_profile_2: solvent sigma-profile (n_bins,)
        area_2: solvent surface area
        x1: mole fraction of component 1
        T: temperature in K
        n_iter: iterations for segment activity coefficient convergence

    Returns:
        gamma1, gamma2: activity coefficients
    """
    x2 = 1 - x1
    n_bins = len(sigma_profile_1)

    # Mixture sigma-profile (area-weighted)
    total_area = x1 * area_1 + x2 * area_2
    p_mix = (x1 * area_1 * sigma_profile_1 + x2 * area_2 * sigma_profile_2) / total_area

    # Segment-segment interaction energy (misfit)
    # Delta_W(σ_m, σ_n) = (α'/2) * (σ_m + σ_n)²
    sigma_m = SIGMA_CENTERS[:, None]  # (n, 1)
    sigma_n = SIGMA_CENTERS[None, :]  # (1, n)
    delta_W = (ALPHA_PRIME / 2) * (sigma_m + sigma_n) ** 2  # (n, n) in cal/mol

    # Solve for segment activity coefficients Γ(σ) iteratively
    def solve_gamma_seg(p_sigma, T, n_iter=50):
        """Solve ln(Γ(σ_m)) = -ln(Σ_n p(σ_n) Γ(σ_n) exp(-ΔW(σ_m,σ_n)/RT))"""
        ln_gamma_seg = np.zeros(n_bins)
        RT = R_CAL * T

        for _ in range(n_iter):
            # exp(-ΔW/RT + ln(Γ))
            exponent = -delta_W / RT + ln_gamma_seg[None, :]  # (n_m, n_n)
            # Numerically stable: subtract max
            max_exp = exponent.max(axis=1, keepdims=True)
            sum_term = np.sum(p_sigma[None, :] * np.exp(exponent - max_exp), axis=1)
            ln_gamma_seg_new = -np.log(sum_term + 1e-30) - max_exp.squeeze()
            ln_gamma_seg = ln_gamma_seg_new

        return ln_gamma_seg

    # Segment activity coefficients in mixture and pure components
    ln_Gamma_mix = solve_gamma_seg(p_mix, T, n_iter)
    ln_Gamma_1 = solve_gamma_seg(sigma_profile_1, T, n_iter)
    ln_Gamma_2 = solve_gamma_seg(sigma_profile_2, T, n_iter)

    # Activity coefficient from COSMO-SAC
    # ln(γ_i) = (n_i / A_eff) * Σ_σ p_i(σ) * [ln(Γ_mix(σ)) - ln(Γ_i(σ))]
    n_seg_1 = area_1 / A_EFF
    n_seg_2 = area_2 / A_EFF

    ln_gamma1 = n_seg_1 * np.sum(sigma_profile_1 * (ln_Gamma_mix - ln_Gamma_1))
    ln_gamma2 = n_seg_2 * np.sum(sigma_profile_2 * (ln_Gamma_mix - ln_Gamma_2))

    gamma1 = np.exp(np.clip(ln_gamma1, -20, 20))
    gamma2 = np.exp(np.clip(ln_gamma2, -20, 20))

    return gamma1, gamma2


def main():
    print("=== COSMO-SAC Baseline ===")
    print("Computing activity coefficients from COSMO surface sigma-profiles")
    print(f"Reference: Lin & Sandler, Ind. Eng. Chem. Res., 2002\n")

    pc_dir = Path("data/pipeline/point_clouds")
    idx_df = pd.read_csv(pc_dir / "index.csv")
    pc_index = dict(zip(idx_df["smiles"], idx_df["filename"]))

    # Load test data
    test_df = pd.read_csv("data/processed/splits/test.csv")
    raw_df = pd.read_csv("data/processed/il_data_raw.csv")

    # Get raw test data (we need unnormalized T, x1, targets)
    test_ils = test_df["il_short_name"].unique()
    raw_test = raw_df[raw_df["il_short_name"].isin(test_ils)].reset_index(drop=True)
    print(f"Test ILs: {list(test_ils)}")
    print(f"Test samples: {len(raw_test)}")

    # Water SMILES (component 2 in all mixtures — common solvent)
    # Check what the second component is
    print(f"\nSMILES examples: {raw_test['smiles'].iloc[0]}")

    # Extract sigma-profiles for all ILs
    print("\nExtracting sigma-profiles from point clouds...")
    sigma_profiles = {}
    surface_areas = {}

    for smiles in raw_test["smiles"].unique():
        fn = pc_index.get(smiles)
        if fn and (pc_dir / fn).exists():
            sp, area = extract_sigma_profile(pc_dir / fn)
            sigma_profiles[smiles] = sp
            surface_areas[smiles] = area
            print(f"  {smiles[:40]:40s}: area={area:.1f}, ESP range=["
                  f"{sp.argmax()}, peak_σ={SIGMA_CENTERS[sp.argmax()]:.4f}]")
        else:
            print(f"  {smiles[:40]:40s}: NO POINT CLOUD")

    # For IL+water system, we need water's sigma-profile
    # Since we don't have water's COSMO file, use a standard water sigma-profile
    # Water has a characteristic bimodal sigma-profile
    water_sigma = np.zeros(50)
    # Water: strong negative peak (H-bond donor) and positive peak (H-bond acceptor)
    neg_idx = np.argmin(np.abs(SIGMA_CENTERS - (-0.015)))  # negative peak
    pos_idx = np.argmin(np.abs(SIGMA_CENTERS - (0.015)))   # positive peak
    water_sigma[neg_idx] = 0.3
    water_sigma[pos_idx] = 0.3
    mid_idx = np.argmin(np.abs(SIGMA_CENTERS - 0.0))
    water_sigma[mid_idx] = 0.2
    # Smooth and normalize
    from scipy.ndimage import gaussian_filter1d
    water_sigma = gaussian_filter1d(water_sigma, sigma=2)
    water_sigma = water_sigma / (water_sigma.sum() * D_SIGMA) * D_SIGMA
    water_area = 42.0  # Å² typical water surface area

    print(f"\nWater sigma-profile: sum={water_sigma.sum():.4f}")

    # Compute COSMO-SAC predictions
    print("\nComputing COSMO-SAC activity coefficients...")
    predictions = []
    targets_list = []

    for _, row in raw_test.iterrows():
        smiles = row["smiles"]
        T = row["temperature"]
        x1 = row["x1"]

        if smiles not in sigma_profiles:
            predictions.append([np.nan] * 7)
            targets_list.append([row[t] for t in TARGET_COLUMNS])
            continue

        sp_il = sigma_profiles[smiles]
        area_il = surface_areas[smiles]

        # Compute gamma1 (IL) and gamma2 (water) using COSMO-SAC
        try:
            gamma1_pred, gamma2_pred = cosmo_sac_gamma(
                sp_il, area_il, water_sigma, water_area, x1, T)

            # Derive other properties from gamma values
            # G_E = RT * (x1*ln(g1) + x2*ln(g2))
            x2 = 1 - x1
            G_E_pred = R_KCAL * T * (x1 * np.log(max(gamma1_pred, 1e-10)) +
                                       x2 * np.log(max(gamma2_pred, 1e-10)))

            # H_E ≈ -T² * d(G_E/T)/dT — approximate numerically
            dT = 1.0  # K
            g1_plus, g2_plus = cosmo_sac_gamma(sp_il, area_il, water_sigma, water_area, x1, T + dT)
            G_E_plus = R_KCAL * (T + dT) * (x1 * np.log(max(g1_plus, 1e-10)) +
                                              x2 * np.log(max(g2_plus, 1e-10)))
            H_E_pred = G_E_pred - T * (G_E_plus / (T + dT) - G_E_pred / T) / (1.0 / (T + dT) - 1.0 / T)

            # G_mix = G_E + RT*(x1*ln(x1) + x2*ln(x2))
            G_mix_pred = G_E_pred + R_KCAL * T * (x1 * np.log(max(x1, 1e-10)) +
                                                     x2 * np.log(max(x2, 1e-10)))

            # H_vap and P can't be directly computed from COSMO-SAC binary mixture
            H_vap_pred = np.nan
            P_pred = np.nan

            predictions.append([gamma1_pred, gamma2_pred, G_E_pred, H_E_pred,
                                G_mix_pred, H_vap_pred, P_pred])
        except Exception as e:
            print(f"  Error for {smiles[:30]}: {e}")
            predictions.append([np.nan] * 7)

        targets_list.append([row[t] for t in TARGET_COLUMNS])

    preds = np.array(predictions)
    targets = np.array(targets_list)

    # Compute metrics (in raw space)
    print(f"\n{'='*60}")
    print("COSMO-SAC RESULTS (raw space)")
    print(f"{'='*60}")

    for i, prop in enumerate(TARGET_COLUMNS):
        valid = ~(np.isnan(preds[:, i]) | np.isnan(targets[:, i]))
        if valid.sum() == 0:
            print(f"  {prop:15s}: No valid predictions")
            continue
        p = preds[valid, i]
        t = targets[valid, i]
        mae = np.mean(np.abs(p - t))
        rmse = np.sqrt(np.mean((p - t) ** 2))
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
        print(f"  {prop:15s}: MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  (n={valid.sum()})")

    # Also normalize and compute metrics in normalized space for fair comparison
    import pickle
    with open("data/processed/target_scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)

    # Normalize predictions
    preds_norm = np.full_like(preds, np.nan)
    targets_norm = np.full_like(targets, np.nan)
    for i in range(7):
        valid = ~np.isnan(preds[:, i])
        if valid.any():
            preds_norm[valid, i] = (preds[valid, i] - target_scaler.mean_[i]) / target_scaler.scale_[i]
        targets_norm[:, i] = (targets[:, i] - target_scaler.mean_[i]) / target_scaler.scale_[i]

    metrics = compute_metrics(preds_norm, targets_norm)
    print(f"\n{'='*60}")
    print("COSMO-SAC RESULTS (normalized space, comparable to ML models)")
    print(f"{'='*60}")
    print(format_metrics(metrics, "COSMO-SAC"))

    # Save
    results = {
        "model": "cosmo_sac",
        "reference": "Lin & Sandler, Ind. Eng. Chem. Res., 2002",
        "description": "COSMO-SAC activity coefficients computed from sigma-profiles "
                       "extracted from COSMO surface point clouds. Simplified implementation.",
        "test_metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                         for k, v in metrics.items()},
        "note": "H_vap and P not computable from binary COSMO-SAC",
    }
    with open("results/cosmo_sac_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: results/cosmo_sac_results.json")


if __name__ == "__main__":
    main()
