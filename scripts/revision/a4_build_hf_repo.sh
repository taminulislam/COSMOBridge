#!/bin/bash
# A4: assemble the COSMOBridge archive as a Hugging Face dataset repository.
set -euo pipefail
ROOT=/work/nvme/bgte/Dataset_Chemistry
OUT=$ROOT/cosmobridge_hf
rm -rf "$OUT"; mkdir -p "$OUT"/{dataset,predictions,dft/nwchem_outputs,checkpoints,analysis,experimental,paper}

# 1. curated dataset and the 19/4/5 IL split
cp -r "$ROOT"/data/processed/splits/. "$OUT"/dataset/

# 2. frozen predictions
cp "$ROOT"/results/cosmobridge_v3_candidates_sigma.csv   "$OUT"/predictions/candidates_110_frozen_ranking.csv
cp "$ROOT"/results/cosmobridge_c1_official.npz           "$OUT"/predictions/test_set_predictions.npz
cp "$ROOT"/cosmobridge_v4/screening/top20_candidates.csv "$OUT"/predictions/

# 3. DFT: raw NWChem outputs, inputs, geometries
cp "$ROOT"/cosmobridge_v4/screening/nwchem_outputs/*.out "$OUT"/dft/nwchem_outputs/
cp -r "$ROOT"/cosmobridge_v4/screening/nwchem_inputs     "$OUT"/dft/
cp -r "$ROOT"/cosmobridge_v4/screening/geometries        "$OUT"/dft/

# 4. trained models for the reported runs
cp -r "$ROOT"/checkpoints/chemprop "$OUT"/checkpoints/ 2>/dev/null || true
for d in "$ROOT"/checkpoints/cosmobridge*; do [ -d "$d" ] && cp -r "$d" "$OUT"/checkpoints/; done

# 5. analysis scripts behind the revised numbers, with their outputs
cp -r "$ROOT"/scripts/revision "$OUT"/analysis/scripts
for f in b5_chemprop_multiseed b6_cluster_bootstrap c1_official_bootstrap; do
  [ -f "$ROOT/results/$f.json" ] && cp "$ROOT/results/$f.json" "$OUT"/analysis/
done

# 6. experimental characterization (Section 5.2 / SI S2)
cp "$ROOT"/paper/word/COSMOBridge_Alex.docx "$OUT"/experimental/characterization_report.docx
for f in tmg_nmr_dmso tmg_ftir_v2 tmg_13c_aqueous tmg_dsc_subambient tmg_tga_dsc_v2 \
         si_tga_run2_dtg si_tga_run2_lowt si_13c_lineshape si_liquidus gamma_consistency; do
  cp "$ROOT"/paper/cosmobridge_Overleaf/figures/$f.png "$OUT"/experimental/
done

# 7. the manuscript and SI as submitted
cp "$ROOT"/paper/cosmobridge_Overleaf/cosmobridge_jcim.pdf    "$OUT"/paper/manuscript.pdf
cp "$ROOT"/paper/cosmobridge_Overleaf/cosmobridge_jcim_SI.pdf "$OUT"/paper/supporting_information.pdf

cat > "$OUT"/.gitattributes <<'GA'
*.npz   filter=lfs diff=lfs merge=lfs -text
*.pt    filter=lfs diff=lfs merge=lfs -text
*.pkl   filter=lfs diff=lfs merge=lfs -text
*.png   filter=lfs diff=lfs merge=lfs -text
*.pdf   filter=lfs diff=lfs merge=lfs -text
*.docx  filter=lfs diff=lfs merge=lfs -text
*.out   filter=lfs diff=lfs merge=lfs -text
GA

( cd "$OUT" && find . -type f -not -name MANIFEST.txt | sort > MANIFEST.txt )
du -sh "$OUT"; find "$OUT" -type f | wc -l
