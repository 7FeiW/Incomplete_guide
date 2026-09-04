Analyze the DAG miss rate for the chemical subclass: **$ARGUMENTS** ([M-H]- negative mode)

This is part of an ongoing investigation of `dag_missing_by_class_m-h.txt` — a file
reporting BFS fragmentation DAG miss rates by ClassyFire subclass (FT instrument, [M-H]-,
min_intensity=0.05, min_spectra=20). Your job is to diagnose WHY the peaks are missing.

---

## Step 1 — Read the subclass entry from the .txt file

Find the section for "$ARGUMENTS" in `dag_missing_by_class_m-h.txt`. Note:
- n= (total spectra)
- DAG miss rate %
- Category A peaks (absent from DAG entirely): m/z bins, n_spectra, mean intensities
- Category B peaks (in DAG but underpredicted): m/z bins, n_spectra, mean pred intensities

## Step 2 — Look up mol_ids in ClassyFire data

```python
import pickle
df = pickle.load(open('data/classyfire/nist23_mol_classyfire.pkl', 'rb'))
subclass_mols = df[df['subklass'] == '$ARGUMENTS'][['mol_id', 'smiles', 'subklass']].drop_duplicates('mol_id')
print(f"{len(subclass_mols)} mols found")
print(subclass_mols.to_string())
```

Note any halogens (F, Cl, Br, I) and acidic groups (COOH, SO3H, PO4, phenol OH) in SMILES.
Note permanent anions (sulfonate, phosphonate zwitterions) — BFS cannot model these.

## Step 3 — Check tar coverage

```python
import tarfile, pickle
df = pickle.load(open('data/classyfire/nist23_mol_classyfire.pkl', 'rb'))
subclass_mol_ids = set(df[df['subklass'] == '$ARGUMENTS']['mol_id'].unique())
with tarfile.open('data/frag/nist23_ma_mi_d3_h4_isoFalse.tar', 'r') as tar:
    in_tar = {int(m.name.split('/')[-1].replace('.pkl.bz2','')) for m in tar.getmembers() if '.pkl.bz2' in m.name}
missing = subclass_mol_ids - in_tar
print(f"In tar: {len(subclass_mol_ids - missing)}/{len(subclass_mol_ids)}")
print(f"Missing from tar: {missing}")
```

To check how many spectra come from missing mols:
```python
import pickle
spec_df = pickle.load(open('data/proc/nist23/spec_df.pkl', 'rb'))
for mid in missing:
    n = len(spec_df[(spec_df['mol_id'] == mid) & (spec_df['inst_type'] == 'FT') & (spec_df['prec_type'] == '[M-H]-')])
    if n > 0:
        print(f"mol_id={mid}: {n} FT [M-H]- spectra (skipped in analysis)")
```

## Step 4 — Classify the top category A m/z bins

Apply these diagnostic rules (negative mode):

**Rule 1 — Halide/chalcogenide mass defect (.9 bins):**
- m/z bins with .9 fractional part → Cl⁻ (34.969), Br⁻ (78.918), or halide-retaining anion fragments
- Root cause: **Direct anion fragmentation (halide-retaining)**

**Rule 2 — Neutral loss series (negative mode):**
- CO₂ loss: [M−H−44]⁻ → carboxylic acids, acyl glucuronides
- SO₃ loss: [M−H−80]⁻ → sulfates, sulfonates
- H₂O loss: [M−H−18]⁻ → hydroxyl-bearing acids
- HF loss: [M−H−20]⁻ → fluorinated compounds
- If these appear as prominent category A peaks at [M−H−Δ]⁻, root cause: **Cascade depth** (neutral loss
  followed by further ring cleavage requires >3 bond cuts total)

**Rule 3 — Carboxylate/phenolate anion series:**
- Fatty acid carboxylates [CₙH₂ₙ₋₁O₂]⁻: e.g., 255.2 (palmitate), 281.2 (oleate), 283.2 (stearate)
- Phenolate [C₆H₅O]⁻: 93.0; chlorophenolate: 127.0, 128.9
- Deprotonated aromatics: 113.0=[C₆H₅O₂]⁻ (catecholate), 179.0=[C₆H₇O₆]⁻
- These are direct one-cut fragments: if absent from DAG, check if H-transfer range covers them;
  if in DAG but model ≈0, root cause: **Model weakness**

**Rule 4 — Category B dominance:**
- If category B entries have higher n_spectra or intensity than category A:
  Root cause: **Model weakness** (fragments ARE in DAG but model predicts ≈0)

**Rule 5 — Low n or fractional m/z bins:**
- If category A bins have low n (< 5 spectra) or fractional m/z bins (e.g., 60.7, 53.6):
  These are noise bins from spectra of mols not in tar (pre-fix artifact)

**Rule 6 — Precursor m/z exceeded:**
- Any category A peak at m/z HIGHER than [M−H]⁻ precursor is a data quality artifact
  (co-isolated contamination)

**Rule 7 — Phosphate/sulfate head group series (lipids):**
- Glycerophospholipids in neg mode: 152.9=[H₂PO₄]⁻, 78.9=[PO₃]⁻, 96.9=[H₂PO₄−H₂O]⁻,
  255.2/281.2/283.2 (acyl chains), 153.0 (glycerophosphate)
- If several of these absent together: root cause is **Cascade depth** (head group requires
  multi-step cleavage from intact glycerophospholipid)

## Step 5 — Verify a key peak assignment (if unclear)

For the highest-intensity category A peak, check actual spectra:
```python
import pickle
import numpy as np
spec_df = pickle.load(open('data/proc/nist23/spec_df.pkl', 'rb'))
df_cf = pickle.load(open('data/classyfire/nist23_mol_classyfire.pkl', 'rb'))
mol_ids = df_cf[df_cf['subklass'] == '$ARGUMENTS']['mol_id'].unique()
spectra = spec_df[(spec_df['mol_id'].isin(mol_ids)) & (spec_df['inst_type'] == 'FT') & (spec_df['prec_type'] == '[M-H]-')]
target_mz = 165.1  # Replace with the actual m/z bin of interest
for _, row in spectra.head(5).iterrows():
    mzs = np.array(row['mzs'])
    ints = np.array(row['intensities'])
    near = np.abs(mzs - target_mz) < 0.15
    if near.any():
        print(f"mol_id={row['mol_id']}, CE={row.get('ce', '?')}: m/z={mzs[near]}, int={ints[near]}")
```
The exact m/z value reveals the elemental formula (mass defect fingerprint).

## Step 6 — Write the findings

Output a section in this format (this will be appended to `docs/missing_peak_findings_m-h.md`
before the Summary section):

```
## [Subclass Name] (n=XXX, DAG miss rate=XX.X%)

**N mols; M in tar.** [One sentence on what these molecules are structurally.]

### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| ...     | ...    | ...       | ...  |

### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| ...     | ...    | ...            |

### Root cause

**[Primary root cause].**

[2-4 sentences explaining the mechanism. Reference confirmed m/z assignments where possible.
If cascade depth, name the neutral loss sequence and the number of bond cuts required.
If model weakness, name the specific fragment and its DAG intensity vs predicted intensity.]

---
```

Then output a one-line summary table row:
```
| [Subclass] | [n] | [miss%] | [root cause summary] |
```

---

## Root cause taxonomy (for your final classification):

1. **Artifact** — mols not in tar inflated category A (pre-fix bug; now moot for current analysis)
2. **Data quality** — contaminated spectra, CE=nan, co-isolated compounds (precursor m/z exceeded)
3. **Cascade depth** — fragment requires >3 sequential bond cuts from [M-H]-
4. **Direct anion** — halide/chalcogenide-retaining anions (.9 mass defect bins)
5. **Rearrangement** — H/methyl migration, retro-Diels-Alder, McLafferty before bond cut
6. **Model weakness** — fragment IS in DAG but model predicts ≈0
7. **Permanent anion** — sulfonate/phosphonate zwitterions; entire ionization model is wrong
