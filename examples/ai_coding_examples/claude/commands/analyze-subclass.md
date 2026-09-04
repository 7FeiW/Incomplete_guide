Analyze the DAG miss rate for the chemical subclass: **$ARGUMENTS**

This is part of an ongoing investigation of `dag_missing_by_class_m+h.txt` — a file
reporting BFS fragmentation DAG miss rates by ClassyFire subclass (FT instrument, [M+H]+,
min_intensity=0.05, min_spectra=20). Your job is to diagnose WHY the peaks are missing.

---

## Step 1 — Read the subclass entry from the .txt file

Find the section for "$ARGUMENTS" in `dag_missing_by_class_m+h.txt`. Note:
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

Note any halogens (F, Cl, Br, I, S) in SMILES — these are the source of direct-cation gaps.
Note permanent cations (N+, I+ salts) — BFS cannot model these at all.

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

Mols not in tar → their spectra are **skipped** (dag_mzs=None → continue) due to the bug fix.
They do NOT inflate category A counts. But if many spectra come from missing mols, note it.

To check how many spectra come from missing mols:
```python
import pickle
spec_df = pickle.load(open('data/proc/nist23/spec_df.pkl', 'rb'))
for mid in missing:
    n = len(spec_df[(spec_df['mol_id'] == mid) & (spec_df['inst_type'] == 'FT') & (spec_df['prec_type'] == '[M+H]+')])
    if n > 0:
        print(f"mol_id={mid}: {n} FT [M+H]+ spectra (skipped in analysis)")
```

## Step 4 — Classify the top category A m/z bins

Apply these diagnostic rules:

**Rule 1 — Halide mass defect (.9 bins):**
- Any m/z bin with .9 fractional part → Br (offset +0.918), Cl (+0.969), or I (+0.904) direct cation
- These are fragments that RETAIN the halogen and carry the charge: BFS cannot model them
- Root cause: **Direct cation fragmentation (halide-retaining)**

**Rule 2 — Aromatic cation cascade series:**
- C₅–C₁₃ series: 65.0=[C₅H₅]⁺, 67.1=[C₅H₇]⁺, 77.0=[C₆H₅]⁺, 79.1, 91.1=[C₇H₇]⁺ (tropylium),
  93.1, 105.0=[C₈H₉]⁺, 115.1=[C₉H₇]⁺, 128.1=[C₁₀H₈]⁺, 129.1=[C₁₀H₉]⁺, 141.1, 155.1
- If 3+ of these appear together in category A, especially with large ring systems in the mols:
  Root cause: **Cascade depth (≥4 sequential bond cuts + aromatization)**

**Rule 3 — Category B dominance:**
- If category B entries have higher n_spectra or intensity than category A:
  Root cause: **Model weakness** (fragments ARE in DAG but model predicts ≈0)

**Rule 4 — Low n or all mols absent from tar:**
- If category A bins have low n (< 5 spectra) or fractional m/z bins (e.g., 60.7, 53.6):
  These are likely noise bins from spectra of mols not in tar (pre-fix artifact)

**Rule 5 — Precursor m/z exceeded:**
- Any category A peak at m/z HIGHER than the [M+H]⁺ precursor is a data quality artifact
  (co-isolated compound contaminating the MS2 isolation window)

## Step 5 — Verify a key peak assignment (if unclear)

For the highest-intensity category A peak, check actual spectra:
```python
import pickle
import numpy as np
spec_df = pickle.load(open('data/proc/nist23/spec_df.pkl', 'rb'))
df_cf = pickle.load(open('data/classyfire/nist23_mol_classyfire.pkl', 'rb'))
mol_ids = df_cf[df_cf['subklass'] == '$ARGUMENTS']['mol_id'].unique()
spectra = spec_df[(spec_df['mol_id'].isin(mol_ids)) & (spec_df['inst_type'] == 'FT') & (spec_df['prec_type'] == '[M+H]+')]
# Find spectra that have the peak of interest (e.g., near 165.1)
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

Output a section in this format (this will be appended to `docs/missing_peak_findings.md`
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
If the dominant root cause is cascade depth, name the specific ring system and the cascade pathway.
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
3. **Cascade depth** — fragment requires >3 sequential bond cuts from [M+H]+
4. **Direct cation** — halide/chalcogenide-retaining cations (.9 mass defect bins)
5. **Rearrangement** — methyl/H migration before bond cut (e.g., TMS⁺, McLafferty)
6. **Model weakness** — fragment IS in DAG but model predicts ≈0
7. **Permanent cation** — quaternary N⁺ or I⁺ salts; entire ionization model is wrong
