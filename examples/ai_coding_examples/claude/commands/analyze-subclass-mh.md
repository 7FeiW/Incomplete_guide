# Analyze a Chemical Subclass in Negative Mode

Analyze the DAG miss rate for the chemical subclass: **$ARGUMENTS** ([M-H]- negative mode)

This is part of an ongoing investigation of `dag_missing_by_class_m-h.txt` — a file
reporting BFS fragmentation DAG miss rates by ClassyFire subclass (FT instrument, [M-H]-,
min_intensity=0.05, min_spectra=20). Your job is to diagnose WHY the peaks are missing.

This command is a project-specific template. The referenced fragmentation
code and datasets are not included in this guide. Verify the project's current
implementation and data provenance before drawing conclusions. Load pickle
files only from trusted project sources. Treat the categories below as
hypotheses until supported by spectra and code evidence.

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
Check charged functional groups against the generator's supported ionization assumptions.

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

Use the bins to select candidates for investigation. Establish each explanation
from the corresponding molecule, spectrum, and current fragmentation code:

- **Possible halogen-containing fragment:** Check the precursor composition,
  candidate formula, exact mass, and isotope pattern. A fractional m/z bin alone
  is insufficient evidence for a formula or fragmentation mechanism.
- **Possible missing pathway or depth limit:** Trace a proposed neutral-loss or
  cleavage sequence against the generator's allowed operations and depth.
  Several missing peaks do not by themselves establish a cascade-depth failure.
- **Possible model underprediction:** For category B, verify the fragment exists
  in the generated directed acyclic graph (DAG), then compare observed and
  predicted intensity under the same matching and normalization rules.
- **Possible data or coverage issue:** Check missing graph files, precursor
  charge, isotope selection, and isolation metadata before classifying a peak
  as contamination or noise. Confirm how the current code handles missing graphs.

### Formula sanity checks

The following are **calculated nominal mass-to-charge ratios (m/z)** for singly
charged ions using the common isotopes. They are not observed peaks or exact
mass assignments, and are not instructions for rounding a project's m/z bins.

| Candidate formula | Nominal m/z |
| --- | ---: |
| [C₆H₅O]⁻ | 93 |
| [C₆H₅O₂]⁻ | 109 |
| [C₆H₇O₆]⁻ | 175 |
| [H₂PO₄]⁻ | 97 |
| [PO₃]⁻, also the composition after H₂O loss from [H₂PO₄]⁻ | 79 |

These nominal sums use ¹²C, ¹H, ¹⁶O, and ³¹P from the CIAAW isotope tables for
[carbon](https://www.ciaaw.org/carbon.htm),
[hydrogen](https://www.ciaaw.org/hydrogen.htm),
[oxygen](https://www.ciaaw.org/oxygen.htm), and
[phosphorus](https://www.ciaaw.org/phosphorus.htm).
For example, [H₂PO₄]⁻ has nominal m/z 2 + 31 + 4 × 16 = 97;
a peak near 153 cannot be assigned that formula. Exact-mass comparisons must
also account for charge, isotope choice, and the instrument's mass tolerance.

Do not assign a confirmed root cause from an aggregate bin alone. If evidence
is insufficient, record the candidate explanation and the next verification step.

## Step 5 — Verify a key peak assignment

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
Use the measured m/z to check candidate formulas within a documented mass
tolerance. Record competing assignments and verify against the precursor
composition and supporting spectral evidence before selecting a formula.

## Step 6 — Write the findings

Output a section in this format (this will be appended to `docs/missing_peak_findings_m-h.md`
before the Summary section):

```markdown
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

**[Supported root cause, or unresolved hypothesis].**

[2-4 sentences explaining the mechanism. Reference confirmed m/z assignments where possible.
If cascade depth, name the neutral loss sequence and the number of bond cuts required.
If model weakness, name the specific fragment and its DAG intensity vs predicted intensity.]

---
```

Then output a one-line summary table row:
```markdown
| [Subclass] | [n] | [miss%] | [root cause summary] |
```

---

## Root cause taxonomy (use only when supported):

1. **Artifact** — an identified analysis or graph-coverage defect, verified against the current implementation.
2. **Data quality** — contaminated spectra, CE=nan, co-isolated compounds (precursor m/z exceeded)
3. **Cascade depth** — fragment requires >3 sequential bond cuts from [M-H]-
4. **Direct anion** — halide/chalcogenide-retaining anions (.9 mass defect bins)
5. **Rearrangement** — H/methyl migration, retro-Diels-Alder, McLafferty before bond cut
6. **Model weakness** — fragment IS in DAG but model predicts ≈0
7. **Ionization-model limitation** — the verified charge state or fragmentation behavior is outside the generator's supported assumptions.
