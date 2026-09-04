Add a new SMARTS fragmentation rule for: **$ARGUMENTS**

If `$ARGUMENTS` starts with `nl:` (e.g. `nl:c2h2_loss`), follow the **NL Rule workflow** below.
Otherwise follow the **FRAG Rule workflow** for a `SmartsFragRule` (ring-cut scaffold rules).

---

# NL RULE WORKFLOW (`nl:` prefix)

Use this path for `NeutralLossRule` entries in `NL_RULES` — rearrangements that expel a
neutral fragment and form a new bond (SO₂ extrusion, CO loss, C₂H₂ loss, etc.).

**`NeutralLossRule` dataclass fields:**
```python
NeutralLossRule(
    name="...",              # snake_case, unique
    reactant_smarts="...",   # SMARTS pattern; all mapped atoms 1-based
    nl_map_nums=[1, 2, 3],  # atom map numbers that leave as the neutral loss
    nl_mass=XX.XXXX,        # exact monoisotopic mass of the neutral fragment (Da)
    min_num_atoms=0,         # skip rule if molecule has fewer heavy atoms
)
```

## NL Step 1 — Identify the reaction and neutral mass

Name the reaction mechanism (e.g. "aryl sulfonate SO₂ loss", "PAH C₂H₂ extrusion").
Compute the exact monoisotopic mass of the neutral fragment:

| Neutral | Formula | Exact mass |
|---------|---------|-----------|
| SO₂ | S O₂ | 63.9619 |
| SO₃ | S O₃ | 79.9568 |
| CO | C O | 27.9949 |
| CO₂ | C O₂ | 43.9898 |
| C₂H₂ | C₂ H₂ | 26.0157 |
| N₂O | N₂ O | 44.0010 |
| S=O (sulfoxide) | S O | 47.9669 |
| H₂O | H₂ O | 18.0106 |

Cross-check: search NIST23 or confirmed_records for the expected [M-neutral]± peak.

## NL Step 2 — Identify the functional group SMARTS

Write a SMARTS pattern matching the substructure that undergoes the reaction.
Mark the **leaving atoms** with map numbers — these become `nl_map_nums`.
Unmapped atoms in the SMARTS are anchor atoms (stay in the product).

Key constraints:
- Map numbers are 1-based (map 0 is silently dropped by RDKit)
- Use `[c:1]` for aromatic C, `[C:1]` for aliphatic, never `[cC:1]`
- Use `[SX4:2]` for sulfone/sulfonate S (4-valent), `[SX3:2]` for sulfoxide (3-valent)
- `+0` charge constraint if needed to exclude charged atoms
- The pattern must not match the neutral-loss atoms themselves as anchors

## NL Step 3 — Validate the SMARTS

```python
import sys; sys.path.insert(0, "src")
from rdkit import Chem

smarts = "YOUR_SMARTS_HERE"
patt = Chem.MolFromSmarts(smarts)
assert patt is not None, "SMARTS failed to compile!"

# Positive controls — should match
for smi in ["TARGET_SMILES_1", "TARGET_SMILES_2"]:
    mol = Chem.MolFromSmiles(smi)
    assert mol.HasSubstructMatch(patt), f"No match on {smi}"
    print(f"Matches on {smi}: {len(mol.GetSubstructMatches(patt))}")

# Negative controls — should NOT match
for neg_smi in ["c1ccccc1", "c1ccc2ccccc2c1"]:
    mol = Chem.MolFromSmiles(neg_smi)
    assert not mol.HasSubstructMatch(patt), f"False positive on {neg_smi}"

# Show which atoms are the leaving group
match = mol.GetSubstructMatches(patt)[0]
q_idx_to_map = {i: patt.GetAtomWithIdx(i).GetAtomMapNum()
                for i in range(patt.GetNumAtoms())
                if patt.GetAtomWithIdx(i).GetAtomMapNum() > 0}
map_to_mol = {m: match[q] for q, m in q_idx_to_map.items()}
nl_map_nums = [1, 2, 3]  # your nl_map_nums
nl_atoms = [map_to_mol[m] for m in nl_map_nums]
print("Leaving atom indices:", nl_atoms)
print("Leaving atoms:", [mol.GetAtomWithIdx(i).GetSymbol() for i in nl_atoms])
```

## NL Step 4 — Verify end-to-end via `_apply_nl_prepass`

```python
from fragnnet.frag.smarts_prepass import NL_RULES, _apply_nl_prepass
import numpy as np

# After adding the rule to NL_RULES, test it on a representative molecule
mol = Chem.MolFromSmiles("TARGET_SMILES")
num_nodes = mol.GetNumAtoms()
results = _apply_nl_prepass(NL_RULES, mol, num_nodes)
print(f"NL prepass produced {len(results)} product masks")
for prod_mask, rule_idx in results:
    retained = [mol.GetAtomWithIdx(i).GetSymbol() for i, v in enumerate(prod_mask) if v]
    lost = [mol.GetAtomWithIdx(i).GetSymbol() for i, v in enumerate(prod_mask) if not v]
    print(f"  rule={NL_RULES[rule_idx].name}  retained={retained}  lost={lost}")
```

## NL Step 5 — Add the rule to `smarts_prepass.py`

Append to `NL_RULES` in `src/fragnnet/frag/smarts_prepass.py`:

```python
# [Short description of the reaction, 1-2 sentences]
# [Include: which bond breaks, what neutral leaves, what new bond forms if any]
# [Confirmed in: spec_id or evidence]
NeutralLossRule(
    name="rule_name",
    reactant_smarts="YOUR_SMARTS",
    nl_map_nums=[1, 2, 3],
    nl_mass=XX.XXXX,
    min_num_atoms=N,
),
```

## NL Step 6 — Write tests

Add a new test class to `tests/test_nl_prepass.py` (create if it doesn't exist).
Minimum 6 tests:

```python
from fragnnet.frag.smarts_prepass import NL_RULES, _apply_nl_prepass
from rdkit import Chem
import numpy as np

class TestRuleNameNL:
    TARGET_1 = "SMILES_WITH_FUNCTIONAL_GROUP"
    TARGET_2 = "SMILES_VARIANT"   # different substituents, still matches
    NEG_1    = "SMILES_WITHOUT_FUNCTIONAL_GROUP"

    def test_rule_exists_in_nl_rules(self):
        assert any(r.name == "rule_name" for r in NL_RULES)

    def test_smarts_compiles(self):
        rule = next(r for r in NL_RULES if r.name == "rule_name")
        assert Chem.MolFromSmarts(rule.reactant_smarts) is not None

    def test_matches_target(self):
        rule = next(r for r in NL_RULES if r.name == "rule_name")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.TARGET_1)
        assert mol.HasSubstructMatch(patt)

    def test_no_match_negative(self):
        rule = next(r for r in NL_RULES if r.name == "rule_name")
        patt = Chem.MolFromSmarts(rule.reactant_smarts)
        mol = Chem.MolFromSmiles(self.NEG_1)
        assert not mol.HasSubstructMatch(patt)

    def test_product_mask_loses_correct_atoms(self):
        mol = Chem.MolFromSmiles(self.TARGET_1)
        results = _apply_nl_prepass(NL_RULES, mol, mol.GetNumAtoms())
        rule_results = [(m, ri) for m, ri in results if NL_RULES[ri].name == "rule_name"]
        assert len(rule_results) >= 1
        prod_mask, _ = rule_results[0]
        # Verify the product has the correct heavy atom count
        n_retained = int(prod_mask.sum())
        assert n_retained == mol.GetNumAtoms() - N_LEAVING_ATOMS

    def test_product_mass_matches_nl(self):
        # Compute the mass of retained atoms and verify it equals [M - nl_mass]
        mol = Chem.MolFromSmiles(self.TARGET_1)
        # ... (use Descriptors.ExactMolWt on sub-molecule)
        ...
```

Run: `conda run -n <project>-gpu pytest tests/test_nl_prepass.py -v`

## NL Step 7 — Update memory

Update `MEMORY.md` and add `memory/nl_rule_<name>.md`.

---

# FRAG RULE WORKFLOW (no prefix — ring-cut scaffold rules)

This is the original path for `SmartsFragRule` entries in `FRAG_RULES`.
Use this for multi-ring-cut fragments that baseline BFS cannot reach within max_depth=3.

---

## Context

**Key files:**
- `src/fragnnet/frag/smarts_prepass.py` — `FRAG_RULES` list, add new `SmartsFragRule` here
- `tests/test_smarts_prepass_inject.py` — add tests here
- `scripts/debug/debug_compare_mbfs_dags.py` — `_get_cc_mask_rows()` helper for DAG traces

**`SmartsFragRule` dataclass fields:**
```python
SmartsFragRule(
    name="...",                         # snake_case, unique
    reactant_smarts="...",              # SMARTS with :N map numbers (1-based; map 0 silently ignored)
    cut_bond_map_pairs=[(a,b), ...],    # list of (map_a, map_b) pairs defining bonds to cut
    merge_outer_map_nums=[x, y],        # atoms identifying "outer" (small, discarded) CCs
    filter_bond_map_nums=[],            # optional: skip if map pair is a direct bond (4-ring guard)
    min_num_atoms=0,                    # skip rule if molecule has fewer heavy atoms than this
)
```

**Critical SMARTS gotcha:** `[OC]` = "atom that is O AND C simultaneously" = impossible = 0 matches.
Never use element AND-combinations in brackets. Use `[O]`, `[c]`, `[C]` etc. separately.

**Map number 0 is silently dropped** in RDKit SMARTS matching — always start map numbers at 1.

**How the prepass works:** injects fragment pairs at depth 2 off the root; deduplicates by
byte-key of the atom mask. The rule fires on ALL molecules matching the SMARTS, so
`min_num_atoms` is the primary guard against false positives on tiny molecules.

---

## Step 1 — Characterize the scaffold (if $ARGUMENTS is a name, not SMILES)

If $ARGUMENTS is a scaffold name (e.g. "xanthone", "morphinan"), look up its Murcko SMILES:

```python
import sys; sys.path.insert(0, "src")
import pandas as pd
mol_df = pd.read_pickle("data/proc/nist20/mol_df.pkl")
# Search by scaffold keyword
hits = mol_df[mol_df["scaffold"].str.contains("$ARGUMENTS", case=False, na=False)]["scaffold"].value_counts()
print(hits.head(10))
```

Identify:
- Scaffold SMILES (Murcko)
- Total NIST20 molecule count
- Whether the scaffold has a known confirmed-record topology (see memory)

---

## Step 2 — Confirm there are useful mbfs-only fragments

Pre-computed baseline and mbfs DAGs for all NIST20 molecules are stored as
`data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags/{mol_id}.pkl.bz2` and
`data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_mbfs/dags/{mol_id}.pkl.bz2`.
Load them directly — no recomputation needed.

```python
import sys; sys.path.insert(0, "src"); sys.path.insert(0, "scripts/debug")
import bz2, pickle, collections
import pandas as pd
from pathlib import Path
from rdkit import Chem
import debug_compare_mbfs_dags as dbg

mol_df = pd.read_pickle("data/proc/nist20/mol_df.pkl").set_index("mol_id")
SCAFFOLD = "PASTE_SCAFFOLD_SMILES"
BASE_DP = Path("data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags")
MBFS_DP = Path("data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse_mbfs/dags")

def load_dag(dp, mol_id):
    p = dp / f"{mol_id}.pkl.bz2"
    if not p.exists():
        return None
    with bz2.open(p, "rb") as f:
        return pickle.load(f)

def get_cuts(mol, mask):
    fa = {i for i, v in enumerate(mask) if v}
    return [(a, b) for bond in mol.GetBonds()
            for a, b in [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())]
            if (a in fa) != (b in fa)]

def elem_formula(mol, mask):
    ec = {}
    for i, v in enumerate(mask):
        if v:
            s = mol.GetAtomWithIdx(i).GetSymbol()
            ec[s] = ec.get(s, 0) + 1
    return ec

# Scan all scaffold molecules; aggregate cut-topology signatures for 4-ring-cut-only new masks
mol_ids = mol_df[mol_df["scaffold"] == SCAFFOLD].index.tolist()
print(f"Scaffold molecules: {len(mol_ids)}")

topology_counts = collections.Counter()  # signature → count of (mol_id, mask) pairs
topology_examples = {}                   # signature → (mol_id, mask, cuts)

for mol_id in mol_ids:
    base_dag = load_dag(BASE_DP, mol_id)
    mbfs_dag = load_dag(MBFS_DP, mol_id)
    if base_dag is None or mbfs_dag is None:
        continue
    smi = mol_df.loc[mol_id, "smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue
    base_masks = set(dbg._get_cc_mask_rows(base_dag))
    for mask in dbg._get_cc_mask_rows(mbfs_dag):
        if mask in base_masks:
            continue
        cuts = get_cuts(mol, mask)
        ring_cuts = [c for c in cuts if mol.GetBondBetweenAtoms(*c).IsInRing()]
        if len(ring_cuts) != 4 or len(cuts) != 4:
            continue  # want exactly 4 ring cuts, 0 exocyclic
        # Build a topology signature: frozenset of (sym_a, sym_b, n_shared_rings) per cut
        ri = mol.GetRingInfo()
        sig = frozenset(
            (mol.GetAtomWithIdx(a).GetSymbol(), mol.GetAtomWithIdx(b).GetSymbol(),
             len([r for r in ri.AtomRings() if a in r and b in r]))
            for a, b in ring_cuts
        )
        topology_counts[sig] += 1
        if sig not in topology_examples:
            topology_examples[sig] = (mol_id, mask, ring_cuts)

print("\nTop cut-topology signatures across scaffold molecules:")
for sig, cnt in topology_counts.most_common(10):
    mid, mask, cuts = topology_examples[sig]
    smi = mol_df.loc[mid, "smiles"]
    mol = Chem.MolFromSmiles(smi)
    ec = elem_formula(mol, mask)
    fa = [i for i, v in enumerate(mask) if v]
    print(f"\n  count={cnt}  formula={ec}  sig={sig}")
    print(f"  example mol_id={mid}: {smi}")
    print(f"  fragment atoms: {fa}")
    ri = mol.GetRingInfo()
    for a, b in sorted(cuts):
        rings = [i for i, r in enumerate(ri.AtomRings()) if a in r and b in r]
        print(f"    CUT {a}({mol.GetAtomWithIdx(a).GetSymbol()})-"
              f"{b}({mol.GetAtomWithIdx(b).GetSymbol()}) rings={rings}")
```

Pick the **highest-count signature** with a chemically interpretable fragment.
If no 4-ring-only-cut new masks exist across any scaffold molecule, this scaffold
cannot be covered by a SMARTS prepass rule.

---

## Step 3 — Map the SMARTS atom numbering

Print the ring system of the representative molecule to assign SMARTS map numbers:

```python
print("Atoms:")
for a in mol.GetAtoms():
    in_rings = [i for i,r in enumerate(ri.AtomRings()) if a.GetIdx() in r]
    print(f"  {a.GetIdx()}: {a.GetSymbol()} arom={a.GetIsAromatic()} H={a.GetTotalNumHs()} rings={in_rings}")

print("\nRings:")
for i, ring in enumerate(ri.AtomRings()):
    syms = [(idx, mol.GetAtomWithIdx(idx).GetSymbol()) for idx in ring]
    print(f"  Ring {i}: {syms}")

print("\nRing bonds:")
for b in mol.GetBonds():
    if b.IsInRing():
        a, c = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        rings = [i for i,r in enumerate(ri.AtomRings()) if a in r and c in r]
        print(f"  {a}({mol.GetAtomWithIdx(a).GetSymbol()})-{c}({mol.GetAtomWithIdx(c).GetSymbol()}) rings={rings}")
```

Assign SMARTS map numbers (:1, :2, ...) to ring atoms, choosing numbers so that:
- The 4 cut bonds can be expressed as `cut_bond_map_pairs=[(a,b),(c,d),(e,f),(g,h)]`
- The "outer" (discarded) CCs contain atoms with map numbers in `merge_outer_map_nums`
- The "inner" (kept) fragment is the remaining CC — this is what gets injected into the DAG

Draw a map number → atom assignment table before writing the SMARTS string.

---

## Step 4 — Write and validate the SMARTS

Draft the SMARTS, then validate it matches the target scaffold and NOT unrelated molecules:

```python
from rdkit import Chem

smarts = "YOUR_SMARTS_HERE"
patt = Chem.MolFromSmarts(smarts)
assert patt is not None, "SMARTS failed to compile!"

# Should match the target molecule
target = Chem.MolFromSmiles("TARGET_SMILES")
assert target.HasSubstructMatch(patt), "SMARTS does not match target!"
print(f"Matches on target: {len(target.GetSubstructMatches(patt))}")

# Should NOT match unrelated molecules
for neg_smi in ["c1ccccc1", "c1ccc2ccccc2c1"]:
    neg = Chem.MolFromSmiles(neg_smi)
    assert not neg.HasSubstructMatch(patt), f"False positive on {neg_smi}"

# Show match atom indices for map numbers
match = target.GetSubstructMatches(patt)[0]
atom_map = {patt.GetAtomWithIdx(i).GetAtomMapNum(): target_idx
            for i, target_idx in enumerate(match)
            if patt.GetAtomWithIdx(i).GetAtomMapNum() > 0}
print("Map num → atom index:", atom_map)
```

Check that the cut bond pairs resolve to actual ring bonds in the target molecule.

---

## Step 5 — Verify the rule fires correctly end-to-end

```python
from fragnnet.frag.smarts_prepass import FRAG_RULES, _apply_smarts_prepass
from fragnnet.utils.frag_utils import _inject_smarts_pairs_into_dag

# Check rule appears in FRAG_RULES after adding it
rule_names = [r.name for r in FRAG_RULES]
assert "YOUR_RULE_NAME" in rule_names

# Run prepass on target mol and verify node count increases vs no-prepass
from fragnnet.utils import frag_utils
mol_id = "test"
_, dag_no_pp = frag_utils.timed_get_dags(mol, mol_id, multi_cut_bfs=False,
    max_cut_size=2, smarts_prepass=False, **GEN_KW)
_, dag_pp = frag_utils.timed_get_dags(mol, mol_id, multi_cut_bfs=False,
    max_cut_size=2, smarts_prepass=True, **GEN_KW)

print(f"Nodes without prepass: {dag_no_pp['dag_num_nodes']}")
print(f"Nodes with prepass:    {dag_pp['dag_num_nodes']}")
assert dag_pp["dag_num_nodes"] > dag_no_pp["dag_num_nodes"], "Rule added no new nodes!"
```

---

## Step 6 — Add the rule to `smarts_prepass.py`

Edit `src/fragnnet/frag/smarts_prepass.py`, appending to `FRAG_RULES` after the last existing rule:

```python
# [scaffold_name] ring cleavage
# [1-2 sentence description of what the scaffold is and what the 4 cuts do]
# [Note which fragment is "kept" and which are "outer"]
# [Include: "min_num_atoms=N (bare scaffold has N heavy atoms)"]
SmartsFragRule(
    name="scaffold_name_ring_cleavage",
    reactant_smarts=(
        "[a+0:1]..."  # break across lines at 100 chars max
    ),
    cut_bond_map_pairs=[(a, b), (c, d), (e, f), (g, h)],
    merge_outer_map_nums=[x, y],
    min_num_atoms=N,
),
```

---

## Step 7 — Write tests

Add a new test class to `tests/test_smarts_prepass_inject.py`. Minimum 8 tests:

```python
class TestScaffoldNameRingCleavage:
    SCAFFOLD_MOL_1 = "SMILES_OF_REPRESENTATIVE_TARGET"
    SCAFFOLD_MOL_2 = "SMILES_OF_SECOND_TARGET"  # different substituents
    NEG_MOL_1 = "NAPHTHALENE_OR_SIMILAR"  # should NOT match
    NEG_MOL_2 = "BENZENE"

    def test_rule_exists_in_frag_rules(self):
        assert any(r.name == "scaffold_name_ring_cleavage" for r in FRAG_RULES)

    def test_smarts_compiles(self):
        rule = next(r for r in FRAG_RULES if r.name == "scaffold_name_ring_cleavage")
        assert Chem.MolFromSmarts(rule.reactant_smarts) is not None

    def test_matches_representative_mol(self):
        # should get exactly 1 match on the target scaffold
        ...

    def test_matches_variant_mol(self):
        # different substituents on the scaffold, still matches
        ...

    def test_no_match_naphthalene(self):
        ...

    def test_no_match_benzene(self):
        ...

    def test_cut_bonds_exist_in_molecule(self):
        # verify each cut bond pair maps to a real bond in the molecule
        ...

    def test_prepass_adds_fragments_to_dag(self):
        # dag with smarts_prepass=True has more nodes than without
        ...
```

Run all tests: `conda run -n <project>-gpu pytest tests/test_smarts_prepass_inject.py -v`

All tests must pass. The total count should increase from the previous number.

---

## Step 8 — Update memory

After all tests pass, update the memory file at
`MEMORY.md` (see the memory system's index conventions)
with a new entry summarizing:
- Rule name and what scaffold it targets
- The 4 cut bonds (in SMARTS map notation)
- merge_outer_map_nums and min_num_atoms
- Number of new tests and total test count
- Any key debug lessons (especially SMARTS gotchas discovered)
- Estimated coverage: N NIST20 molecules with this scaffold

Also add a corresponding memory file at `memory/smarts_rule_<name>.md`.
