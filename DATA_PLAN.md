# Data plan — Idea 2: Ranking Within the Therapeutic Area

Confirmed by literature review (`literature_review.md`) and a full local data inventory.

## Reuse directly
- Same base compound/label resources as Idea 1 (`ICHI2027_mgan/data/compounds.parquet`, `cmp_ecfp.npy`, `cmp_expr.npy`, the nine `labels_*.csv` / epilepsy files, `chembl_max_phase_all.csv`).
- **Prior ICHI result to reproduce, not just cite**: `ICHI2027_mgan/results/candidate_ranking.csv` already shows bemegride ranked #1 for epilepsy (`score=0.794`, `nn_sim_to_aed=0.538`). Reproduce this exact finding first as the paper's motivating evidence, using `ICHI2027_mgan/code/candidates.py` as the reference implementation.
- ATC codes: partially embedded already (`atc` column in each `labels_*.csv`, `chembl_atc_n03.csv` for N03). ATC Level-1 is not separately cached — fetch via the ChEMBL API (`/chembl/api/data/molecule/{chembl_id}` → `atc_classifications`, same API already used by `fetch_labels.py`/`survey_atc.py`) and cache as `data/atc_level1.csv` (ik14, atc_level1_code, atc_level1_name).

## SIDER — partial coverage, must extend
- Existing mapping (`ICLR2027_repurpose/data/lincs_sider_map.csv` + `sider_pairs.csv` + raw `meddra_all_se.tsv.gz`) only covers the 1,725-compound Phase-2 subset (535 of those have a SIDER/PubChem CID match).
- For the full 9,878-compound matrix used here, extend the PubChem-CID-to-SIDER join to the full compound set using the same `meddra_all_se.tsv.gz` raw file (already on disk — no new download needed) and `compounds.parquet`'s existing identifiers. Output: `data/sider_full.parquet` (ik14, se_name/MedDRA PT) for every compound that resolves to a PubChem CID.
- Hard-negative construction: for each indication, flag drugs where SIDER lists the opposite/characteristic adverse effect (e.g. "convulsion"/"seizure" PT for epilepsy; "hypotension" PT for antihypertensives) as a side effect — these become the hardest negatives per literature_review.md §5.

## Proconvulsant list — integrity fix required before use
- The "43 proconvulsants" are **not** a standalone citable file; they exist only as a Python set literal (`PROCONV`, 86 raw strings including synonyms) duplicated across `PACT/PACT/coverage.py`, `modal.py`, `rigor.py`, `rank.py`, `sta.py`, `audit.py`, `audit2.py`.
- Action: extract and deduplicate into a single clean table `data/proconvulsants.csv` (canonical_name, synonyms, source), with provenance split three ways per literature_review.md §3:
  1. Primary citation: Chen, Albertson & Olson, "Treatment of drug-induced seizures," *Br. J. Clin. Pharmacol.* 2016 (the actual literature basis per PACT's own README — cite this directly, not "PACT").
  2. Cross-validation column: does SIDER list "convulsion"/"seizure" as a side effect for this drug? (fully public, derived from `data/sider_full.parquet` above.)
  3. Cross-validation column: risk tier from Ruffmann, Bogliun & Beghi, "Epileptogenic drugs: a systematic review," *Expert Rev. Neurother.* 2006.
- **Do not cite "PACT" as an external resource anywhere in the paper.** If self-citation to the PACT abstract is wanted, disclose plainly that it is the authors' own prior unpublished/non-archival work, per literature_review.md §3's "minimum fix."
- Also diff `PACT/PACT/coverage.py`'s `PROCONV` set against the near-duplicate codebase at `lincs-detectability-atlas/lincs-detectability-atlas/` before treating PACT as canonical (inventory flagged these as near-identical — resolve any discrepancy and note which was used).

## Baselines — status
- **Naive area-score subtraction**: already has a documented result to reproduce (depression AUPRC 0.45 → 0.17) — implement exactly as the lower-bound failure-mode baseline.
- **NRBdMF-style signed matrix factorization**: implement a within-drug signed-label MF baseline (SIDER indication = −1, side effect = +1) per Azuma et al. 2022 (arXiv:2208.04312) as the closest existing method — must be clearly differentiated from the proposed negative-control-indication mechanism, not just nominally included.
- **Plain indication model** (no area correction): the uncorrected starting point.

## Known caveats to carry into the paper
- Novelty claim must be scoped to the repurposing/DTI literature specifically; RUV (Gagnon-Bartsch & Speed 2012) and negative-control pharmacoepidemiology (Lipsitch, Tchetgen Tchetgen, Cohen 2010) are well-established outside this domain and should be cited as the borrowed methodology, not disclaimed as novel.
- Read DRDM (*J. Biomed. Inform.* 2025, confirmed venue) and "Escaping the Drug-Bias Trap" (TCBBIO 2025) in full before finalizing the novelty paragraph — literature review only checked abstracts for these two.
- Run the RUV-style negative-control validity check (no residual association among negative-control indications post-correction) as an actual diagnostic, not an assumption.
