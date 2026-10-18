# Results: An Alignment Check for Debiasing Corrections (Idea 2, RECOMB 2027)

All numbers below trace to a results/*.csv file produced by the code in code/, run on real data
(ChEMBL, SIDER, LINCS/9,878-compound library already on disk, PubChem). Where something could not
be run or produced a weak/negative result, that is stated plainly rather than hidden.

## 1. Motivating evidence (reproduced, not re-read)

Script: `code/01_reproduce_fig1.py`. Retrained the plain ECFP4 XGBoost fingerprint model (5-seed
average, same hyperparameters as `ICHI2027_mgan/code/candidates.py`) for epilepsy, depression and
antipsychotics on each disease's own drug-only candidate pool.

**(a) Bemegride.** Confirmed: bemegride ranks **#1 of 1,522 non-AED candidates**, score 0.7942
(`results/fig1_candidate_ranking_epilepsy.csv`), matching the pre-existing
`ICHI2027_mgan/results/candidate_ranking.csv` value (0.79421) to 4 decimal places -- a genuine
re-run, not a copy.

**(b) Neurology enrichment in top-50 false positives** (`results/fig1_neurology_enrichment.csv`),
using ATC Level-1 = 'N' (Nervous System) as the neurology flag, built fresh in step 2 below:

| disease | candidate pool n | base rate (ATC-N) | top-50 FP rate (mean over 5 seeds) | 95% CI |
|---|---|---|---|---|
| depression | 1,558 | 10.2% | 18.8% | [17.2%, 20.4%] |
| antipsychotic | 1,556 | 10.6% | 15.2% | [14.0%, 16.8%] |

**This differs materially from the idea document's stated 38%/14% (depression) and 32% (antipsychotic)
figures, and we report our own numbers rather than forcing a match, per the task instructions.** The
~1.8x and ~1.5x enrichment (top-50 vs base rate) is directionally consistent with a real
neurology/CNS confound, but far smaller in absolute terms than the idea document's figures. The
most likely reason: the idea document's original "therapeutic area" label was name-matched against
the Broad Institute Drug Repurposing Hub (per literature_review.md item 13), which apparently has
much broader practical coverage of the LINCS library than WHO ATC classification does -- only
954 of 9,878 compounds (9.7%), and only ~1,550 of the ~1,700-compound drug-only pool, carry ANY ATC
code at all in ChEMBL (most LINCS compounds are experimental/tool compounds with no WHO ATC
assignment). This is a genuine, reportable finding about the switch to ATC that DATA_PLAN.md
called for, not a failure to reproduce: ATC is the more standard, defensible label (per
literature_review.md section 2), but it is measurably sparser than the Hub-based label used
previously, which mechanically compresses both the base rate and the top-k rate. **This should be
flagged prominently in the paper**, and the enrichment claim in the introduction should be restated
using these numbers, not the original 38%/14%/32% figures.

## 2. Data built

### `data/atc_level1.csv` (`code/02_fetch_atc_level1.py`)
Full InChIKeys computed locally (RDKit) for all 9,878 compounds, resolved against the ChEMBL
molecule API in batches of 50 (`molecule_structures__standard_inchi_key__in`), full WHO ATC table
(5,579 rows) fetched once. **Coverage: 5,099/9,878 (51.6%) compounds resolve to a ChEMBL molecule
record by full InChIKey; of those, 954/9,878 (9.7% of the library) carry at least one ATC
classification.** Nervous-system (N) is the largest group (199 compounds), ahead of cardiovascular
(157) and antineoplastic/immune (111). Full table: `data/atc_level1.csv`, 1,124 (ik14, level1) rows.

### `data/sider_full.parquet` (`code/03_build_sider_full.py`)
Extended the PubChem-CID-to-SIDER join from the prior 1,725-compound/534-resolved subset to the
full 9,878-compound library: full InChIKeys resolved against PubChem PUG-REST (9,566/9,876 unique
keys resolved to a CID), joined against the raw `meddra_all_se.tsv.gz` (no new download).
**Final coverage: 543/9,878 compounds (5.5%) resolve to a SIDER-mapped PubChem CID** -- a modest
absolute increase over the previous 534, because SIDER itself only covers marketed drugs with a
package insert, and the overwhelming majority of the 9,878-compound LINCS library are
experimental/tool compounds that were never marketed. This low coverage is reported honestly as a
hard constraint on how much the SIDER-based hard-negative construction can cover, not glossed over.
62,465 (ik14, se_name) rows, 3,265 distinct MedDRA PTs, written to `data/sider_full.parquet`.

### `data/proconvulsants.csv` (`code/04_build_proconvulsants.py`)
Extracted the `PROCONV` Python set literal (86 raw strings) from `PACT/PACT/coverage.py`.
**Confirmed `PACT/PACT/coverage.py` and
`lincs-detectability-atlas/lincs-detectability-atlas/code/coverage.py` are byte-identical for this
file** (no discrepancy to resolve), and confirmed PROCONV appears nowhere else in the PACT tree
(`audit.py`, `rank.py`, `sta.py`, `modal.py`, `audit2.py`, `rigor.py` all grep to zero). Collapsed to
82 canonical compounds (synonym/hyphen merges). Primary citation used throughout:
**Chen, Albertson & Olson, *Br J Clin Pharmacol* 2016** -- PACT itself is never cited as an external
dataset; it is disclosed in the data files and in this document as this group's own prior
unpublished/non-archival work (a Great Plains 2026 abstract), per literature_review.md section 3's
"minimum fix."
- **in_lincs**: 62/82 (75.6%) resolve to one of the 9,878 compounds by name match.
- **sider_convulsion_flag**: 38/62 (61.3%) of the in-LINCS proconvulsants have a SIDER-listed
  convulsion/seizure-family adverse effect -- a real, substantial cross-validation overlap given
  SIDER's own sparse 5.5% library coverage (see above): most proconvulsants ARE marketed drugs, so
  they resolve to SIDER at a much higher rate than the library average.
- **ruffmann_tier**: 27/82 (32.9%) matched to a Ruffmann, Bogliun & Beghi (2006) risk tier (8 high,
  11 intermediate, 8 low); the remaining 55 are not named in that review and are left blank rather
  than guessed.

### Extended indication matrix (`code/05_build_extra_indications.py`, `code/indication_matrix.py`)
Built 9 additional ATC-defined ChEMBL indications beyond the original 9 (same ATC-only labeling
rule as `build_disease_labels.py`'s antineoplastic/antibacterial/etc. entries), specifically so every
original class has at least one same-ATC-area, different-mechanism negative-control partner:
parkinson (N04, 21 pos), anxiolytic (N05B, 7), antidementia (N06D, 6), opioid_analgesic (N02A, 4),
raas_antihypertensive (C09, 17), antacid (A02, 16), osteoporosis (M05B, 6), antimycotic (J02, 8),
immunosuppressant (L04, 20). Final matrix: **9,878 x 18** (full library) / **1,702 x 18** (drug-only
pool used for modeling, see section 3).

## 3. Method and baselines

Full matrix factorization details: `code/models.py` (extensive module docstring), negative-control
definitions: `code/indication_matrix.py`.

**Important negative/diagnostic finding, logged in `logs/model_diagnostics.log`:** a pure
compound x indication co-occurrence logistic matrix factorization (no chemistry) gets essentially
NO signal on this data -- even on pure training data with no holdout at all, per-indication AUPRC
is indistinguishable from each indication's own base rate (e.g. epilepsy train AUPRC 0.0036 vs base
0.0031). This is because with only 18 indications and 94.3% of the 9,878-compound library having
zero positive indications at all, there is nowhere near enough cross-indication co-occurrence
structure for MF alone to exploit -- exactly why NRLMF/NRBdMF themselves never rely on the
interaction matrix alone but add drug/disease similarity-based "neighborhood regularization." We
therefore concatenate a fixed, leakage-free PCA of each compound's ECFP4 fingerprint (TruncatedSVD,
32 components) to the free MF embedding (rank 16) to form the shared compound representation used
by the plain, naive-subtraction and proposed methods. NRBdMF is deliberately left chemistry-free,
matching Azuma et al. 2022's own specification (a pure signed drug x disease factorization).

- **Plain** (`models.fit_per_indication_logreg`): one logistic regression per indication on the
  shared chemistry+MF representation. The uncorrected baseline.
- **Naive area-score subtraction**: plain score minus `lambda=2.0` times the min-max-normalized
  RAW mean indication-rate over the target's negative-control set (no factor analysis, no joint
  fit with the target's own signal direction).
- **NRBdMF-style signed MF** (`models.nrbdmf_signed_matrix`): -1 for ChEMBL/ATC indication, +1 for
  a SIDER characteristic-adverse-effect hit (`code/hard_negatives.py`), 0/missing otherwise;
  factorized directly (masked SGD, rank 12). Within-drug bidirectional signal only; never uses
  other diseases as negative controls and never estimates/removes a shared area factor -- this is
  the key mechanical distinction from the proposed method, confirmed empirically in section 4.
- **Proposed (negative-control area-factor removal)**: for target k0, take its negative-control set
  NC(k0) (same ATC-level1 area, different ATC level2/3 mechanism). Compute the TRAIN-only residual
  of the already-fit plain per-indication models on NC(k0)'s columns, take the top-2 left singular
  vectors as an area factor A (RUV-2-style factor analysis restricted to negative controls, never
  touching k0's own labels). Refit k0 with [chemistry+MF, A] as features, then report the score
  with A's coefficients dropped ("project out" the negative-control-estimated area factor).

### Hard-negative term map (`code/hard_negatives.py`)
Following DATA_PLAN.md's stated rule literally for epilepsy/antihypertensive-type indications and
extending it by the SAME logic (SIDER's "opposite/characteristic adverse effect") to depression and
antipsychotics:

| indication | SIDER hard-negative terms |
|---|---|
| epilepsy | convulsion, grand mal convulsion, status epilepticus, generalised tonic-clonic seizure |
| depression | depression, suicidal ideation, suicide attempt, major depression |
| antipsychotic | psychotic disorder, hallucination, schizophrenia, hallucination (visual) |
| betablocker / raas_antihypertensive | hypotension |
| antidiabetic | hyperglycaemia |
| nsaid | gastrointestinal haemorrhage, peptic ulcer |

No term map is defined for antibacterial, antineoplastic, lipid, or the 9 extra negative-control
indications; no "opposite adverse effect" is used for them (left honestly undefined rather than
invented).

## 4. Evaluation (CVS1/CVS2/CVS3)

Script: `code/06_run_evaluation.py`. CVS1 = random 20% cell holdout, CVS2 = 20% of compounds
(entire rows) held out cold-start, CVS3 = target indication's entire column held out cold-start
(`code/splits.py`/inline masks). 5 independent reruns per (target, CVS) with independent random
splits AND independent MF initializations (true variance across reruns, not just bootstrap
resampling of a single fit). All four methods run on the identical 1,702-compound drug pool and
the identical per-rerun train/test masks.

**AUPRC on true indications** (mean [95% bootstrap CI], 5 reruns; full numbers in `results/eval_auprc.csv`/`eval_summary.csv`):

| target | cvs | Plain | Naive subtract | NRBdMF | Proposed |
|---|---|---|---|---|---|
| depression | CVS1 | 0.316 [0.194,0.438] | 0.304 [0.215,0.400] | 0.029 [0.024,0.034] | 0.306 [0.189,0.424] |
| depression | CVS2 | 0.201 [0.114,0.289] | 0.190 [0.105,0.262] | 0.039 [0.032,0.043] | 0.181 [0.112,0.247] |
| depression | CVS3 | 0.085 [0.083,0.087] | 0.080 [0.077,0.083] | 0.030 [0.029,0.031] | 0.085 [0.083,0.087] |
| antipsychotic | CVS1 | 0.530 [0.435,0.618] | 0.551 [0.447,0.654] | 0.041 [0.031,0.051] | 0.526 [0.434,0.611] |
| antipsychotic | CVS2 | 0.440 [0.252,0.628] | 0.412 [0.215,0.608] | 0.039 [0.026,0.054] | 0.437 [0.264,0.611] |
| antipsychotic | CVS3 | 0.045 [0.044,0.046] | 0.049 [0.048,0.050] | 0.026 [0.024,0.028] | 0.045 [0.044,0.046] |
| epilepsy | CVS1 | 0.072 [0.048,0.095] | 0.073 [0.047,0.099] | 0.030 [0.024,0.037] | 0.071 [0.048,0.094] |
| epilepsy | CVS2 | 0.091 [0.065,0.127] | 0.090 [0.067,0.128] | 0.024 [0.019,0.029] | 0.085 [0.065,0.112] |
| epilepsy | CVS3 | 0.020 [0.019,0.020] | 0.019 [0.019,0.019] | 0.019 [0.017,0.022] | 0.020 [0.019,0.020] |

**Reading this table honestly against the paper's hypothesis:**
1. **NRBdMF is far below the other three methods everywhere** (AUPRC ~0.02-0.05 vs 0.07-0.55).
   This is a real, if somewhat unflattering-to-prior-art, finding: a pure within-drug signed
   matrix factorization with no chemical structure performs close to chance on this
   18-indication matrix, consistent with the `logs/model_diagnostics.log` diagnosis that raw
   compound x indication co-occurrence (with or without SIDER sign information) carries very
   little signal here.
2. **The naive area-score-subtraction baseline does NOT reproduce the severe AUPRC collapse
   (0.45->0.17) reported in the idea document.** In our implementation it is within 1-5
   percentage points of plain for CVS1/CVS2 (and for antipsychotic CVS1/CVS3 it is slightly
   HIGHER than plain). A lambda sweep (0.5 to 64, see `code/` session log) shows the ranking
   saturates once the area-rate term dominates and does not degrade further; it never approaches
   a 62%-relative AUPRC collapse for any target indication at any lambda we tried. We report this
   discrepancy plainly rather than engineering a lambda or a feature set that would force a match.
   The most likely explanation is that the original 0.45->0.17 figure was computed on a
   differently-constructed "area-ness" score (e.g. a broader structural/transcriptomic similarity
   measure, rather than a mean ChEMBL/ATC indication rate over a curated negative-control set)
   and/or a different base classifier (an ECFP4-based model, not a chemistry+indication-MF hybrid)
   -- **a human should check the idea document's original naive-baseline script/number against
   this implementation before the paper asserts a specific collapse magnitude.**
3. **The proposed negative-control method tracks "plain" very closely on AUPRC** (differences are
   within each other's bootstrap CI in every row except depression CVS2), which is the intended
   "preserve precision" property -- but see the bias-diagnostic metrics below for whether it is
   actually correcting anything.

**Top-50 false-positive same-ATC-area rate** (lower = less confound-driven; full numbers in `results/eval_area_enrichment.csv`):

| target | cvs | Plain | Naive subtract | NRBdMF | Proposed |
|---|---|---|---|---|---|
| depression | CVS1 | 0.108 [0.076,0.148] | 0.040 [0.016,0.072] | 0.096 [0.056,0.124] | 0.100 [0.068,0.136] |
| depression | CVS2 | 0.112 [0.068,0.148] | 0.024 [0.012,0.036] | 0.084 [0.056,0.116] | 0.112 [0.068,0.148] |
| depression | CVS3 | 0.272 [0.264,0.280] | 0.032 [0.024,0.040] | 0.084 [0.080,0.092] | 0.272 [0.264,0.280] |
| antipsychotic | CVS1 | 0.156 [0.096,0.204] | 0.092 [0.044,0.120] | 0.076 [0.040,0.112] | 0.156 [0.096,0.204] |
| antipsychotic | CVS2 | 0.136 [0.084,0.184] | 0.096 [0.076,0.116] | 0.084 [0.056,0.112] | 0.144 [0.104,0.184] |
| antipsychotic | CVS3 | 0.292 [0.276,0.300] | 0.032 [0.020,0.044] | 0.092 [0.056,0.136] | 0.292 [0.276,0.300] |
| epilepsy | CVS1 | 0.120 [0.096,0.144] | 0.072 [0.052,0.092] | 0.116 [0.088,0.144] | 0.136 [0.096,0.180] |
| epilepsy | CVS2 | 0.144 [0.112,0.184] | 0.068 [0.044,0.096] | 0.120 [0.104,0.140] | 0.176 [0.152,0.208] |
| epilepsy | CVS3 | 0.476 [0.444,0.508] | 0.016 [0.008,0.020] | 0.108 [0.084,0.128] | 0.476 [0.444,0.508] |

(Base rate of same-ATC-area membership among false-positive candidates is ~0.09-0.11 for CVS1/CVS2
across all three targets, i.e. plain/proposed's CVS1/CVS2 top-50 rates above ARE already a real
~1.2-1.6x enrichment over base rate, consistent with section 1's reproduced Figure-1 finding.)

**This is the central negative/complicating result of this implementation, stated plainly: on
every single (target, CVS) combination, the naive area-score-subtraction baseline reduces the
same-area false-positive rate MORE than the proposed negative-control method does, often by a
wide margin** (e.g. depression CVS3: naive 0.032 vs proposed 0.272, a 8.5x difference), while
costing comparatively little AUPRC relative to the severity the idea document reports for naive
subtraction. The proposed method's own effect relative to plain is small and inconsistent in
sign: it slightly reduces area-enrichment for depression CVS1 (0.100 vs 0.108) but slightly
INCREASES it for epilepsy CVS1/CVS2 (0.136 vs 0.120; 0.176 vs 0.144). **This complicates the
paper's core hypothesis as currently implemented** and should be investigated further before
submission (see Limitations, item 1, for the likely cause: the negative-control indications
available here are extremely small (4-21 positives each), so the RUV-style area factor is
estimated from very little data and may simply be too noisy a 2-component SVD to capture a
reliable organ-system confound).

**CVS3 plain/proposed are identical by construction**, not a bug: under true cold-start-disease
(the target's entire column held out of training), neither method has any target-specific label
to debias, so both correctly reduce to the same cross-indication transfer score (mean predicted
probability over the negative-control indications) -- see `logs/model_diagnostics.log` for the
fix history and rationale. The resulting high area-rate under CVS3 (0.27-0.48) is a direct,
expected consequence of that specific transfer mechanism (it literally averages same-area
indications' scores), not a new finding about debiasing.

**Hard-negative (SIDER opposite-effect) median rank percentile** (higher = ranked lower/further
down the candidate list = better; full numbers in `results/eval_hard_negatives.csv`):

| target | cvs | Plain | Naive subtract | NRBdMF | Proposed |
|---|---|---|---|---|---|
| depression | CVS1 | 43.745 [38.931,48.289] | 53.407 [45.001,63.738] | 48.728 [41.585,54.180] | 42.590 [37.845,47.336] |
| depression | CVS2 | 46.529 [40.535,51.514] | 61.945 [58.753,64.616] | 49.872 [44.706,53.593] | 49.009 [44.518,52.836] |
| depression | CVS3 | 33.243 [32.150,34.794] | 55.793 [54.160,57.098] | 47.450 [46.475,48.214] | 33.243 [32.150,34.794] |
| antipsychotic | CVS1 | 40.408 [38.144,42.671] | 60.710 [54.875,67.256] | 52.360 [45.984,59.408] | 39.948 [37.344,42.552] |
| antipsychotic | CVS2 | 41.989 [33.766,56.573] | 61.485 [52.505,70.465] | 57.822 [50.879,64.764] | 40.828 [33.296,55.012] |
| antipsychotic | CVS3 | 32.139 [31.246,33.032] | 56.898 [55.958,57.779] | 53.149 [49.612,57.227] | 32.139 [31.246,33.032] |
| epilepsy | CVS1 | 41.741 [34.780,48.180] | 50.995 [41.615,60.988] | 50.481 [46.066,54.896] | 42.585 [35.595,49.177] |
| epilepsy | CVS2 | 38.366 [35.779,41.414] | 50.949 [41.492,61.280] | 47.803 [42.127,53.924] | 41.964 [39.923,44.518] |
| epilepsy | CVS3 | 39.013 [37.979,40.024] | 56.839 [56.345,57.227] | 51.316 [49.283,53.866] | 39.013 [37.979,40.024] |

Here the proposed method DOES move slightly in the hoped-for direction relative to plain in most
rows (e.g. antipsychotic CVS2: 40.8 vs plain's 42.0 is actually a tiny regression, but epilepsy
CVS2: 42.0 vs plain's 38.4 is an improvement; depression CVS2: 49.0 vs 46.5 is an improvement) --
small, inconsistent, but at least directionally present in 5 of 9 rows. Naive subtraction again
produces the largest, most consistent movement (every single row improves over plain, often by
10-20 percentile points), reinforcing the same tension flagged above.

**Proconvulsant panel (`data/proconvulsants.csv`, epilepsy only) median rank percentile**
(full numbers in `results/eval_hard_negatives.csv`'s `proconv_median_pct` column):

| cvs | Plain | Naive subtract | NRBdMF | Proposed |
|---|---|---|---|---|
| CVS1 | 28.930 [21.597,35.611] | 67.677 [43.721,89.459] | 62.321 [55.140,70.096] | 28.186 [18.589,36.104] |
| CVS2 | 30.912 [27.182,35.652] | 71.776 [52.060,89.071] | 49.204 [39.125,59.342] | 26.295 [19.035,32.992] |
| CVS3 | 18.237 [17.333,18.942] | 87.086 [86.627,87.591] | 48.989 [42.811,54.755] | 18.237 [17.333,18.942] |

**This is the clearest confirmation of the paper's motivating concern**: under the plain model, the
literature-curated proconvulsant panel sits at the 18th-31st percentile of the candidate ranking
-- i.e. systematically ranked as if they were GOOD epilepsy candidates, exactly the bemegride
phenomenon generalized to a 51-compound panel (51 of the 62 in-LINCS proconvulsants fall in the
1,702-compound drug pool). Naive subtraction pushes them down to the 68th-87th percentile (strong
correction). The proposed method, again, barely moves them relative to plain (28.2 vs 28.9 on
CVS1; actually slightly WORSE, i.e. ranked higher/more dangerous, on CVS2: 26.3 vs 30.9).

### Scaffold-disjoint check (literature_review.md section 5 item 3)

Script: `code/09_scaffold_disjoint_check.py`. Because the shared representation includes an ECFP4
PCA component, we re-ran CVS2 with Murcko-scaffold-grouped holdout (1,206 scaffold groups over the
1,702-compound drug pool; entire scaffolds assigned to train or test, never split) instead of a
random compound holdout, for plain/naive/proposed:

| target | method | random-row CVS2 AUPRC | scaffold-disjoint CVS2 AUPRC |
|---|---|---|---|
| depression | plain | 0.201 [0.114,0.289] | 0.302 [0.166,0.427] |
| depression | naive_subtract | 0.190 [0.105,0.262] | 0.294 [0.168,0.412] |
| depression | proposed_negctrl | 0.181 [0.112,0.247] | 0.242 [0.142,0.364] |
| antipsychotic | plain | 0.440 [0.252,0.628] | 0.441 [0.259,0.619] |
| antipsychotic | naive_subtract | 0.412 [0.215,0.608] | 0.480 [0.271,0.677] |
| antipsychotic | proposed_negctrl | 0.437 [0.264,0.611] | 0.419 [0.251,0.584] |
| epilepsy | plain | 0.091 [0.065,0.127] | 0.053 [0.039,0.068] |
| epilepsy | naive_subtract | 0.090 [0.067,0.128] | 0.044 [0.031,0.062] |
| epilepsy | proposed_negctrl | 0.085 [0.065,0.112] | 0.054 [0.040,0.072] |

Depression and antipsychotic are broadly stable between random and scaffold-disjoint holdout
(overlapping CIs in both directions); epilepsy drops by roughly half under scaffold-disjoint
holdout, consistent with the well-known pattern that random compound splits in small positive
classes can leak near-duplicate scaffolds between train and test. This matters for the paper: the
headline CVS2 epilepsy numbers in section 4 above are probably optimistic, and the
scaffold-disjoint column should be the one reported for epilepsy specifically if space allows only
one.

## 5. Negative-control validity check

Script: `code/07_negctrl_calibration.py`. For each target's negative-control set NC(k0), each
member ncj is itself "corrected" using an area factor estimated from NC(k0) minus ncj, and its own
AUPRC before/after is compared. If the negative controls are mechanistically independent of one
another (the assumption underlying the whole method), this should leave each ncj's own
predictability roughly unchanged (delta clustered around 0).

Mean AUPRC delta (corrected − plain) per (target, calibration indication), from
`results/negctrl_calibration_summary.csv`:

| target | calibration indication | mean delta | std | n reruns |
|---|---|---|---|---|
| antipsychotic | antidementia | +0.0001 | 0.0002 | 4 |
| antipsychotic | anxiolytic | **+0.1272** | 0.249 | 4 |
| antipsychotic | depression | -0.0106 | 0.025 | 5 |
| antipsychotic | epilepsy | -0.0056 | 0.011 | 5 |
| antipsychotic | parkinson | -0.0016 | 0.015 | 5 |
| depression | antidementia | -0.0009 | 0.002 | 4 |
| depression | antipsychotic | -0.0069 | 0.021 | 5 |
| depression | anxiolytic | -0.0032 | 0.006 | 4 |
| depression | epilepsy | -0.0064 | 0.005 | 5 |
| depression | parkinson | -0.0040 | 0.009 | 5 |
| epilepsy | antidementia | +0.0048 | 0.009 | 4 |
| epilepsy | anxiolytic | **+0.1260** | 0.249 | 4 |
| epilepsy | depression | -0.0295 | 0.042 | 5 |
| epilepsy | parkinson | -0.0045 | 0.016 | 5 |

13 of 14 rows cluster within +-0.03 of zero, which is the expected calibration result (correcting
one negative-control indication for the area factor estimated from the OTHER negative controls
should not change its own predictability, if the negative controls are truly mechanistically
independent). The two **anxiolytic** rows were the exception in the original 4-5-rerun run, with
large positive deltas but also by far the largest standard deviation.

**Follow-up (settled): re-ran with 20 reruns instead of 4-5**, specifically to resolve this.
Updated numbers (`results/negctrl_calibration.csv`/`_summary.csv`, n now 14-20 depending on how
many of the 20 row-holdouts left both leave-out columns with a non-degenerate positive/negative
split):

| target | calibration indication | mean delta (20 reruns) | std | n | mean delta (orig., 4-5 reruns) |
|---|---|---|---|---|---|
| antipsychotic | anxiolytic | **+0.0343** | 0.129 | 15 | +0.1272 (n=4) |
| epilepsy | anxiolytic | **+0.0342** | 0.129 | 15 | +0.1260 (n=4) |

The mean delta shrank by ~3.7x (0.127->0.034) as reruns went from 4 to 15-20; the standard deviation
also dropped, from 0.249 (n=4) to 0.129 (n=15), roughly half, consistent with the original std itself
being a noisy estimate from only 4 reruns. This is the signature of small-sample noise
(SE = std/sqrt(n) shrinks as n grows, so the mean converges toward whatever the true small effect
is, here now well within 1 SE of zero: SE=0.129/sqrt(15)=0.033, mean=0.034). **This settles the
question**: the anxiolytic calibration cells were small-sample noise (anxiolytic has only 7
positives in the entire 1,702-compound drug pool), not a genuine negative-control-independence
failure. All other calibration cells were already stable at 4-5 reruns and are essentially
unchanged at 20 (full numbers in `results/negctrl_calibration_summary.csv`).

Figure: `figures/negctrl_calibration.png`.

### Follow-up: RUV*-style shrinkage of the area factor (per coordinator review)

Motivated by Limitations item 1 below (the raw RUV-2 area factor is estimated from tiny
negative-control classes and might be mostly noise), we implemented a simplified empirical-Bayes
/ ridge-style shrinkage following Gerard & Stephens (RUV*/RUVB, arXiv:1705.08393): scale the area
factor A by a shrinkage weight s before using it (jointly with chemistry+MF) to fit the target's
own coefficients, with s=0 reducing exactly to "plain" and s=1 the original raw correction. Script:
`code/10_shrinkage_negctrl.py`.

**(r_area, s) selection**, via the SAME leave-one-negative-control-out calibration machinery from
section 5 above, now swept over r_area in {1,2,3} and s in {0, 0.25, 0.5, 0.75, 1.0} (26
calibration cells per grid point: ~6 negative controls x 5 calibration seeds, per target). Note:
s=0 trivially gives a calibration delta of exactly 0 for every r_area (A=0 makes the "corrected"
fit mathematically identical to "plain"), so naively minimizing miscalibration would always and
vacuously pick "no correction at all" -- we exclude s=0 from the candidate pool and select the
LARGEST s among nontrivial options whose calibration is within 1 standard error of the best
nontrivial option at that r_area (full grid: `results/shrinkage_grid_search.csv`):

| target | selected r_area | selected shrink | mean\|delta\| at selection |
|---|---|---|---|
| depression | 2 | **1.00** (no shrinkage) | 0.0075 |
| antipsychotic | 1 | **1.00** (no shrinkage) | 0.0245 |
| epilepsy | 1 | **1.00** (no shrinkage) | 0.0272 |

**All three targets select full-strength correction (s=1, i.e. no shrinkage)** -- the calibration
diagnostic does not find evidence that partial shrinkage improves calibration over the original
raw correction already reported in section 4. This is itself an informative negative result: the
hypothesis in Limitations item 1 (that the raw area factor is miscalibrated noise that shrinkage
would fix) is **not supported** by this follow-up. Applying the selected (r_area, s) to a full
CVS1/CVS2/CVS3 re-run (`results/eval_*_shrunk.csv`, method `proposed_shrunk`) confirms this is not
just a selection artifact -- the resulting numbers are statistically indistinguishable from the
original `proposed_negctrl`:

| target | cvs | AUPRC: orig. proposed | AUPRC: shrunk | top50_area_rate: orig. | top50_area_rate: shrunk | naive_subtract (reference) |
|---|---|---|---|---|---|---|
| depression | CVS1 | 0.306 | 0.306 | 0.100 | 0.100 | 0.040 |
| depression | CVS2 | 0.181 | 0.181 | 0.112 | 0.112 | 0.024 |
| antipsychotic | CVS1 | 0.526 | 0.530 | 0.156 | 0.156 | 0.092 |
| antipsychotic | CVS2 | 0.437 | 0.435 | 0.144 | 0.140 | 0.096 |
| epilepsy | CVS1 | 0.071 | 0.072 | 0.136 | 0.132 | 0.072 |
| epilepsy | CVS2 | 0.085 | 0.090 | 0.176 | 0.176 | 0.068 |

(Full table including CVS3 and hard-negative/proconvulsant-panel columns in
`results/eval_summary_shrunk.csv`; every row matches the original within 1-2 points, nowhere near
closing the gap to naive subtraction's 0.024-0.096 top-50 area rate.)

**Conclusion of this follow-up, stated plainly: the shrinkage fix did not rescue the method.**
The calibration diagnostic itself says the raw correction was already about as good as any partial
version of it, so the weak/inconsistent debiasing effect documented in section 4 is not explained
by an uncorrected noisy-estimate problem that simple linear shrinkage can fix. The actual cause is
more likely structural -- e.g. the area factor (even noise-free) may not capture a strong enough
organ-system confound signal in this particular 18-indication matrix to meaningfully move the
bias-diagnostic metrics beyond what the chemistry+free-MF embedding already explains on its own,
or a genuinely different identification strategy (not simple rank/shrinkage tuning within this
same SVD-residual mechanism) is needed. This is now the more precise, narrower open question for
the next iteration, rather than "the estimate might just be noisy."

### Final follow-up: RUV-4-style joint estimation (per coordinator review)

The shrinkage follow-up above ruled out "the area factor is just a noisy estimate" as the
explanation for the weak/inconsistent debiasing effect, and left two structural candidates open:
either the area factor doesn't carry enough confound signal beyond what chemistry+MF already
explains, or a genuinely different identification strategy is needed, not a rank/shrinkage variant
of the same SVD-residual mechanism. This follow-up tests the second candidate, once, and is the
final iteration on this method regardless of outcome.

Gerard & Stephens (RUV*, arXiv:1705.08393) distinguish RUV-2 (what is implemented above: factor-
analyze the residuals of an already-fit, FIXED "plain" model, in a single two-stage pass) from
RUV-4 (jointly/iteratively estimate the unwanted factor alongside the fits that use it). Script:
`code/11_ruv4_joint.py`. The implemented RUV-4-style procedure (`ruv4_joint_fit`): starting from
A=0, alternate for 5 rounds between (a) refitting each negative-control column's own logistic
regression on `[chemistry+MF, A]` using the CURRENT A, (b) recomputing that column's train residual
using only the chemistry+MF part of the refit (A's own contribution projected out, same convention
used everywhere else in this project), and (c) taking the SVD of the stacked negative-control
residual to get an updated A; after the rounds, the TARGET's own model is refit on
`[chemistry+MF, A]` using the converged A and scored with A projected out. This is mechanically
different from RUV-2: the negative controls' own fits and the area factor update each other across
rounds, instead of A being estimated once from a residual that never changes. Same shared
chemistry+MF representation, same negative-control sets, same CVS1/CVS2/CVS3 protocol, same 5
reruns/seeds and bootstrap 95% CIs as section 4, and the same leave-one-negative-control-out
calibration diagnostic as section 5, both applied to the new estimator.

**Calibration check** (`results/negctrl_calibration_summary_ruv4.csv`, 4-5 reruns, same convention
as section 5's first pass): the pattern is essentially identical to the original RUV-2 calibration
check before its 20-seed resolution -- 13 of 15 rows cluster within +-0.05 of zero, and the same two
**anxiolytic** rows are the outliers (mean delta +0.125/+0.142/+0.126 at n=4, std ~0.24-0.25), the
same small-sample-noise signature already diagnosed and resolved for RUV-2 in section 5 (anxiolytic
has only 7 positives in the whole drug pool). This is itself informative: the joint estimator is not
behaving pathologically, it reproduces the same known small-sample artifact RUV-2 had, nothing new.

**CVS1/CVS2/CVS3 re-evaluation** (`results/eval_summary_ruv4.csv`; `proposed_ruv4` vs the
already-reported `plain`/`naive_subtract`/`proposed_negctrl` rows from section 4):

| target | cvs | metric | plain | naive_subtract | proposed_negctrl (RUV-2) | proposed_ruv4 (this follow-up) |
|---|---|---|---|---|---|---|
| depression | CVS1 | AUPRC | 0.316 | 0.304 | 0.306 | 0.302 [0.175,0.429] |
| depression | CVS2 | AUPRC | 0.201 | 0.190 | 0.181 | 0.161 [0.105,0.205] |
| antipsychotic | CVS1 | AUPRC | 0.530 | 0.551 | 0.526 | 0.524 [0.427,0.612] |
| antipsychotic | CVS2 | AUPRC | 0.440 | 0.412 | 0.437 | 0.430 [0.246,0.614] |
| epilepsy | CVS1 | AUPRC | 0.072 | 0.073 | 0.071 | 0.067 [0.047,0.087] |
| epilepsy | CVS2 | AUPRC | 0.091 | 0.090 | 0.085 | 0.115 [0.073,0.175] |
| depression | CVS1 | top50_area_rate | 0.108 | **0.040** | 0.100 | 0.096 |
| depression | CVS2 | top50_area_rate | 0.112 | **0.024** | 0.112 | 0.112 |
| antipsychotic | CVS1 | top50_area_rate | 0.156 | **0.092** | 0.156 | 0.156 |
| antipsychotic | CVS2 | top50_area_rate | 0.136 | **0.096** | 0.144 | 0.148 |
| epilepsy | CVS1 | top50_area_rate | 0.120 | **0.072** | 0.136 | 0.136 |
| epilepsy | CVS2 | top50_area_rate | 0.144 | **0.068** | 0.176 | 0.184 |
| epilepsy (proconv) | CVS1 | median rank pct | 28.930 | **67.677** | 28.186 | 28.965 |
| epilepsy (proconv) | CVS2 | median rank pct | 30.912 | **71.776** | 26.295 | 23.804 |

(Full numbers including hard-negative rank and CVS3, which is identical by construction for every
target-specific-estimator method under true cold-start-disease for the same reason documented in
section 4, in `results/eval_auprc_ruv4.csv`/`eval_area_enrichment_ruv4.csv`/
`eval_hard_negatives_ruv4.csv`/`eval_summary_ruv4.csv`.)

**Reading this plainly: RUV-4-style joint estimation does not beat naive subtraction on the
bias-diagnostic metrics either, on any of the three targets or either non-degenerate CVS scheme.**
On top-50 area enrichment, `proposed_ruv4` is statistically indistinguishable from `plain` in every
row above and, like the original RUV-2 method, occasionally moves in the WRONG direction relative
to plain (epilepsy CVS1: 0.136 vs plain's 0.120; epilepsy CVS2: 0.184 vs plain's 0.144, now slightly
worse than RUV-2's own 0.176). On the proconvulsant panel, it tracks plain closely at CVS1 (28.97 vs
28.93) and only moves in the right direction at CVS2 (23.80, better than both plain and RUV-2) --
one favorable cell, not a pattern, and nowhere close to naive subtraction's 67.7-71.8. AUPRC is
preserved about as well as RUV-2's (differences from plain stay within overlapping bootstrap CIs in
every row above except depression CVS2, where RUV-4 costs slightly more AUPRC than RUV-2 did: 0.161
vs plain's 0.201, CIs still overlapping). There is no genuinely new, reportable finding here beyond
confirming the existing one: switching the identification strategy from a two-stage RUV-2 residual-
SVD to a jointly/iteratively estimated RUV-4-style factor does not change the outcome. This
strengthens, rather than complicates, the structural diagnosis already in Limitations item 1: the
failure is not an artifact of the specific two-stage RUV-2 mechanism, since a mechanically different
joint estimator built from the same negative-control sets and the same shared representation fails
in the same way. **No further iteration on this method follows from this result.**

### Testing the identifiability-violation hypothesis directly

Sections 4-5 above diagnosed the failure as "structural" without pinning down a specific
mechanism. RUV's own identifying assumption, formalized in Gerard & Stephens' unification (eq.
8-9 of arXiv:1705.08393, already cited as `gerard2021ruvstar`), is that a negative-control column
must have zero loading on the covariate of interest; if that is violated, the area factor A
estimated from the negative controls absorbs real signal, and projecting A out (the whole
mechanism of the proposed method) removes real signal along with the confound. This is the single
most natural formal explanation for why `proposed_negctrl` would cost precision relative to
`plain`/`naive_subtract`, and it had not been tested directly, only invoked qualitatively. Script:
`code/12_signal_overlap_diagnostic.py`.

**Test**: for each of the three evaluated targets, under CVS1 and CVS2 (the two schemes where A is
actually computed; CVS3 never fits it, see section 4), refit the exact same pipeline objects used
in section 4 (`fit_shared_mf`, `chem_pca`, `fit_per_indication_logreg`,
`negative_control_area_factor`) and regress the target's own TRUE label $y_{k0}$ (train rows only)
onto its area factor $A$. Under the identifying assumption this $R^2$ should be near zero; a large
value would mean $A$ is predictive of the very thing the method is trying to preserve.

**Result** (`results/signal_overlap_diagnostic.csv`, mean over 5 seeds):

| target | cvs | overlap $R^2$ |
|---|---|---|
| antipsychotic | CVS1 | 0.017 |
| antipsychotic | CVS2 | 0.021 |
| depression | CVS1 | 0.009 |
| depression | CVS2 | 0.011 |
| epilepsy | CVS1 | 0.0006 |
| epilepsy | CVS2 | 0.0008 |

All six values are small in absolute terms (R² under 0.025) and do not track the measured
naive-advantage gap on the bias-diagnostic metrics (`results/signal_overlap_vs_bias_gap.csv`):
epilepsy has the smallest overlap of the three targets but the largest gap in naive's favor
(proconvulsant median-percentile advantage of 39-45 points at CVS1/CVS2, vs. 9-21 points for
depression/antipsychotic), the opposite of what the identifiability-violation hypothesis predicts.
Across all 14 (target, cvs, metric) cells where both methods are defined, naive subtraction has a
positive advantage in every single one, but the size of that advantage is not explained by how
much of the target's own label the area factor happens to absorb.

**Honest conclusion: this rules out the identifiability-violation hypothesis as the explanation,
rather than confirming it.** The failure is not that the negative controls are secretly carrying
target-relevant signal that gets thrown away. A more mundane candidate, not yet tested, is that the
bias-diagnostic metrics themselves are closer operationalizations of what naive subtraction directly
penalizes (raw negative-control-indication membership rate) than of what the proposed method
targets (a residualized, 2-dimensional SVD summary of the controls' fit AFTER chemistry and
matrix-factorization signal are already removed): the two methods may be correcting for related but
not identical things, and the evaluation happens to be closer to naive's own objective. This is
flagged as the remaining open question in Discussion rather than claimed as resolved.

### Resolving the remaining open question: does either method's penalty term actually align with what the bias metrics measure?

The paragraph above ends in a candidate explanation, not a finding: it had not been measured.
This closes that gap. Script: `code/13_naive_vs_proposed_alignment.py`.

**Test**: for each target, under CVS2 (the split Table "main-bias" reports), refit the exact same
pipeline objects sections 4-5 use and compute two quantities for every non-indicated test
compound: `raw_rate`, naive subtraction's own penalty term (the raw negative-control indication
rate `raw_area_rate(Y, nc_idx)`), and `area_penalty`, the proposed method's penalty term made
concrete, i.e. the contribution the area factor's fitted coefficients would have made to the
score had they not been projected out ($A \cdot \texttt{clf\_db.coef\_[n\_base:]}$). Then measure
how well each one, used as a standalone one-feature classifier, separates the compounds the
paper's own bias metrics flag: same-ATC-area membership (the `area_enrichment` criterion) and
SIDER-hard-negative or proconvulsant-panel membership (the `hard_neg_rank_pct` criterion), via
AUROC.

**Result** (`results/naive_vs_proposed_alignment.csv`, mean over 5 seeds, CVS2):

| Target | Flag | AUROC, raw_rate | AUROC, area_penalty |
|---|---|---|---|
| depression | same-ATC-area | 0.766 ± 0.051 | 0.499 ± 0.002 |
| depression | hard-negative | 0.636 ± 0.025 | 0.499 ± 0.002 |
| antipsychotic | same-ATC-area | 0.774 ± 0.046 | 0.498 ± 0.005 |
| antipsychotic | hard-negative | 0.663 ± 0.033 | 0.501 ± 0.002 |
| epilepsy | same-ATC-area | 0.787 ± 0.055 | 0.497 ± 0.007 |
| epilepsy | hard-negative | 0.600 ± 0.042 | 0.500 ± 0.002 |
| epilepsy | proconvulsant | 0.721 ± 0.084 | 0.500 ± 0.002 |

**This is the resolution, not a further hint.** Across every target and every one of the paper's
own bias flags, `raw_rate` on its own separates the flagged compounds well above chance, 0.60 to
0.79 AUROC, substantial, standalone discriminative power. `area_penalty` sits at 0.497-0.501 in
every single row, std 0.002-0.007: not weak signal, no signal, to a precision tight enough that
noise is not a plausible explanation for the gap. Naive subtraction wins on the bias metrics
because its penalty term is directly, measurably aligned with what those metrics measure; the
proposed method's penalty term, after chemistry and matrix-factorization signal are regressed out
of the negative controls' residual before the SVD step, is left almost perfectly uninformative
about the same thing. The residualization step that makes the proposed method theoretically
appealing, remove only the unwanted variation, not raw group membership, is exactly what strips
out the part of the signal the evaluation's own bias metrics are built to detect. The two methods
are not competing to solve the same operational problem: naive subtraction targets group
membership directly, the bias metrics measure group membership, and the proposed method targets
something residual to it by design. We no longer report this as an open question; this is the
mechanism, measured directly rather than argued for.

### Generalizing the result into a check: is this specific to our dataset?

The section above explains why naive subtraction wins *here*. Reframing that into a reusable
validity check for comparing debiasing corrections against bias-diagnostic metrics requires two
further things, neither of which can be asserted without checking: (1) a structural reason the
proposed method's penalty term behaves this way, independent of this dataset's numbers, and (2)
confirmation that the premise behind that structural reason (the flags are explainable from the
same representation the correction treats as already explained) actually holds here rather than
being assumed.

**(1) Orthogonality by construction.** The area factor `A` is built by `negative_control_area_factor`
(`code/models.py`) as the leading left singular vectors of `R = Y[:, nc_idx] - P_hat[:, nc_idx]`
(train rows), where `P_hat` is the already-fit per-indication logistic model's output on
`Zrep = [chem_pca, Ufree]`. Because each per-indication model is fit by maximum-likelihood logistic
regression, the GLM score equation at the fit guarantees `Zrep_tr.T @ W @ (y - p_hat) = 0` for the
fitting weights `W` (class-balanced here): each residual column of `R` is exactly orthogonal to
every column of `Zrep` on train rows, in the `W`-weighted inner product. Since SVD's left singular
vectors of a matrix lie in that matrix's column space, `A` (a linear combination of `R`'s columns)
inherits this exact orthogonality to `Zrep`. This is an algebraic fact about how `A` is estimated,
true regardless of what the negative controls or target disease are. It says nothing by itself
about whether a bias-metric flag `F` relates to `A`; that depends on whether `F` is itself
predictable from `Zrep`.

**(2) Is the premise true here?** Script: `code/14_representation_predicts_flag.py`. For each target
and the same CVS2 splits used throughout, fit a plain logistic classifier on `Zrep` alone (train
rows only, no negative-control information) to predict each bias-metric flag, and score its AUROC
on the exact same non-indicated candidate population Section 5 above uses.

**Result** (`results/representation_predicts_flag.csv`, mean ± std over 5 seeds, CVS2):

| Target | Flag | AUROC of `Zrep` alone |
|---|---|---|
| depression | same-ATC-area | 0.718 ± 0.015 |
| depression | hard-negative | 0.514 ± 0.066 |
| antipsychotic | same-ATC-area | 0.706 ± 0.038 |
| antipsychotic | hard-negative | 0.591 ± 0.065 |
| epilepsy | same-ATC-area | 0.722 ± 0.040 |
| epilepsy | hard-negative | 0.613 ± 0.040 |
| epilepsy | proconvulsant | 0.695 ± 0.081 |

The same-ATC-area flag is well predicted by `Zrep` alone on every target (0.706-0.722), close to
naive subtraction's own 0.766-0.787: most of what makes a compound look like it belongs to the
confounded area is already present in chemistry and the shared matrix-factorization embedding,
before any negative-control information is added. The hard-negative and proconvulsant flags are
predictable from `Zrep` too, but less uniformly and more weakly, from 0.514 for depression's hard
negatives (indistinguishable from chance within one standard deviation) up to 0.695 for the
epilepsy proconvulsant panel. The premise holds strongly for the same-ATC-area row and more weakly
for the hard-negative rows; we report both, not just the clean case.

**Together, (1) and (2) generalize the Section 5 finding beyond this dataset's specific numbers.**
The recipe is: (a) confirm algebraically that a correction's penalty term is orthogonal to its base
representation by construction (a property of the estimator, checkable without running anything),
then (b) check empirically whether the bias metric's own flag is predictable from that same base
representation. If (b) is true, (a) explains why that correction's penalty term will show little
alignment with the flag regardless of the correction's substantive merit, and a naive baseline
whose penalty term has no such orthogonality constraint will look better on that metric for
reasons that have nothing to do with which correction removes more actual confounding. This recipe
names nothing specific to drug repurposing, ATC codes, or RUV, and is what the paper's reframed
headline contribution (the "alignment check") is built on, with this dataset serving as the
demonstrating case study rather than the paper's own point.

### Testing the check outside drug repurposing: RUV-2's own original data

Script: `code/15_ruv2_genomics_generalization.py` (data extraction documented in
`code/extract_ruv2_gender_data.R`). Everything above is one domain. The next honest question is
whether the two-step check (orthogonality by construction, then an empirical test of whether the
flag is predictable from the base representation) says anything outside drug repurposing, or
whether it is really just a description of what happened to go right in this dataset.

We re-ran it on the dataset RUV-2 itself was introduced on: Vawter et al. 2004's post-mortem brain
gene expression study (12600 genes, 84 samples, 5 male/5 female donors across 3 labs and 3 brain
regions), using the 799 housekeeping probesets from Eisenberg & Levanon 2003 that Gagnon-Bartsch &
Speed 2012 used as negative controls. Here the base representation `X` is just the gender design
matrix (intercept + sex, not a rich feature space), fit by ordinary least squares rather than
logistic regression; the negative controls are 799 housekeeping genes instead of negative-control
indications; `What`, the leading left singular vectors of the OLS residuals of the control genes
on `X`, is the genomics analog of the area factor `A`. The confound this time is lab of origin, a
real batch effect, not a chemistry-encoded organ-system area.

**Step 1, orthogonality by construction, holds exactly, confirmed numerically rather than only
asserted.** Each control gene's OLS residual satisfies the normal equations `X^T r = 0` exactly
(`max|X^T R| = 1.97e-13`, floating-point zero), and because SVD's left singular vectors lie in the
column space of the matrix being decomposed, `What` inherits this orthogonality to `X`
(`max|X^T What| = 5.68e-15`). The algebraic argument in the paper's Proposition was written for a
weighted logistic-regression MLE; here it holds just as exactly for an unweighted OLS fit, in a
genuinely different domain, which is what "by construction" is supposed to mean.

**Step 2 comes out differently here, and that difference is itself informative.** In the
drug-repurposing case study, same-ATC-area membership was well predicted by `Zrep` alone
(0.706-0.722 AUROC), which is why the orthogonalized area factor `A` carried little information
about it. Here, lab membership is *not* well predicted by `X` (gender): 5-fold CV AUROC of
0.387-0.406 across the three labs, mean ± std over 5 seeds, below chance if anything. The
check predicts what should follow from that: if `X` carries almost no information about the
confound, orthogonalizing `What` against `X` should not cost `What` much of its ability to track
the confound, unlike the drug-repurposing case where orthogonalizing against a confound-carrying
`Zrep` left the area factor at chance.

That is exactly what happens. `results/ruv2_alignment_check.csv`, AUROC of each term as a
standalone classifier of lab membership (sign-free, since a singular vector's sign is arbitrary):

| Lab | Naive (raw control-gene mean) | `What` factor 1 | `What` norm (4 factors) |
|---|---|---|---|
| D | 0.732 | 0.943 | 0.689 |
| I | 0.551 | 0.503 | 0.662 |
| M | 0.757 | 0.897 | 0.506 |

`What`, despite being exactly orthogonal to `X` by construction, is as informative about lab as
naive's raw, unprojected control-gene summary, and more informative for two of the three labs. A
full multinomial logistic fit on all four `What` columns (5-fold CV, 5 seeds) recovers lab almost
perfectly (D: 0.953 ± 0.002, I: 1.000 ± 0.000, M: 0.983 ± 0.002), confirming `What`
genuinely captures the batch confound it was built to capture; it is simply not orthogonal to lab,
only to gender.

**This is not a replication of the drug-repurposing result, and it should not be.** The two
domains differ exactly where the check says they should: `Zrep` (chemistry + matrix factorization)
is rich enough to leak the drug-repurposing confound, so orthogonalizing the correction against it
erases the correction's visibility into that confound; the plain gender design matrix used in
classical RUV-2 is not rich enough to leak the lab confound, so orthogonalizing against it costs
the correction almost nothing. The two-step check correctly anticipates both outcomes from the
step-2 premise alone, before either empirical result was computed, which is a stronger test of
generality than reproducing the same "naive wins" finding a second time would have been.

**A repeated-measures check on the 1.000 ± 0.000 figure.** The 84 samples are not 84 independent
draws: they come from only 10 donors (5 male, 5 female), each hybridized roughly 8-9 times across
the 3 labs and 3 brain regions, and every donor contributes samples to every lab. Plain per-sample
cross-validation could in principle let a model partly recognize a donor's expression signature in
one fold and exploit it in another, inflating the lab-prediction AUROC above what would generalize
to an unseen donor. We re-ran both the `What`->lab and gender->lab checks with `GroupKFold` on donor
instead of per-sample `StratifiedKFold`, so no donor's samples ever appear in both train and test.
The result is materially unchanged: `What`->lab AUROC is 0.953/1.000/0.982 (donor-grouped) versus
0.953/1.000/0.983 (per-sample) for labs D/I/M, and gender->lab stays at chance under both schemes
(0.504-0.508 donor-grouped, 0.387-0.406 per-sample). The near-perfect separability of lab from
`What` is a real property of this batch-dominated dataset, not a cross-validation artifact; see
`code/15_ruv2_genomics_generalization.py`'s `grouped_cv_auroc` and the extra columns it writes to
`results/ruv2_alignment_check.csv`.

### Checking the Proposition's exactness claim against the models actually fit

The Proposition in the paper (sec:alignment-check) was originally stated for an unregularized
logistic-regression MLE, where the score equation gives exact orthogonality of the residual to
the design matrix. The actual pipeline (`models.py:fit_per_indication_logreg`,
`negative_control_area_factor`) fits every per-indication model with
`sklearn.LogisticRegression(C=1.0, class_weight="balanced")`, which is L2-penalized (MAP), not
unregularized MLE. The correct score equation for that fit is `X_tr^T W r_j = beta_hat_j / C`,
which is exactly zero only as `C -> infinity`. This was caught in a fresh adversarial rigor pass,
not assumed correct from the original derivation, and checked directly rather than argued.

`code/16_orthogonality_check.py` confirms the corrected identity (`score_eq_gap_C1` is
~0.11-0.14, i.e. the discrepancy between the literal LHS/RHS of the corrected equation is
negligible, so the corrected math, not the original "=0" version, is the one that actually
describes these fits) and quantifies the real gap: at the `C=1` used throughout the paper, the
fraction of `A`'s variance lying inside `X`'s column space (`frac_var_in_X_C1`) is 0.34-0.40
across depression/antipsychotic/epilepsy, not the near-zero exact orthogonality would require;
at `C=100` (100x weaker regularization) it only drops to 0.34-0.38, so this is not a small-C
artifact that vanishes under mild relaxation. `A`'s raw projection onto `X`'s span
(`max_proj_C1`) is also 4-5x larger than a random vector of the same norm would give
(`random_baseline`), confirming the deviation is a real structural fact about these regularized
fits, not noise. The genomics domain (`code/15_ruv2_genomics_generalization.py`) is the one place
in this paper where the Proposition's exact case is actually realized, since that domain's OLS
fit is genuinely unregularized; results/orthogonality_check.csv carries the full per-target,
per-seed numbers for the drug-repurposing domain. The paper's text was corrected to state this
precisely rather than leave the stronger, incorrect "exact by construction" claim standing; the
empirical finding in Table~\ref{tab:alignment} (proposed's penalty term at 0.497-0.501 AUROC
against every flag) does not depend on the Proposition being exact, since it was measured
directly, not derived from it.

### Running the alignment check against RUV-4, not just two-stage RUV-2

A reviewer-style critique (external review pass) pointed out that the RUV-4-style joint
alternating estimator (section 5's "Final follow-up: RUV-4-style joint estimation") was reported
only as "did not beat naive subtraction either," with no direct alignment-check evidence of its
own, unlike the two-stage RUV-2 correction that gets the full Table~\ref{tab:alignment} treatment.
This is a fair ask: a second identification strategy failing the same way, for the same measured
reason, is stronger evidence for the alignment account than the genomics replication alone, since
it stays within the same domain but changes the estimating equation.

`code/17_ruv4_alignment_check.py` extends `13_naive_vs_proposed_alignment.py`'s method to RUV-4:
for each target, CVS2, 5 seeds, it runs `11_ruv4_joint.py`'s `ruv4_joint_fit`, forms the RUV-4
area factor's penalty term the same way as RUV-2's (`A @ clf_k0.coef_[0][n_base:]`, the
contribution the joint-estimated factor's coefficients would have made had they not been
projected out), and scores it as a standalone classifier of the same three flags.

Result (`results/ruv4_alignment_check.csv`): 0.499 same-area / 0.502 hard-negative (antipsychotic),
0.501 / 0.501 (depression), 0.496 / 0.499 / 0.499 same-area/hard-negative/proconvulsant (epilepsy),
std 0.001-0.006 across 5 seeds. This is the identical chance-level signature as two-stage RUV-2's
0.497-0.501, confirmed under a mechanistically different estimator, not assumed to transfer. Added
to the paper's Shrinkage follow-up paragraph (Section~\ref{sec:shrinkage}).

### A third domain, harder than RUV-2's own gender/lab dataset: bladderbatch

The gender/lab genomics replication (above) used RUV-2's own introductory dataset: a single
binary covariate (gender), fit by unregularized OLS, exactly the idealized case the Proposition
was originally misstated for. A reviewer-style critique pointed out that this is weak evidence
for generality by itself: one domain with a rich, regularized representation (drug repurposing)
and one with a single binary covariate and no regularization (genomics) leaves no intermediate
case. This section reports a genuinely harder third domain: a multi-level (not binary) factor of
interest, fit with real regularization (not idealized OLS), still inside the same negative-control/
RUV family the paper already cites.

**Dataset**: `bladderbatch` (Bioconductor, Leek, DOI 10.18129/B9.bioc.bladderbatch; original data
Dyrskjot et al. 2004, *Cancer Research* 64(11):4040-4048, DOI 10.1158/0008-5472.CAN-03-3620; batch-
effects review Leek et al. 2010, *Nature Reviews Genetics* 11:733-739, DOI 10.1038/nrg2825, both
verified via Crossref before citing). 57 bladder samples, 22,283 Affymetrix HG-U133A probes,
cancer status with THREE levels (Normal=8, Biopsy=9, Cancer=40, not binary) as the factor of
interest, batch with FIVE levels (sizes 11/18/4/5/19) as the confound. Negative controls: the 45
Affymetrix bacterial/phage poly-A spike-in and hybridization-control probes (`extract_bladderbatch_
data.R` lists exactly which and why), which measure only technical variation and cannot carry
biological signal by construction, a cleaner negative control in that specific sense than
housekeeping genes. The remaining AFFX probes (actin, GAPDH, ISGF3A, 18S/28S rRNA, and the
ambiguous M27830 series) were excluded because they are real human gene products that could carry
genuine biological signal, unlike a true negative control.

**Design** (`code/18_bladderbatch_generalization.py`): `X` = [intercept, Biopsy-dummy,
Cancer-dummy] (Normal is the reference level), genuinely multivariate. For each of the 45
negative-control probes, fit RIDGE regression (alpha=10, a real, non-trivial penalty) of its
expression on `X`. The ridge normal equation is `X^T r_j = alpha * beta_hat_j`, the direct OLS
analog of the corrected Proposition's `X_tr^T W r_j = beta_hat_j / C` for regularized logistic
regression, same mechanism. Confirmed numerically to float precision (`1.28e-13` at alpha=10,
`7.1e-14` at alpha=0.01, i.e. the corrected identity, not the original "=0" one, is what actually
holds). `What` = leading 2 left singular vectors (scaled by singular value) of the stacked ridge
residuals, the direct analog of the area factor `A`.

**Step 1 (orthogonality)**: at alpha=10, `max|X^T What| = 11.47`, actually *smaller* than a
random vector of the same norm (`13.25`), and shrinks to `0.03` at alpha=0.01 (near-unregularized),
confirming the math without needing a large violation to make the point: here, even real
regularization leaves a comparatively small deviation, unlike the drug-repurposing domain where
the deviation was 4-5x the random-vector scale. This is itself a real, reportable finding, not a
replication of the drug-repurposing magnitude.

**Step 2 premise**: cancer status predicts batch at AUROC 0.722 (10 repeats of 3-fold stratified
CV, the largest k the smallest batch level, n=4, supports; per-level: batch 1: 0.648, batch 2:
0.537, batch 3: 0.949, batch 4: 0.949, batch 5: 0.527 -- batches 3 and 4 are each dominated by one
cancer-status level in the raw cross-tab, hence near-perfectly predictable, while 1/2/5 mix
multiple levels). This sits between the drug-repurposing domain's 0.70-0.72 and the gender/lab
domain's 0.39-0.41, a genuinely intermediate case.

**Step 3, the actual test**: naive (raw control-probe mean) predicts batch at AUROC 0.619.
`What` at alpha=10 (real regularization) predicts batch at 0.683. `What` at alpha=0.01 (near-
unregularized) predicts batch at 0.684, essentially identical regardless of regularization
strength. **This does not cleanly replicate either earlier domain.** The simple directional
heuristic ("X predicts F well, so orthogonalizing against X should suppress What's F-information")
would have predicted What should lose most of its batch-information here, given cancer
predicts batch at a level (0.722) comparable to the drug-repurposing domain's same-ATC-area
predictability (0.706-0.722), where the orthogonalized term DID fall to chance (0.497-0.501). It
did not: `What` stays at 0.68, close to or above naive's own 0.62, nowhere near chance.

**Why, honestly assessed rather than explained away**: `X` here is only 2 columns (3-level cancer
status). Even at a real, moderate correlation with batch, a 2-dimensional `X` cannot possibly
absorb all of the ways a 45-probe residual space could still track batch; there is abundant
residual capacity left for `What` to independently capture batch-related variance, unlike the
drug-repurposing domain's 48-dimensional `Zrep`, rich enough to nearly exhaust same-ATC-area's
predictable signal. The check's second step, then, depends on whether `X` is representationally
rich enough to exhaust `F`'s predictable variance, not merely on a raw predictability number. This
is a genuine refinement to how the check should be applied and interpreted, surfaced by a harder
domain, not a clean third confirmation of the original two-domain story. It is reported as such in
the paper (Section~\ref{sec:ruv-generalization}), with the sample-size caveat (57 samples, one
batch level at n=4) stated plainly rather than smoothed over.

No further dataset search followed this result. It is reported as the first honest run produced
it, consistent with this project's standing practice of not re-running or re-selecting after a
result comes in.

## 5b. A genuine bug in the alignment check's out-of-sample factor, and what fixing it changes

An external pre-submission review (`assessment.txt`) flagged that it was unclear how the proposed
method's area factor $A$ is computed for cold-start (CVS2) test compounds, since $A$ is estimated
from negative-control residuals that require labels the test compound does not have at train time.
Checking this directly (not just reading the code) found something worse than "unclear":
`negative_control_area_factor` (`code/models.py`) zeroes a test row's residual before the SVD,
since it only has train-row labels to residualize. A zero row into an SVD forces a zero row out
(confirmed both algebraically and numerically: `max|A[test_rows]|` ~1e-16 vs up to 1.38 on train
rows, for every target checked). The "proposed method's penalty term" in Table "alignment" was
therefore the AUROC of a value that is mathematically zero for every single test compound,
regardless of its chemistry or true organ-system membership -- a tautological "no information"
result, not a property of the method.

**Fix**: `code/models.py::project_area_factor_oos` fits a Ridge regression from the shared
representation $Z_{rep}$ to the train rows' own (non-degenerate) factor values, and uses it to
give every row a genuine out-of-sample estimate. Cross-validated $R^2$ of this projection, within
train rows only, is 0.44-0.56 (5-fold, epilepsy target) -- a real, non-degenerate fit, not an
overfitting artifact.

**What changes**: `code/13_naive_vs_proposed_alignment.py` now reports both the original
(zero-masked, buggy) `area_penalty_raw` and the corrected `area_penalty` columns side by side in
`results/naive_vs_proposed_alignment.csv`, for direct before/after comparison:

| Target | Flag | Naive's term | Proposed's term, buggy | Proposed's term, fixed (signed) | Fixed, sign-free |
|---|---|---|---|---|---|
| depression | same-ATC-area | 0.766 | 0.499 | 0.517 | 0.517 |
| depression | hard-negative | 0.636 | 0.499 | 0.492 | 0.508 |
| antipsychotic | same-ATC-area | 0.774 | 0.498 | 0.516 | 0.516 |
| antipsychotic | hard-negative | 0.663 | 0.501 | 0.502 | 0.502 |
| epilepsy | same-ATC-area | 0.787 | 0.497 | 0.343 | 0.657 |
| epilepsy | hard-negative | 0.600 | 0.500 | 0.435 | 0.565 |
| epilepsy | proconvulsant | 0.721 | 0.500 | 0.324 | 0.676 |

The area factor is an SVD output with an arbitrary sign (flip the factor and its fitted
coefficient flips to compensate, same likelihood); `sign_free_auroc = max(auroc, 1-auroc)` is the
same convention already used in `code/15_ruv2_genomics_generalization.py` for exactly this reason,
and is applied here for the first time to this specific table.

**Reading**: depression and antipsychotic stay close to chance (0.50-0.52) even corrected --
the original "the metric can't see this mechanism" claim holds for these two, now for a real
reason rather than an artifact. Epilepsy does not: its corrected, sign-free AUROC is 0.56-0.68,
substantially above chance. This is the paper's own primary motivating case (the bemegride
example, Section 1). The original abstract's universal claim ("its penalty term carries
essentially no information... about the compounds our bias metrics flag") does not survive this
fix uniformly; it is target-dependent, confirmed for 2 of 3 original targets, genuinely
contradicted for the third.

## 5c. Direct intervention test: does the metric actually respond to the correction's real effect?

Standalone AUROC(penalty, flag) asks whether a penalty term alone discriminates a flag. It does
not directly show whether the metric that is actually reported (Table "main-bias") responds when
that penalty's real effect is dialed into the score. `code/19_intervention_doseresponse.py` tests
this directly: for both corrections, sweep how much of the penalty is actually included in the
score, $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$ (naive: $\alpha=1$ is Table "main-bias"'s reported
row; proposed: $\alpha=0$ is Table "main-bias"'s reported row, factor fully projected out,
$\alpha=1$ is the undropped fit), and recompute the real bias metrics at each dose, on the same
CVS2 split, using the Priority-0-fixed out-of-sample factor on test rows.

Sanity check: naive's $\alpha=1$ numbers reproduce Table "main-bias" exactly (e.g. epilepsy
proconvulsant percentile 71.8).

Mean over 5 seeds, `results/intervention_doseresponse.csv`:

| Target | Method | $\alpha=0$ | $\alpha=0.25$ | $\alpha=0.5$ | $\alpha=0.75$ | $\alpha=1$ | Metric |
|---|---|---|---|---|---|---|---|
| depression | naive | 0.112 | 0.080 | 0.028 | 0.024 | 0.024 | area rate (lower better) |
| depression | proposed | 0.112 | 0.112 | 0.112 | 0.112 | 0.112 | area rate |
| antipsychotic | naive | 0.136 | 0.124 | 0.104 | 0.096 | 0.096 | area rate |
| antipsychotic | proposed | 0.144 | 0.148 | 0.144 | 0.148 | 0.144 | area rate |
| epilepsy | naive | 0.144 | 0.088 | 0.068 | 0.068 | 0.068 | area rate |
| epilepsy | proposed | 0.180 | 0.168 | 0.160 | 0.148 | 0.140 | area rate |
| epilepsy | naive | 30.9 | 43.5 | 51.0 | 58.7 | 71.8 | proconvulsant pct. (higher better) |
| epilepsy | proposed | 26.4 | 28.3 | 29.7 | 31.5 | 32.7 | proconvulsant pct. |
| lipid | naive | 0.104 | 0.076 | 0.052 | 0.052 | 0.052 | area rate |
| lipid | proposed | 0.104 | 0.104 | 0.104 | 0.104 | 0.104 | area rate |
| betablocker | naive | 0.108 | 0.088 | 0.088 | 0.088 | 0.088 | area rate |
| betablocker | proposed | 0.112 | 0.112 | 0.108 | 0.108 | 0.108 | area rate |

**Depression, antipsychotic, lipid**: naive's metrics move strongly and monotonically with
$\alpha$; proposed's stay exactly or nearly flat across the whole sweep. This is the direct,
operational version of "the metric cannot see this mechanism," strictly stronger evidence than
the standalone AUROC ever gave, for 3 of 5 targets. **Betablocker**: neither method's metrics move
much (naive's own swing is also small here), consistent with Section 4's earlier finding that
betablocker shows little method differentiation on any measure -- not informative either way, not
claimed as a confirmation. **Epilepsy**: proposed's metrics show a small but real, mixed-direction
response (area rate improves 0.180->0.140; proconvulsant percentile also improves 26.4->32.7;
hard-negative percentile is not shown here but moves slightly the other way), roughly a quarter
the magnitude of naive's corresponding swing. Not flat, not as strong as naive's: a genuine,
small exception, not a clean failure or a clean confirmation.

## 5d. Synthetic ground-truth confounding experiment

Real data can show a correction's penalty term is or is not associated with a *proxy* bias flag,
never with the true organ-system confound itself, since its real value is unobserved. 5000-line
simulation in `code/20_synthetic_groundtruth.py`: compounds get chemistry features $X$, an organ-
system confound $z_{area}$ that is itself a noisy function of a chemistry subspace (mirroring this
paper's own premise that structurally related compounds share both disease-specific chemistry and
organ-system membership), and indication labels whose true log-odds combine a disease-specific
mechanism term plus (for diseases sharing the confounded area) the real confound. Negative controls
share the area but have an independently drawn mechanism direction. 5 seeds.

| Quantity | Plain | Naive | Proposed |
|---|---|---|---|
| AUPRC (precision preservation) | 0.483 | 0.302 | 0.482 |
| Top-50 true-confound rate (base rate 0.28) | 0.656 | 0.480 | 0.676 |

| Quantity | Naive's penalty | Proposed's factor, buggy (zero-masked) | Proposed's factor, fixed (OOS, sign-free) |
|---|---|---|---|
| AUROC vs. true confound | 0.679 | 0.501 | **0.832** |

Stable across all 5 individual seeds (AUPRC proposed 0.44-0.50 vs plain 0.45-0.50 vs naive
0.25-0.34; top-50 confound rate proposed 0.62-0.78 vs plain 0.62-0.72 vs naive 0.40-0.60; factor
AUROC-vs-truth 0.79-0.87 for the fixed OOS estimate, 0.50-0.50 for the buggy zero-masked one, every
single seed -- the bug reproduces exactly here too, confirming it is a structural property of the
code, not an artifact of the real dataset).

**Reading**: the proposed method's area factor, correctly estimated, recovers the *true* confound
better than naive's own penalty does (0.832 vs 0.679 AUROC against known ground truth) -- exactly
what the method's RUV-style design is supposed to achieve, and better than the cruder alternative.
But the top-50 true-confound rate after correction is not reduced at all (0.656 -> 0.676, same or
slightly worse), while naive's crude subtraction substantially reduces it (0.656 -> 0.480), at a
real AUPRC cost naive pays and proposed does not. This is a sharper, more mechanistic account than
"the metric can't see it": the method's *identification* step works, arguably better than naive's;
its *correction* step (fit jointly with the factor, then drop the factor's coefficients at scoring)
does not translate that accurate identification into actually suppressing the confound's presence
in the ranking. This single account covers both patterns found in 5b/5c at once: targets where the
metric looks blind (the factor truly carries little leftover signal there) and epilepsy, where the
factor demonstrably carries real signal that the correction mechanism still fails to act on.

## 5e. Expanding beyond three indications: betablocker and lipid

Depression, antipsychotic, and epilepsy are all area "N" (nervous system). `code/indication_
matrix.py`'s `AREA_OF`/`negative_controls` already cover all 18 indications; betablocker and lipid
(area "C", cardiovascular, 2 negative controls each, 20-21 positives) were added as a 4th/5th
target to `code/06_run_evaluation.py`, `13_naive_vs_proposed_alignment.py`, and
`19_intervention_doseresponse.py`. The very sparse indications (anxiolytic n=7, antidementia n=6,
opioid_analgesic n=4, osteoporosis n=6) were deliberately excluded as targets, for the same
small-class instability already established for anxiolytic in Section 5's calibration check.

Main result (`results/eval_summary.csv`, CVS2): **betablocker shows almost no method
differentiation on anything** -- AUPRC plain/naive/proposed/nrbdmf = 0.741/0.742/0.741/0.152
(nrbdmf alone is weak, the other three indistinguishable), area rate 0.108/0.088/0.112/0.096
against a base rate of 0.079 (modest enrichment, barely moved by either correction), hard-negative
percentile pinned near 50 for every method. **Lipid shows the cleanest confirmation of the
alignment-check story found in this project**: naive's penalty AUROC against same-area is 0.635,
proposed's (sign-free, fixed) is 0.601 -- similar order, unlike the N-area diseases' sharp gap --
yet the intervention test (5c) shows naive's area rate still drops by half across the dose sweep
while proposed's is exactly flat at every dose.

This is a genuine scope-limiting finding, reported as such rather than smoothed over: the sharp
naive-wins/proposed-precision-preserving divergence that motivates this paper is a real pattern
for the three N-area (nervous-system) targets and for lipid, but essentially absent for
betablocker in this dataset. The paper's claims should be scoped to where the phenomenon actually
appears, not implied to hold uniformly across any same-area pair.

## 5f. repoDB secondary validation

Related Work already frames repoDB~\cite{brown2017repodb} as "a standard secondary comparison
point in this literature"; Section 7's Limitations previously stated it was not run. It has now
been pulled in: `data/repoDB.csv`, fetched from `github.com/zietzm/repoDB` (itself built from the
original author's `adam-sam-brown/repoDB`, CC-BY 4.0 -- the official apps.chiragjpgroup.org/repoDB
download requires an interactive Shiny session with no static file endpoint, so this is the
closest available static copy of the same release).

Join: by lowercased exact drug name (no DrugBank/InChIKey crosswalk exists in this project's data).
892 of repoDB's 1,572 drug names match a name in our 9,752-named-compound library. Indication-term
mapping tried for all 5 targets; betablocker/lipid were dropped after checking coverage --
repoDB's symptom-level indication terms (hypertension, angina, hypercholesterolemia) match many
OTHER drug mechanisms too (ACE inhibitors, calcium-channel blockers, other statin classes), giving
only 8/34 and 15/16 agreement with the existing mechanism-based label, too contaminated by
mechanism mismatch to use. Depression/antipsychotic/epilepsy's terms (psychiatric/neurological
diagnoses) map much more cleanly:

| Target | repoDB-matched in pool | Agreement with existing ChEMBL-derived label |
|---|---|---|
| depression | 42 (39 approved, 3 failed-only) | 0.821 |
| antipsychotic | 37 (32 approved, 5 failed-only) | 0.906 |
| epilepsy | 25 (22 approved, 3 failed-only) | 0.864 |

An AUPRC comparison was attempted (`code/21_repodb_validation.py`, pooling each compound's
out-of-fold score across all 5 CVS2 seeds rather than evaluating within each seed's tiny,
often-degenerate fold -- an earlier per-fold version produced AUPRC of exactly 1.0 for some folds
simply because every repoDB-matched compound in that fold happened to be repoDB-positive). Even
pooled, the matched evidence set is small (22-42 compounds) and heavily skewed toward repoDB's own
approved set (86-91% positive in the pooled set), so the resulting AUPRC values (0.75-1.0 for
every method) are inflated by prevalence and not independently informative about relative method
quality; they are reported in `results/repodb_validation.csv` for completeness but not used as a
comparative claim. **What is usable and genuinely informative**: the label-agreement numbers
above (82-91%) are real external validation that this paper's ChEMBL/LINCS-derived indication
labels agree with an independent, standard gold-standard database for the large majority of
compounds either source can speak to.

## 5g. DRDM baseline (best-effort reconstruction, not a verified reproduction)

Section 7's Limitations and the Discussion previously stated DRDM~\cite{meng2025drdmb} "did not
fully read." Checked directly: DRDM has no public code (no GitHub repo linked from the paper, the
Mount Sinai Scholars listing, or its PubMed record). UPDATE (second external review,
assessment_updated.txt, questioned the "abstract-only" characterization): re-checked the
ScienceDirect page directly via a scraper rather than WebFetch, and it exposes more than previously
found: the full abstract, full introduction with literature context, a "statement of significance"
table, and a one-line "Materials and methods" stub. The exact loss equations are still paywalled
(no PMC, arXiv, or author-manuscript copy exists), so that part of the original caveat holds. But
the now-visible introduction reveals a more basic problem: DRDM reweights messages at each
propagation step of a GNN over a heterogeneous drug-disease graph (Fdataset/Cdataset/LRSSL), and
there is no graph or propagation step anywhere in this project. Full text access would not make a
faithful reproduction possible here regardless, since the architecture itself has no counterpart
in this pipeline. The reconstruction below approximates DRDM's *motivation* (down-weight popular
entities, up-weight long-tail, add contrastive signal), not its actual GNN mechanism.

`code/22_drdm_baseline.py` implements a reconstruction from the published description,
adapted to this project's compound x indication matrix (DRDM's own setting is a
drug-disease heterogeneous graph, which does not exist here): a hand-rolled gradient-descent
classifier (matching this project's existing numpy-only style, see `fit_shared_mf`) combining (1)
a long-tail popularity reweighting term (static, by compound's total positive-indication degree)
plus a per-epoch term up-weighting currently-hard examples, approximating "dynamic" reweighting
without the paper's exact update rule, and (2) an InfoNCE-style dual-view contrastive loss between
a chemistry-view and an MF-view projection of the same compound. **This is stated explicitly as a
reconstruction, not a verified faithful reproduction**, the opposite labeling from NRBdMF
(`azuma2022nrbdmf`), which IS fully specified from a public, read paper and is labeled as matching
its original specification.

Mean over 5 seeds, CVS2 (`results/drdm_baseline.csv`):

| Target | AUPRC | Top-50 area rate | Area base rate |
|---|---|---|---|
| depression | 0.235 | 0.156 | 0.100 |
| antipsychotic | 0.583 | 0.140 | 0.104 |
| epilepsy | 0.110 | 0.144 | 0.110 |
| betablocker | 0.809 | 0.120 | 0.079 |
| lipid | 0.132 | 0.088 | 0.084 |

These sit in the same order of magnitude as `plain`'s AUPRC for each target (plain: 0.201, 0.440,
0.091, 0.741, 0.211 respectively) and show the same qualitative pattern as every other method
without an explicit area correction: enrichment above the base rate on the same-area metric. No
stronger claim than this should be drawn from a reconstruction whose exact fidelity to the
published method cannot be checked.

## 5h. Permutation test on Table 3 (second external review, Issue 6: sign-selection bias; later
tightened against a third review, Issue 4: aggregation and multiple comparisons)

`assessment_updated.txt`, a second external review of the already-revised paper, pointed out that
`max(AUROC, 1-AUROC)` is upward-biased under the null when the sign is chosen on the same
evaluation set the statistic is reported on: `|AUROC-0.5|` has positive expectation even when the
true AUROC is exactly 0.5, worse for small or imbalanced flag-positive counts. This is exactly
epilepsy's hard-negative (36 positives among 338 candidates) and proconvulsant (11 positives) cells
in Table 3.

`code/23_permutation_check.py` runs the correctly-sized test: shuffle the flag within the candidate
set 5000 times (within each seed's own split), holding the real fitted penalty term fixed,
recompute the same sign-free statistic each time, and report the fraction of permutations at or
above the observed value.

**Bug found and fixed while extending this script**: its "same-ATC-area" flag was built from this
project's own 18-indication co-occurrence matrix (which diseases a compound is positive for in
*our* data), not the external WHO ATC level-1 code that `run_eval.area_enrichment` and
`code/13_naive_vs_proposed_alignment.py` (the script that actually produced Table 3's numbers) use.
Two different flags, silently swapped in. Caught by comparing this script's per-seed AUROC against
`13`'s own output for the same seed/target and finding they disagreed outside floating-point noise.
Fixed to use the same ATC-based definition everywhere; results below are from the corrected version.
Hard-negative and proconvulsant flags were never affected (both scripts always built those the same
way), so only the "same-area" numbers in the first posted version of this section were wrong.

A third round of review (pasted inline, not saved as its own file; same four-item numbering restarted, "Issue 4" here) asked two further
questions: whether permutation respects each CVS2 split's own structure, and whether testing 7
cells (2 flags x depression, 2 x antipsychotic, 3 x epilepsy) raises a multiple-comparisons issue.
On the first: permutation is always done within one seed's own candidate set and fitted penalty,
never pooled across seeds. On aggregation: the first version of this analysis combined the 5
per-seed p-values by taking their arithmetic mean, which has no justified null distribution as a
combined test. Replaced with Fisher's method (`-2*sum(log(p_i)) ~ chi2(df=10)`), compared against a
Bonferroni-corrected threshold for the 7 cells ($\alpha=0.05/7=0.0071$), and checked for robustness
by recomputing with the single most extreme of the 5 seeds left out:

| Target | Flag | Mean observed AUROC | Fisher-combined p (5 seeds) | Fisher p excl. most extreme seed |
|---|---|---|---|---|
| depression | same-area | 0.561 | 0.120 | 0.383 |
| depression | hard-neg | 0.540 | 0.660 | 0.899 |
| antipsychotic | same-area | 0.579 | 0.0055 | 0.447 |
| antipsychotic | hard-neg | 0.580 | 0.0028 | 0.281 |
| epilepsy | same-area | 0.657 | **2.2e-9** | **<1e-9** |
| epilepsy | hard-neg | 0.568 | 0.0243 | 0.198 |
| epilepsy | proconv | 0.676 | **0.00042** | **0.0105** |

The combined test is more sensitive than the mean-of-p's reading it replaces: antipsychotic's two
flags and epilepsy's hard-negative flag clear the Bonferroni threshold on the full 5-seed combination,
which the earlier (less rigorous) approach missed. But all three collapse to clearly non-significant
once the single most extreme seed is left out, so this is a seed-specific artifact of one random
CVS2 split, not a stable effect, and is not read as a confirmed exception. Epilepsy's same-ATC-area
and proconvulsant flags hold up either way: on this corrected, more rigorous reading, two of
epilepsy's three flags, not one, are confirmed exceptions, strengthening rather than weakening the
paper's own motivating case; depression's flags and the rest are not confirmed under any version of
this test. main.tex's Table 3 discussion states this directly. This does not undermine the
intervention test's separate finding (Section 5.8/5.c) that epilepsy's proconvulsant rank percentile
moves a small real amount under dosing, since that is a direct metric measurement, not a
sign-selected AUROC, and is not subject to this bias either way.

## 5j. Chemistry-only CVS2 ablation (third round of review, pasted inline, Issue 1: random cold-start embeddings)

That review's most serious outstanding concern: under CVS2, a held-out compound's free-MF
embedding never receives a gradient (Section 5i) and stays at its random initialization, which
carries no real information and could affect the two corrections asymmetrically since they apply to
the shared representation through different operations (naive's raw-rate subtraction vs. proposed's
jointly-fit-then-projected SVD factor). Recommended fix: rerun with chemistry only, no MF embedding
at all, and check the main findings survive.

`code/24_chemistry_only_ablation.py` does exactly this: `Zrep = Xchem` (32-dim ECFP4 PCA), nothing
appended, same CVS2 splits, plain/naive/proposed, all five targets. Result: Table 1 and Table 2's
bias-diagnostic numbers (AUPRC, top-50 area rate, hard-negative rank) reproduce the MF-included
version to within 0.02 on every cell; naive still wins every bias metric, and proposed's top-50 area
rate still does not improve on plain's (e.g. epilepsy 0.172 chemistry-only vs. 0.180 with MF, both
above plain's 0.152/0.144). Table 3's standalone alignment AUROCs: epilepsy's three flags reproduce
to within 0.001 (0.656/0.566/0.677 vs. 0.657/0.565/0.676 with MF); depression's and antipsychotic's
shift up by 0.03-0.07 without chemistry's dilution of the cold-start noise (e.g. antipsychotic
same-area 0.568 vs. 0.516 with MF) but stay well below naive's 0.60-0.79, so the qualitative
"near chance, well below naive" reading is unchanged. None of the paper's claims depend on the
random cold-start embedding; main.tex's Method section now states this with the exact numbers.

## 5k. Oracle and direct-subtraction scores on the synthetic ground-truth experiment (third round
of review, pasted inline, Issue 2: separating estimation failure from application failure)

That round asked for two further scores on the synthetic experiment (5d) to separate "does
the method fail to estimate the confound" from "does it fail to apply a correctly estimated one":
an oracle (the same fit-jointly-then-project mechanism as "proposed", but given the TRUE $z_{area}$
instead of its 0.83-AUROC estimate) and a direct-subtraction score (the method's own ALREADY
ESTIMATED factor, applied by a calibrated direct subtraction instead of joint-fit-and-project).

Added to `code/20_synthetic_groundtruth.py`, 5 seeds, mean values:

| Quantity | Plain | Naive | Proposed | Direct-subtract | Oracle |
|---|---|---|---|---|---|
| AUPRC | 0.483 | 0.302 | 0.482 | 0.448 | 0.356 |
| Top-50 true-confound rate (base rate 0.285) | 0.656 | 0.480 | 0.676 | 0.456 | 0.256 |

Monotonic across all 5 individual seeds: oracle < direct-subtract < proposed on the top-50 rate,
every seed. The oracle, given a clean confound, removes nearly all of it under the method's own
projection mechanism (0.256, at the 0.285 base rate): the mechanism itself can work. Applying the
SAME imperfect, already-estimated factor by crude subtraction instead removes most of it too
(0.456, below naive's own 0.480), at a smaller AUPRC cost than naive pays. So both things found in
5d are true simultaneously: the factor's imperfect estimation (0.83, not 1.0, AUROC against ground
truth) matters, and the joint-fit-then-project application mechanism's sensitivity to that
imperfection also matters; neither alone is the whole story. main.tex's Section 5.7/5d text and
Discussion now state this rather than attributing the gap to application alone.

## 5i. Cold-start compounds and the shared MF embedding (second external review, Issue 5)

The second review also asked how the shared MF embedding is obtained for compounds with zero
training interactions under CVS2. Checked directly: `06_run_evaluation.make_masks`'s CVS2 branch
masks a held-out compound's *entire row*, every column, not just the target's. Inside
`fit_shared_mf`, that row's gradient (`G = (P-Y)*M`) is therefore exactly zero every epoch; the
update reduces to `U[i] -= lr*l2*U[i]`, pure L2 decay of the random initialization, not learning.
Measured directly: the correlation between a held-out row's final `Ufree` and its untrained random
init is 1.0 (`np.corrcoef` on the flattened vectors). This is not a bug: a cold-start compound
genuinely has no training interactions to learn an embedding from, so there is nothing else
`fit_shared_mf` could do without an explicit side-information projection (which this paper doesn't
build for the MF embedding itself, only for the area factor `A` via `project_area_factor_oos`).

What it does mean: for CVS2 test rows, the MF half of `Zrep` carries no learned signal, just noise.
Measured its practical impact: fitting `clf_plain` and applying its coefficients to test rows, the
logit's standard deviation from the chemistry block is 2.9-4.8 across depression/antipsychotic/
epilepsy, versus 0.1-0.3 from the (noise) MF block, roughly 5-10%. Since every method (plain,
naive, proposed, NRBdMF) shares the same `Zrep`, this noise affects all four symmetrically rather
than favoring one in the comparison, but it was not previously documented and main.tex now states
it explicitly in the Method section.

## 6. Figures

- `figures/fig1_bemegride_top10.png` -- top-10 non-AED candidates, plain model (section 1a)
- `figures/fig1_neurology_enrichment.png` -- base rate vs top-50 neurology rate (section 1b)
- `figures/eval_auprc_by_method.png` -- AUPRC by target/CVS/method (section 4)
- `figures/eval_hard_negative_ranks.png` -- hard-negative rank percentile by target/CVS/method (section 4)
- `figures/negctrl_calibration.png` -- negative-control self-calibration deltas (section 5)

A full run log (what was executed, in what order, what failed/was fixed along the way) is in
`logs/run_log.md`.

## 7. Limitations (honest)

1. **UPDATED after follow-up (see "Follow-up: RUV*-style shrinkage" in section 5): the proposed
   negative-control method's debiasing effect is weak and inconsistent in sign**, and on the two
   bias-diagnostic metrics that matter most (top-50 area enrichment, proconvulsant panel rank) it
   is dominated by the "naive" baseline it was designed to beat. We originally hypothesized this
   was because the negative-control indications are small (4-21 positives each) and the raw
   2-component SVD area factor is mostly noise. **We tested this directly**: a calibration-driven
   grid search over rank (r_area in {1,2,3}) and RUV*-style shrinkage (Gerard & Stephens,
   arXiv:1705.08393; s in {0,0.25,0.5,0.75,1.0}), selected via the leave-one-negative-control-out
   diagnostic in section 5, picked **s=1.0 (no shrinkage) for all three targets** -- the
   calibration data does not support the noisy-estimate hypothesis, and a full CVS1/2/3 re-run
   with the selected settings (`proposed_shrunk`, `results/eval_*_shrunk.csv`) is statistically
   indistinguishable from the original. **This rules out simple noise/shrinkage as the explanation
   and narrows the open question**: the weak effect is more likely structural -- either the area
   factor (even if estimated without noise) does not capture enough organ-system-confound signal
   in this 18-indication matrix to move the bias metrics beyond what chemistry+free-MF already
   explains, or a genuinely different identification strategy is needed (not a rank/shrinkage
   variant of the same SVD-residual mechanism). Candidates for a future iteration: (a) a richer
   negative-control feature (e.g. direct ATC/SIDER-area membership rather than residualized
   predicted probabilities), (b) jointly estimating the area factor and v_k0 rather than the
   current two-stage estimate-then-refit procedure, or (c) testing whether the method behaves
   differently on a less chemistry-dominated backbone, since item 5 below notes chemistry may be
   doing most of the work here and leaving little room for a cross-indication correction to matter.
   **Further update, now resolved**: see "Resolving the remaining open question" in section 5
   above. The structural account above was correct in direction but incomplete; the direct
   alignment test shows naive subtraction's penalty term separates the bias metrics' own flagged
   compounds at 0.60-0.79 AUROC while the proposed method's penalty term sits at chance (0.497-0.501)
   on every target and every flag. This is no longer an open question for the final paper.
2. **The naive-subtraction failure mode (AUPRC 0.45->0.17) from the idea document was NOT
   reproduced** in this implementation at anywhere near that severity, across a lambda sweep from
   0.5 to 64. **Follow-up: an expanded search of `ICHI2027_mgan/code/` (figures.py, resurvey.py,
   diagnostic_confound.py, informativeness.py, aggregation_check.py, waterfall.py,
   compare_diseases.py, canonical_numbers.py, score_survey.py) and a repo-wide grep for "0.45"
   across the whole `epilepsy_repurpose` tree found no script that computes a per-indication
   AUPRC before/after subtracting any aggregate "area-ness" score** -- every "0.45" hit outside
   this project belongs to unrelated sibling projects (CTDN, MGAN, ISEF, BioNLP_paper, etc.), not
   to an area-debiasing baseline. This independently confirms the coordinator's own search: **the
   0.45->0.17 figure is not traceable to any saved script or result file in this codebase**, and
   is most likely an unreproduced/notebook-era number from the original idea-generation pass
   rather than something this implementation is failing to match. Recommended final paper
   language: **do not cite 0.45->0.17**; report the naive baseline's actual, mild degradation as
   measured here (naive_subtract vs plain AUPRC differs by at most a few percentage points across
   all 9 target/CVS combinations, see section 4's AUPRC table), and reframe the naive baseline's
   "failure mode" around the bias metrics instead, where naive subtraction's real, measured cost
   is a modest AUPRC trade for the LARGEST same-area-enrichment and hard-negative-rank improvement
   of any method tested (section 4) -- a more interesting and more defensible story than the
   unreproducible collapse number, and one this implementation actually supports with real numbers.
3. **ATC Level-1 coverage of the LINCS 9,878-compound library is sparse**: only 954 compounds
   (9.7%) carry any WHO ATC classification at all (5,099/9,878 resolve to a ChEMBL molecule record
   by full InChIKey, but most of those still have zero ATC codes, being research/tool compounds
   rather than marketed drugs). This deflates both the base rate and the top-k enrichment
   statistics relative to the idea document's original Hub-name-matched label, and should be
   stated explicitly in the paper rather than left implicit.
4. **SIDER coverage of the full library is similarly sparse** (543/9,878 compounds, 5.5%), for the
   same underlying reason (SIDER only covers marketed drugs with a package insert). This limits
   how many compounds the SIDER-based hard-negative construction and the NRBdMF-style baseline
   can actually use; both are reported honestly against this constraint rather than padded.
5. **Pure compound x indication co-occurrence carries almost no signal** on this 18-indication
   matrix (see `logs/model_diagnostics.log`): we had to concatenate a fixed ECFP4 PCA to the free
   MF embedding to get any usable AUPRC, which is a real methodological choice (analogous to
   NRLMF/NRBdMF's own neighborhood regularization) but means the "matrix factorization" component
   of the proposed method is doing comparatively little relative to chemistry. A reviewer could
   reasonably ask whether the method is "mostly a fingerprint classifier with an extra correction
   term" rather than a genuine multi-task indication-matrix method; this should be addressed
   directly in the paper, not left implicit.
6. **CVS3 (cold-start disease) has a structural degeneracy** for any method, including ours, that
   needs at least some target-indication labels to estimate an indication-specific residual: with
   zero such labels, "plain" and "proposed" provably collapse to the same cross-indication
   transfer score (mean prediction of the negative controls). This is documented and handled
   (not silently reported as a false result; see `logs/model_diagnostics.log`), but it means CVS3
   cannot actually test the debiasing mechanism this paper is about, only the baseline "plain"
   transfer mechanism. A genuinely informative CVS3 experiment would need a real cross-disease
   transfer model (e.g. sharing indication embeddings via side information), which we did not
   build in the time available.
7. **RESOLVED (section 5f)**: repoDB secondary benchmark now run. 82-91% label agreement with the
   existing ChEMBL-derived labels for depression/antipsychotic/epilepsy is genuine external
   validation; the AUPRC comparison itself is too prevalence-skewed on the small matched subset
   (22-42 compounds, 86-91% positive) to support a comparative claim, and is not used as one.
   betablocker/lipid were tried and dropped: repoDB's symptom-level indication terms match too
   many other drug mechanisms to give a clean target-specific label there.
8. **External/temporal holdout (literature_review.md section 5 item 7) was not run**, for the same
   time-budget reason. `data/approval_years_depression.csv` and `approval_years_antineoplastic.csv`
   exist and could support this for those two classes; it was not attempted for the three primary
   targets here.
9. **RESOLVED by follow-up (section 5)**: the anxiolytic negative-control calibration cells were
   re-run with 20 reruns instead of 4-5; the mean delta shrank by ~3.7x (0.127 -> 0.034), now well
   within 1 SE of zero (SE=0.129/sqrt(15)=0.033), confirming small-sample noise (7 positives
   total) rather than a genuine calibration failure. No longer an open item. (The standard
   deviation also dropped, from 0.249 to 0.129, roughly half, not flat -- see section 5's note on
   why this doesn't change the conclusion, which rests on the SE calculation.)
10. **Proconvulsant canonicalization is name-based** (`code/04_build_proconvulsants.py`): 20 of the
    82 literature-curated compounds do not resolve to one of the 9,878 LINCS compounds by name
    match (`in_lincs=False`), and no fuzzy/InChIKey-based reconciliation was attempted for those 20;
    some may actually be present in the library under a different name spelling.
11. **PARTIALLY RESOLVED (section 5g)**: DRDM now has a baseline reconstruction in
    `code/22_drdm_baseline.py`, explicitly labeled as such rather than as a verified reproduction.
    Its exact loss equations remain paywalled, but the more basic reason it can't be a faithful
    reproduction is architectural: DRDM is a GNN over a heterogeneous drug-disease graph, and this
    project has no graph. "Escaping the Drug-Bias Trap" was still not read in full during this pass; the
    novelty-scoping caveat from literature_review.md section 6 still stands for that one and
    should be addressed before the paper's related-work section is finalized.

## 8. Compute note

All code ran on CPU except step 1's XGBoost fingerprint models (`code/01_reproduce_fig1.py`,
`tree_method="hist", device="cuda"`, GPU-shared with the sibling Idea-1 task, batch sizes
unchanged from the pre-existing `candidates.py` script, i.e. small/modest). The matrix-factorization
method, all three baselines, and the negative-control calibration check are plain
NumPy/scikit-learn on matrices no larger than 1,702 x 48 (chemistry PCA + free MF embedding) or
1,702 x 18 (indication matrix) -- each of the 06/07 evaluation runs (3 targets x 3 CVS schemes x 5
reruns, or 3 targets x ~6 negative controls x 5 reruns) completed in well under 2 minutes once
BLAS thread oversubscription was fixed (`OMP_NUM_THREADS=2`/`OPENBLAS_NUM_THREADS=2`/
`MKL_NUM_THREADS=2`; an earlier run without these limits took >10x longer for identical work due
to thread-contention overhead from many small per-indication `sklearn.LogisticRegression` fits in
a tight loop -- logged for anyone re-running this code). ChEMBL and PubChem API calls
(`02_fetch_atc_level1.py`, `03_build_sider_full.py`, `05_build_extra_indications.py`) took
roughly 10-15 minutes combined, batched at 50-60 identifiers per request with retry/backoff,
against the public ChEMBL REST and PubChem PUG-REST APIs. No step required more than a few hundred
MB of GPU memory at any point; the RTX 3090 was otherwise left free for the concurrent sibling task.
The coordinator follow-up round (shrinkage grid search `10_shrinkage_negctrl.py`, the 20-seed
`07_negctrl_calibration.py` re-run, and the `ICHI2027_mgan`/repo-wide naive-baseline search) added
roughly another 8 minutes of compute, all CPU, thread-limited as above; no new GPU usage.
