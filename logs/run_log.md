# Run log: Idea 2 (area-debiasing) pipeline

All runs below are on real data; no numbers in results/ were hand-edited after generation except
for the clerical removal of scratch _table_*.md/_tmp_*.csv helper files from results/ (their
content was copied verbatim into RESULTS.md's tables, nothing was altered in transit).

1. code/01_reproduce_fig1.py -- bemegride + neurology enrichment reproduction. Ran successfully,
   GPU (XGBoost), ~1 min. Output: results/fig1_candidate_ranking_{epilepsy,depression,antipsychotic}.csv,
   results/fig1_neurology_enrichment.csv.
2. code/02_fetch_atc_level1.py -- full ChEMBL ATC fetch. Ran successfully in background, ~12 min
   (network-bound, ChEMBL REST API, batched 50 InChIKeys/request). Output: data/atc_level1.csv,
   data/_atc_class_full.csv (cache), logs/02_fetch_atc_level1.log.
3. code/03_build_sider_full.py -- PubChem CID resolution + SIDER join. Ran successfully in
   background, concurrently with (2) (different API, PubChem), ~10 min. Output:
   data/sider_full.parquet, logs/03_build_sider_full.log.
4. code/04_build_proconvulsants.py -- PROCONV extraction + cross-validation. Ran successfully,
   <5s (local only, depends on (3)'s output). Output: data/proconvulsants.csv.
5. code/05_build_extra_indications.py -- 9 additional ATC-defined indications. Ran successfully
   in background, ~15 min (network-bound, ChEMBL REST, one ATC-class + molecule-listing query set
   per new indication; immunosuppressant/L04 was the slowest, ~8 min alone). Output:
   data/labels_{parkinson,anxiolytic,antidementia,opioid_analgesic,raas_antihypertensive,antacid,
   osteoporosis,antimycotic,immunosuppressant}.csv, data/indication_area_map.csv,
   logs/05_build_extra_indications.log.
6. code/06_run_evaluation.py -- main CVS1/CVS2/CVS3 evaluation, 4 methods x 3 targets.
   FIRST attempt (indication-matrix-only MF, no chemistry): ran to completion but produced
   near-base-rate AUPRC for all methods -- diagnosed as a genuine data-sparsity finding, not a
   bug (see logs/model_diagnostics.log), and used to motivate adding chem_pca to the shared
   representation (models.py revision).
   SECOND attempt (with chem_pca + free MF): initially 1559% CPU / very slow due to BLAS thread
   oversubscription from many small per-column sklearn.LogisticRegression fits; killed and re-run
   with OMP_NUM_THREADS=2/OPENBLAS_NUM_THREADS=2/MKL_NUM_THREADS=2, completed in <2 min. Discovered
   (a) an NRBdMF train/test leakage bug (ignored train_mask) and (b) a CVS3 scoring degeneracy
   (all-zero score ties for plain/proposed); both fixed in models.py/06_run_evaluation.py (see
   logs/model_diagnostics.log for the exact fix), THIRD attempt with both fixes is the version
   whose numbers are reported in RESULTS.md and results/eval_*.csv.
7. code/07_negctrl_calibration.py -- negative-control self-calibration check. Ran successfully,
   ~1 min (thread-limited as above). Output: results/negctrl_calibration*.csv, figures/negctrl_calibration.png.
8. code/08_make_figures.py -- all figures from results/*.csv. Ran successfully, <5s, twice (once
   before the 06 bugfix rerun, once after; figures/ holds only the post-fix versions).
9. code/09_scaffold_disjoint_check.py -- scaffold-grouped CVS2 re-run per literature_review.md
   section 5 item 3. Ran successfully, ~1 min (thread-limited). Output:
   results/scaffold_disjoint_cvs2_{auprc,summary}.csv.

Not run (see RESULTS.md section 7 for why): repoDB secondary benchmark; external/temporal holdout
for the three primary targets; fuzzy/InChIKey reconciliation of the 20 proconvulsants that did not
resolve by name match; a lambda/rank/negative-control-pooling sweep for the proposed method beyond
the single lambda=2.0, r_area=2 configuration reported.

## Coordinator follow-up round (post-handback review)

10. code/10_shrinkage_negctrl.py -- RUV*-style shrinkage grid search + re-evaluation.
    FIRST attempt: selection criterion (minimize mean|calibration delta|) naively picked shrink=0.0
    for all three targets every time -- a trivial/vacuous result, since shrink=0 makes the area
    factor identically zero, which makes the "corrected" fit mathematically IDENTICAL to "plain",
    guaranteeing delta=0 exactly regardless of data. Caught this before accepting the result; fixed
    by excluding shrink=0 from the candidate pool and selecting the largest nontrivial shrink
    within 1 SE of the best nontrivial calibration. SECOND attempt (reported in RESULTS.md) selected
    shrink=1.0 (no shrinkage) for all three targets -- i.e. the calibration diagnostic does not
    support the noisy-estimate hypothesis. Ran in background, ~2 min (thread-limited). Output:
    results/shrinkage_grid_search.csv, results/eval_*_shrunk.csv.
11. code/07_negctrl_calibration.py -- re-run with SEEDS bumped from 5 to 20, specifically to
    settle the anxiolytic calibration cells. Ran successfully in background, ~3 min. Confirmed
    small-sample noise (delta shrank 0.127->0.034 as n went 4->15). Output: results/negctrl_calibration*.csv
    (overwrites the 5-seed version; the 5-seed numbers are superseded, not separately retained).
12. Searched ICHI2027_mgan/code/{figures,resurvey,diagnostic_confound,informativeness,
    aggregation_check,waterfall,compare_diseases,canonical_numbers,score_survey}.py and grepped
    "0.45" across the whole epilepsy_repurpose tree for the source of the claimed 0.45->0.17 naive
    AUPRC collapse. No matching script found; all "0.45" hits outside this project belong to
    unrelated sibling repos. Confirms the coordinator's own independent search. Conclusion:
    0.45->0.17 is not traceable to any script/result file and should not be cited in the paper;
    RESULTS.md section 7 item 2 gives the recommended replacement framing.
