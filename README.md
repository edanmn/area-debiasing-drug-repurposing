# An alignment check for debiasing corrections

Code and results for a RECOMB 2027 submission. The core contribution is a general validity check,
the alignment check, for deciding whether a bias-diagnostic metric can even detect what a given
debiasing correction removes: score the correction's own penalty term as a standalone classifier
of whatever the metric flags, and treat near-chance AUROC as evidence the metric cannot see that
mechanism at all, independent of the correction's substantive quality.

The check is developed through a case study testing whether negative-control indications (diseases
in the same organ system, treated by a different mechanism), following the logic of removing
unwanted variation in genomics (Gagnon-Bartsch & Speed 2012; Gerard & Stephens 2021) and
negative-control exposures in pharmacoepidemiology (Lipsitch et al. 2010), can remove therapeutic-area
confounding from drug-repurposing rankings without the precision cost a naive correction is assumed
to cause. The honest result: a naive area-score subtraction outperforms the proposed negative-control
correction on every bias-diagnostic metric tested, under two independently implemented identification
strategies (RUV-2-style and RUV-4-style). Applying the alignment check explains why: the proposed
method's penalty term is, by construction, exactly orthogonal to the representation its base model
is fit on, and the bias metrics' own flags turn out to be substantially predictable from that same
representation, so the metric was structurally unable to reward that correction's mechanism. We
re-ran both steps on a second domain, the gene-expression dataset RUV-2 was itself introduced on,
and the two domains diverge exactly where the check predicts they should.

## Layout

- `code/` -- numbered pipeline scripts (`01`-`15`) covering motivating-evidence reproduction, data
  builds (ATC Level-1, full-library SIDER join, proconvulsant panel extraction), the four compared
  methods, evaluation, two rounds of follow-up (shrinkage, RUV-4 joint estimation), the
  identifiability check, the alignment check itself (`13`, `14`), and its re-run on a second
  domain (`15`, `extract_ruv2_gender_data.R`), plus shared modules (`common.py`, `models.py`,
  `indication_matrix.py`, `splits.py`, `hard_negatives.py`).
- `data/` -- derived data, several files symlinked into the sibling `ICHI2027_mgan` project for the
  base compound/label tables (see that project's `README.md` for rebuild instructions from public
  LINCS/ChEMBL sources); this project's own additions (`atc_level1.csv`, `sider_full.parquet`,
  `proconvulsants.csv`, the nine extra negative-control `labels_*.csv` files) are reproducible from
  the public ChEMBL REST API, PubChem PUG-REST, and the already-on-disk raw SIDER export.
  `ruv2_gender_controlgenes.csv` and `ruv2_gender_pheno.csv` are extracted from the Bioconductor
  `RUVnormalizeData` package (see `code/extract_ruv2_gender_data.R` for provenance and the exact
  source and version).
- `results/` -- every CSV referenced in `RESULTS.md`.
- `figures/` -- plots referenced in the paper.
- `paper/` -- the LaTeX source (`main.tex`), bibliography (`refs.bib`), and compiled `main.pdf`.
- `RESULTS.md` -- the full, source-of-truth experimental write-up, including two full rounds of
  follow-up investigation into why the proposed method underperforms, and what was ruled out.
- `literature_review.md` -- the related-work research this submission's citations are drawn from,
  including the investigation into the provenance of the proconvulsant panel (see its section 3:
  the panel is literature-curated from a public source, Chen, Albertson & Olson 2016, not drawn from
  an external database, and is disclosed as such rather than presented as one).
- `DATA_PLAN.md` -- the dataset-provenance plan this project followed.

## Reproducing

See `RESULTS.md` for exactly which script produced which file and the compute notes at the end of
each round for environment/GPU-sharing caveats.
