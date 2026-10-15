# Ranking within the therapeutic area

Code and results for a RECOMB 2027 submission testing whether negative-control indications
(diseases in the same organ system, treated by a different mechanism), following the logic of
removing unwanted variation in genomics (Gagnon-Bartsch & Speed 2012; Gerard & Stephens 2021) and
negative-control exposures in pharmacoepidemiology (Lipsitch et al. 2010), can remove therapeutic-area
confounding from drug-repurposing rankings without the precision cost a naive correction is assumed
to cause. The honest result: a naive area-score subtraction outperforms the proposed negative-control
correction on every bias-diagnostic metric tested, under two independently implemented identification
strategies (RUV-2-style and RUV-4-style).

**Status: not yet published.** This README and the local commit history exist so the repository is
ready to push in one step once the authors confirm it should be made public (see
`paper/main.tex`'s `AUTHOR ACTION REQUIRED` blocks for the two open decisions that gate this).

## Layout

- `code/` -- numbered pipeline scripts (`01`-`11`) covering motivating-evidence reproduction, data
  builds (ATC Level-1, full-library SIDER join, proconvulsant panel extraction), the four compared
  methods, evaluation, and two rounds of follow-up (shrinkage, RUV-4 joint estimation), plus shared
  modules (`common.py`, `models.py`, `indication_matrix.py`, `splits.py`, `hard_negatives.py`).
- `data/` -- derived data, several files symlinked into the sibling `ICHI2027_mgan` project for the
  base compound/label tables (see that project's `README.md` for rebuild instructions from public
  LINCS/ChEMBL sources); this project's own additions (`atc_level1.csv`, `sider_full.parquet`,
  `proconvulsants.csv`, the nine extra negative-control `labels_*.csv` files) are reproducible from
  the public ChEMBL REST API, PubChem PUG-REST, and the already-on-disk raw SIDER export.
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

## Before this repository is made public

- Replace the symlinked paths in `data/` with either the real files or a fetch script.
- Resolve the two `AUTHOR ACTION REQUIRED` items in `paper/main.tex`, including the proposed
  (not yet committed) related-work paragraph addressing overlap with a companion manuscript.
- Confirm no large raw intermediates are swept in by an unqualified `git add`.

## Reproducing

See `RESULTS.md` for exactly which script produced which file and the compute notes at the end of
each round for environment/GPU-sharing caveats.
