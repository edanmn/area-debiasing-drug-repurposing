"""CVS1/CVS2/CVS3 cross-validation masks over the compound x indication matrix, matched to the
NRLMF/NRBdMF convention (literature_review.md section 5):
  CVS1 = random pair (compound, indication) holdout
  CVS2 = cold-start drug: entire rows (compounds) held out
  CVS3 = cold-start disease: entire columns (indications) held out
Returns, for each fold, a boolean train_mask (N x K) and the list of (i, k0_target_col) test cells
to score for the target indication.
"""
import numpy as np


def cvs1_masks(N, K, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    cell_fold = rng.integers(0, n_folds, size=(N, K))
    for f in range(n_folds):
        train_mask = cell_fold != f
        test_mask = cell_fold == f
        yield f, train_mask, test_mask


def cvs2_masks(N, K, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    row_fold = rng.integers(0, n_folds, size=N)
    for f in range(n_folds):
        train_mask = np.tile((row_fold != f)[:, None], (1, K))
        test_mask = np.tile((row_fold == f)[:, None], (1, K))
        yield f, train_mask, test_mask


def cvs3_masks(N, K, n_folds=5, seed=0):
    rng = np.random.default_rng(seed)
    col_fold = rng.integers(0, n_folds, size=K)
    for f in range(n_folds):
        train_mask = np.tile((col_fold != f)[None, :], (N, 1))
        test_mask = np.tile((col_fold == f)[None, :], (N, 1))
        yield f, train_mask, test_mask


def scaffold_groups(cmp):
    """Murcko scaffold id per compound (fallback: own ik14, i.e. singleton group), for
    Tanimoto/scaffold-disjoint ECFP4-based splits per literature_review.md section 5 item 3."""
    import pandas as pd
    return pd.factorize(cmp.scaffold.replace("", np.nan).fillna(cmp.ik14))[0]
