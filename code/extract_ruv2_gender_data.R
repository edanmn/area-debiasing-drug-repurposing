# One-time data-extraction step for 15_ruv2_genomics_generalization.py.
#
# Source: Bioconductor package RUVnormalizeData (v1.32.0), data object `gender`, an
# ExpressionSet built by Laurent Jacob from Vawter et al. 2004 (Neuropsychopharmacology
# 29(2):373-384) and distributed with the original RUV-2 paper (Gagnon-Bartsch & Speed
# 2012, Biostatistics 13(3):539-552) as its worked example. 12600 genes x 84 post-mortem
# brain samples, 5 male / 5 female donors, hybridized across 3 labs and 3 brain regions.
# The `isNegativeControl` feature flag marks the 799 housekeeping probesets from
# Eisenberg & Levanon 2003 (Trends in Genetics 19(7):362-365) that Gagnon-Bartsch & Speed
# used as negative controls. Requires R with Bioconductor's Biobase package:
#   install.packages("BiocManager"); BiocManager::install("Biobase")
#   download.file("https://bioconductor.org/packages/release/data/experiment/src/contrib/RUVnormalizeData_1.32.0.tar.gz", "RUVnormalizeData.tar.gz")
#   untar("RUVnormalizeData.tar.gz")
#
# Writes only the 799 negative-control-gene rows (the full 12600-gene matrix is not
# needed by the Python analysis and is left out to keep the derived file small).

suppressMessages(library(Biobase))
load("RUVnormalizeData/data/gender.rda")

expr <- exprs(gender)
pheno <- pData(gender)
feat <- fData(gender)
is_nc <- feat[, 1]

write.csv(expr[is_nc, , drop = FALSE], "../data/ruv2_gender_controlgenes.csv")
write.csv(pheno, "../data/ruv2_gender_pheno.csv")
cat(sprintf("wrote %d negative-control genes x %d samples\n", sum(is_nc), ncol(expr)))
