# One-time data-extraction step for the bladderbatch alignment-check generalization (third
# domain, harder than RUV-2's own gender/lab dataset: multi-level cancer status (3 levels,
# not binary), multi-level batch confound (5 levels), and used here with a regularized (ridge)
# fit rather than unregularized OLS.
#
# Source: Bioconductor package bladderbatch (v1.50.0), Leek (2006 data; package DOI
# 10.18129/B9.bioc.bladderbatch), object `bladderEset`. 22283 Affymetrix HG-U133A probes x 57
# bladder samples, cancer status (Normal=8, Biopsy=9, Cancer=40) and batch (5 levels,
# sizes 11/18/4/5/19), used throughout the sva/ComBat/RUV tutorial literature as the standard
# worked example for batch-effect correction.
#
# Negative controls: the 45 Affymetrix bacterial/phage poly-A spike-in and hybridization
# control probes (AFFX-BioB/BioC/BioDn/CreX/DapX/LysX/PheX/ThrX/TrpnX and the r2-Bs-*/r2-Ec-*/
# r2-P1-cre series), which measure only technical hybridization variation, not human gene
# expression, and so carry no biological signal by construction. The remaining AFFX probes
# (actin HSAC07, GAPDH, ISGF3A, RGE, 18S/28S rRNA, hum_alu, and the ambiguous M27830 series) are
# excluded because they ARE real human gene products and could carry genuine biological signal
# correlated with cancer status, unlike a true negative control.
#
# Requires R with Bioconductor's Biobase package (already installed for the gender extraction):
#   download.file("https://bioconductor.org/packages/3.23/data/experiment/src/contrib/bladderbatch_1.50.0.tar.gz", "bladderbatch.tar.gz")
#   untar("bladderbatch.tar.gz")

suppressMessages(library(Biobase))
load("bladderbatch/data/bladderdata.rda")
eset <- bladderEset

expr <- exprs(eset)
pheno <- pData(eset)

all_probes <- rownames(expr)
nc_patterns <- c("^AFFX-BioB-", "^AFFX-BioC-", "^AFFX-BioDn-", "^AFFX-CreX-", "^AFFX-DapX-",
                  "^AFFX-LysX-", "^AFFX-PheX-", "^AFFX-ThrX-", "^AFFX-TrpnX-",
                  "^AFFX-r2-Bs-dap-", "^AFFX-r2-Bs-lys-", "^AFFX-r2-Bs-phe-", "^AFFX-r2-Bs-thr-",
                  "^AFFX-r2-Ec-bioB-", "^AFFX-r2-Ec-bioC-", "^AFFX-r2-Ec-bioD-", "^AFFX-r2-P1-cre-")
is_nc <- rep(FALSE, length(all_probes))
for (p in nc_patterns) is_nc <- is_nc | grepl(p, all_probes)
cat(sprintf("Negative-control (bacterial/phage spike-in) probes: %d\n", sum(is_nc)))

write.csv(expr[is_nc, , drop = FALSE], "../data/bladderbatch_controlgenes.csv")
write.csv(pheno, "../data/bladderbatch_pheno.csv")
cat(sprintf("wrote %d negative-control probes x %d samples\n", sum(is_nc), ncol(expr)))
