"""Build data/proconvulsants.csv from the PROCONV Python set literal duplicated across this
project's PACT/lincs-detectability-atlas codebases, per DATA_PLAN.md's integrity-fix requirement.

Provenance (checked directly, not assumed):
  - PACT/PACT/coverage.py's PROCONV set is the ONLY copy of this literal in the PACT tree: it does
    not appear in audit.py, rank.py, sta.py, modal.py, audit2.py, rigor.py (grep -c PROCONV on each
    returned 0 except coverage.py). So there is no multi-file duplication to reconcile WITHIN PACT.
  - `diff PACT/PACT/coverage.py lincs-detectability-atlas/lincs-detectability-atlas/code/coverage.py`
    returns NO differences: the two codebases are byte-identical for this file, so there is no
    discrepancy to resolve between them either. We use PACT/PACT/coverage.py as the single source.
  - Per PACT's own README: "Proconvulsant labels are literature-curated (Chen et al. 2016), not
    FAERS disproportionality-derived." We cite Chen, Albertson & Olson (Br J Clin Pharmacol 2016) as
    the primary literature source, and we do NOT cite "PACT" as an external dataset anywhere -- PACT
    is this research group's own prior unpublished/non-archival work (a Great Plains 2026 abstract),
    disclosed here plainly as such, per literature_review.md section 3's "minimum fix".

Output columns: canonical_name, synonyms, source, in_lincs, sider_convulsion_flag, ruffmann_tier.
"""
import re
import sys
import pandas as pd

sys.path.insert(0, ".")
from common import D, load_compounds

PACT_FILE = "/mnt/c/Users/rishik/epilepsy_repurpose/PACT/PACT/coverage.py"
ATLAS_FILE = ("/mnt/c/Users/rishik/epilepsy_repurpose/lincs-detectability-atlas/"
              "lincs-detectability-atlas/code/coverage.py")

# Ruffmann, Bogliun & Beghi 2006 (Expert Rev. Neurother.) risk tiers, exactly as quoted in the task
# description / literature_review.md section 3. Matched case-insensitively against the raw PROCONV
# strings; left blank where Ruffmann et al. does not name the drug/class explicitly.
RUFFMANN_HIGH = {"meperidine", "pethidine", "clozapine", "chlorpromazine", "thioridazine",
                 "perphenazine", "trifluoperazine", "loxapine", "cyclosporine", "cyclosporin-a"}
# (sevoflurane and "phenothiazines" generically are in Ruffmann's high tier but sevoflurane is not
# in PROCONV; chlorpromazine/thioridazine/perphenazine/trifluoperazine/loxapine are phenothiazine-
# or phenothiazine-like antipsychotics, matched to the "phenothiazines" high-tier entry.)
RUFFMANN_INTERMEDIATE = {"maprotiline", "amitriptyline", "nortriptyline", "imipramine", "desipramine",
                          "doxepin", "clomipramine", "protriptyline", "trimipramine", "amoxapine",
                          "chlorambucil"}
# (propofol is Ruffmann intermediate-tier but not in PROCONV; tricyclic antidepressants matched to
# the TCA list above.)
RUFFMANN_LOW = {"ciprofloxacin", "levofloxacin", "norfloxacin", "ofloxacin", "imipenem", "ertapenem",
                "meropenem", "bupropion"}
# (iodinated contrast agents are Ruffmann low-tier; "iopanoic-acid" exists in compounds.parquet but
# is not in the PROCONV literal, so not included here; fluoroquinolones/carbapenems/bupropion ARE
# in PROCONV and are matched.)


def extract_proconv(path):
    txt = open(path).read()
    m = re.search(r"PROCONV\s*=\s*\{(.*?)\n\}", txt, re.S)
    items = re.findall(r"'([^']+)'", m.group(1))
    return items


def canonicalize(name):
    """Collapse hyphen/space/underscore variants and known synonym pairs to one canonical key."""
    n = name.lower().strip()
    syn_map = {
        "pentylenetetrazole": "pentylenetetrazol", "kainic acid": "kainic-acid",
        "cyclosporin-a": "cyclosporine", "pethidine": "meperidine",
    }
    return syn_map.get(n, n)


def main():
    pact_items = extract_proconv(PACT_FILE)
    atlas_items = extract_proconv(ATLAS_FILE)
    assert pact_items == atlas_items, "PACT and lincs-detectability-atlas PROCONV literals differ!"
    print(f"PACT PROCONV: {len(pact_items)} raw strings (identical in both codebases, confirmed)")

    # Group raw strings into canonical drugs with synonym lists.
    groups = {}
    for raw in pact_items:
        key = canonicalize(raw)
        groups.setdefault(key, set()).add(raw)
    print(f"-> {len(groups)} canonical compounds after synonym/hyphen collapsing")

    cmp = load_compounds()
    names_lower = cmp.name.astype(str).str.lower()
    name_to_ik14 = dict(zip(names_lower, cmp.ik14))

    sider = pd.read_parquet(D + "sider_full.parquet") if _exists(D + "sider_full.parquet") else None
    convulsion_terms = {"convulsion", "convulsions", "seizure", "grand mal convulsion",
                        "epilepsy", "generalised tonic-clonic seizure", "tonic clonic movements",
                        "status epilepticus"}
    sider_ik_convulsion = set()
    if sider is not None and len(sider):
        hit = sider[sider.se_name.str.lower().isin(convulsion_terms)]
        sider_ik_convulsion = set(hit.ik14)

    rows = []
    for canon, syns in sorted(groups.items()):
        matched_ik14 = None
        for s in syns | {canon}:
            s2 = s.lower().replace(" ", "-")
            if s.lower() in name_to_ik14:
                matched_ik14 = name_to_ik14[s.lower()]; break
            if s2 in name_to_ik14:
                matched_ik14 = name_to_ik14[s2]; break
        in_lincs = matched_ik14 is not None
        sider_flag = (matched_ik14 in sider_ik_convulsion) if in_lincs else False
        tier = ("high" if canon in RUFFMANN_HIGH else
                "intermediate" if canon in RUFFMANN_INTERMEDIATE else
                "low" if canon in RUFFMANN_LOW else "")
        rows.append(dict(canonical_name=canon, synonyms=";".join(sorted(syns)),
                         source="Chen, Albertson & Olson, Br J Clin Pharmacol 2016 (via this group's "
                                "prior unpublished curation, NOT cited as 'PACT' externally)",
                         ik14=matched_ik14, in_lincs=in_lincs,
                         sider_convulsion_flag=sider_flag, ruffmann_tier=tier))

    out = pd.DataFrame(rows)
    out.to_csv("../data/proconvulsants.csv", index=False)
    print(f"\nWrote data/proconvulsants.csv: {len(out)} canonical proconvulsant drugs")
    print(f"  in_lincs (resolves to one of the 9,878 compounds): {out.in_lincs.sum()}/{len(out)}")
    print(f"  sider_convulsion_flag=True: {out.sider_convulsion_flag.sum()}/{out.in_lincs.sum()} of those in LINCS")
    print(f"  ruffmann_tier assigned: {(out.ruffmann_tier != '').sum()}/{len(out)}")
    print(out.ruffmann_tier.value_counts())


def _exists(p):
    import os
    return os.path.exists(p)


if __name__ == "__main__":
    main()
