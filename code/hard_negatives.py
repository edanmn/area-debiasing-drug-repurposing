"""Per-indication SIDER 'opposite/characteristic adverse effect' hard-negative term map, applying
DATA_PLAN.md's stated rule (convulsion/seizure for epilepsy, hypotension for antihypertensives)
to every primary evaluation indication. Documented explicitly (not silently assumed) because this
determines both the NRBdMF-style baseline's +1 labels and the step-5(c) hard-negative ranking check.

epilepsy and the betablocker/raas_antihypertensive pair follow the literal DATA_PLAN.md examples.
depression and antipsychotic extend the SAME stated rule ("SIDER lists the opposite/characteristic
adverse effect") to their own indication: a drug whose SIDER-listed side effect is itself the
disease being treated (SIDER PT 'depression' for a depression candidate; 'psychotic disorder' /
'hallucination' for an antipsychotic candidate) is a direct analogue of 'convulsion for epilepsy'.
Other indications are left WITHOUT an hard-negative term map where no defensible "opposite effect"
exists in SIDER's PT vocabulary; this is reported honestly in RESULTS.md rather than invented.
"""
HARD_NEG_TERMS = {
    "epilepsy": {"convulsion", "convulsions", "grand mal convulsion", "status epilepticus",
                 "generalised tonic-clonic seizure"},
    "depression": {"depression", "suicidal ideation", "suicide attempt", "major depression"},
    "antipsychotic": {"psychotic disorder", "hallucination", "schizophrenia", "hallucination, visual"},
    "betablocker": {"hypotension"},
    "raas_antihypertensive": {"hypotension"},
    "antidiabetic": {"hyperglycaemia"},
    "nsaid": {"gastrointestinal haemorrhage", "peptic ulcer"},
}
