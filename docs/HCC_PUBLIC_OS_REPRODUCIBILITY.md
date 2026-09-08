# HCC Public OS Reproducibility Commands

The following PowerShell sequence starts from an empty data/output directory. Run it
from a clean checkout after reviewing both dataset licenses. Raw data and generated
research outputs are Git-ignored.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,public-data,research]"

$researchDataRoot = "E:\hcc-public-data"
$researchOutputRoot = "research-output"

hcc-demo sync-prognosis-data --dataset waw-tace `
  --tier metadata --data-root $researchDataRoot --accept-license
hcc-demo sync-prognosis-data --dataset hcc-tace-seg `
  --tier pilot --data-root $researchDataRoot --accept-license

hcc-demo build-prognosis-cohort `
  --data-root $researchDataRoot `
  --output "$researchOutputRoot\cohort"

hcc-demo train-prognosis-model `
  --cohort "$researchOutputRoot\cohort\development-cohort.json" `
  --output "$researchOutputRoot\models" `
  --seed 1729 --bootstrap-iterations 1000

# Inspect protocol, QC, exclusions, folds, model hashes and internal results here.
# Only after the model is frozen, prepare the full external image/SEG cohort:
hcc-demo sync-prognosis-data --dataset hcc-tace-seg `
  --tier full --data-root $researchDataRoot --accept-license
hcc-demo build-prognosis-cohort `
  --data-root $researchDataRoot `
  --output "$researchOutputRoot\cohort-frozen"

hcc-demo evaluate-prognosis-model `
  --cohort "$researchOutputRoot\cohort-frozen\external-test-cohort.json" `
  --models "$researchOutputRoot\models" `
  --output "$researchOutputRoot\external" `
  --unlock-external --seed 1729 --bootstrap-iterations 1000

python -m ruff check .
python -m mypy src
python -m pytest -q --cov=hcc_multimodal --cov-branch
```

WAW portal-venous CT is not needed for the primary model. Download WAW `full` only
when running the prespecified portal-HU feasibility analysis or GLM image examples.
Never send outcome files, AFP, model risk, or patient identifiers to GLM.
