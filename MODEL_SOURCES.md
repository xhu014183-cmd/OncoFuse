# Optional image-representation providers

The core demo does not require a learned image model. NIfTI/mask measurement,
AFP/DCP parsing, transparent fusion, and LLM rendering remain usable without
PyTorch or model weights.

## Provider registry

`src/hcc_multimodal/volume_encoders.py` exposes two providers:

| Provider | Output | Default installation | Intended role |
| --- | --- | --- | --- |
| `statistical-v1` | 25 deterministic volume and mask summary features | Included | Exercise and test the provider/artifact contract |
| `m3d-clip` | 768-dimensional normalized CLS embedding | Optional | Research spike for a learned 3D image representation |

The statistical provider is not a learned model and must not be presented as a
clinical image representation. It records windowed ROI quantiles, masked
quantiles, mask fraction, target spacing, and cropped shape so the complete
physical preprocessing and artifact path can be tested offline.

## M3D-CLIP provenance

- GitHub project: <https://github.com/BAAI-DCAI/M3D>
- Hugging Face model: `GoodBaiBai88/M3D-CLIP`
- Pinned revision: `ae091d89a0ef38b533ecc4ed21426f7658853963`
- GitHub code license: MIT
- Hugging Face model-card license: Apache-2.0
- Expected input: `[batch, 1, 32, 256, 256]`
- Selected output: normalized CLS token from `encode_image`, dimension 768
- Approximate parameter count: 198 million FP32 parameters

The adapter never calls the model's training `forward` method. It invokes
`encode_image(...)` and selects token zero. The full embedding is stored in a
`.npy` artifact. Only its manifest and SHA-256 digest are written to JSON.

## Security and download policy

The model uses Hugging Face custom code. The demo therefore:

1. Pins the exact model revision.
2. Defaults to `local_files_only=True`.
3. Refuses to execute custom model code without `--allow-m3d-remote-code`.
4. Refuses to download weights without `--allow-model-download`.
5. Does not import Torch, Transformers, or MONAI during normal core use.

Install the optional runtime:

```powershell
.\.venv\Scripts\python -m pip install -e ".[m3d]"
```

Run an explicitly authorized research encoding:

```powershell
.\.venv\Scripts\hcc-demo encode-volume `
  --image case_ct.nii.gz `
  --mask case_mass_mask.nii.gz `
  --patient-id RESEARCH_CASE_001 `
  --study-date 2026-07-15 `
  --image-encoder m3d-clip `
  --allow-m3d-remote-code `
  --allow-model-download `
  --output embedding-output
```

The upstream custom model constructor also initializes `bert-base-uncased`.
An actually offline first run therefore requires both M3D-CLIP and that BERT
model to be present in the Hugging Face cache.

## NIfTI preprocessing

The adapter records the following research preprocessing in every manifest:

```text
NIfTI -> closest canonical orientation
      -> finite-value replacement -> fixed CT window [-200,300] HU
      -> physical resampling to 1.5x1.5x2.5 mm
      -> lesion-mask ROI plus 16 mm margin, image/mask kept synchronized
      -> X,Y,Z to D,H,W -> trilinear image and nearest-mask resize
      -> 1x32x256x256 model input
```

All transformation parameters, phase, mask voxel counts, finite-value checks,
and embedding L2 norm are written to the manifest. M3D does not publish a
clinically validated mapping from raw CT Hounsfield units to the rendered
training images. This remains an engineering integration experiment; resizing
an ROI to 32 slices can remove small-lesion detail. The expert-mask geometry
branch remains the auditable evidence source.

## Rights boundary

The M3D README states that its Radiopaedia-derived data support non-commercial
machine-learning use. Code and model-card licenses do not by themselves settle
rights in training data or derived weights. Before any Roche, United Imaging,
or other commercial use, perform a separate legal review of the data lineage,
model terms, and intended deployment.
