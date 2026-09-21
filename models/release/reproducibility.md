# ViralLens AI — Reproducibility & Audit Protocol

## 1. Execution Environment Reproducibility
To reproduce the engineering baseline of ViralLens AI Release Candidate `0.18.0-rc1`, follow this exact sequence:

### Environment Setup
```powershell
Set-Location "E:\new viral"
py -3.12 -m venv .venv
& ".venv\Scripts\Activate.ps1"
pip install -r requirements.txt
```

### Deterministic Random Seeds
Where stochastic operations or train/test splits are evaluated, the fixed random seed `42` is employed throughout the codebase (`random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`).

## 2. Frozen Model Hashes & Paths
All machine learning artifacts are frozen in read-only states:
- **NEMAR Model**: `models/human_attention/model.pkl` (SHA-256 prefix: `38b8c952d0806a2f`)
- **NEMAR Scaler**: `models/human_attention/scaler.pkl` (SHA-256 prefix: `438102936820e4f1`)
- **NEMAR Feature Schema**: `models/human_attention/feature_schema.json` (SHA-256 prefix: `bde5b68e4256d763`)
- **TinySalNet Spatial Weights**: `models/human_attention_experiments/dhf1k_spatial/spatial_model.pt`
  - Exact SHA-256: `fd6f27fd5ef6334c597ee3981086fc70f639cd59c7d57399fa72253d883b95c9`

## 3. Dataset Provenance & Immutability
- **NEMAR Raw Dataset**: `E:\v1.0.0` (READ-ONLY, untouched). DOI: `10.82901/nemar.nm000150`.
- **DHF1K Dataset**: Dynamic Human Fixations (READ-ONLY external benchmark).

## 4. Benchmark & External Discovery Reproducibility Boundary
> [!IMPORTANT]
> While all internal model predictions (NEMAR attention curves, OCR recognition, pattern aggregation, recommendation graph) are 100% deterministic and reproducible given identical inputs, **external Instagram discovery results inherently fluctuate over time** due to live Instagram platform activity, creator uploads, changing view counts, and platform rate limits. Cached discovery fixtures in `cache/apify_discovery/` allow deterministic offline regression testing.
