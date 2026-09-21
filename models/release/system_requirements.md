# ViralLens AI — System Requirements

## 1. Ground Truth Runtime Specifications

| Component | Minimum Tested Environment | Recommended Production Environment |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11 64-bit | Windows 10/11 64-bit or Ubuntu 22.04 LTS |
| **Python Runtime** | Python 3.10.x | Python 3.12.x (Tested: 3.12.4) |
| **Processor (CPU)** | 4-Core x86_64 processor | 8-Core Intel Core i7/i9 or AMD Ryzen 7/9 |
| **System Memory (RAM)** | 4 GB (Observed pipeline peak: ~1.4 GB) | 16 GB DDR4/DDR5 |
| **Disk Storage** | 2 GB free space (SSD recommended) | 10 GB free space for video/competitor caching |
| **GPU Acceleration** | Not required (CPU inference supported) | NVIDIA GPU with CUDA 12+ (faster saliency) |
| **FFmpeg Binary** | FFmpeg 6.x or 7.x on PATH | FFmpeg 7.x Static Build |
| **Tesseract OCR** | Tesseract 5.x (Optional for on-screen text) | Tesseract 5.3+ installed |
| **Network** | Outbound HTTPS (Ports 443) | Stable broadband (min 15 Mbps download) |

## 2. API Credentials & Authentication
1. **Twelve Labs API Key**: Required for target Reel multimodal video analysis (`TWELVELABS_API_KEY`).
2. **Apify API Token**: Required for production Instagram competitor discovery (`APIFY_API_TOKEN`).
3. **Agent-Reach / OpenCLI (Optional)**: Experimental; requires local CLI executable and active session.

## 3. Storage & Cache Allocation
- **Media Cache**: `media/` (persists uploaded target videos and downloaded competitor MP4s).
- **Competitor Artifact Cache**: `cache/competitor_artifacts/` (deterministic JSON analysis cache).
- **Discovery Cache**: `cache/apify_discovery/` (isolated per-query candidate cache).
- **Reports**: `reports/` (JSON & Markdown audit files).
