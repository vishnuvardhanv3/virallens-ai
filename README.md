<p align="center">
  <img src="docs/images/virallens_logo.png" alt="ViralLens AI Logo" width="180" />
</p>

# ViralLens AI

A multimodal Instagram Reel intelligence system.

ViralLens AI combines state-of-the-art multimodal video understanding (Twelve Labs Pegasus 1.5), laboratory-derived human visual attention modeling, multi-agent qualitative critique, automated competitor discovery via Apify, semantic relevance filtering, and unsupervised machine learning to perform rigorous comparative audits of short-form video content.

---

## 1. System Overview & Architecture

ViralLens AI is engineered to evaluate Instagram Reels against genuine, top-performing competitors in the same content niche. It extracts high-density signals from video frames, audio speech, and on-screen text, predicts laboratory visual-attention potential across time, discovers active competitor Reels, and clusters empirical patterns.

```
                  ┌──────────────────────────────────────────────┐
                  │            Target Reel Upload (MP4)          │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
  ┌─────────────────────────────┐                 ┌─────────────────────────────┐
  │   Twelve Labs Multimodal    │                 │   Laboratory Attention &    │
  │      Understanding AI       │                 │     Multimodal Signals      │
  │  (Topics, Niche, Hooks,     │                 │  • NEMAR Visual Attention   │
  │   Summary, Transcript)      │                 │  • Shadow Spatial Saliency  │
  └──────────────┬──────────────┘                 │  • Librosa Audio & Whisper  │
                 │                                │  • Tesseract On-Screen OCR  │
                 │                                └──────────────┬──────────────┘
                 ▼                                               │
  ┌─────────────────────────────┐                                │
  │    Multi-Agent Framework    │◄───────────────────────────────┘
  │ (Attention, Hook, Behavior, │
  │  Emotion, Editing, Script)  │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │   Competitor Discovery &    │
  │     Relevance Auditing      │
  │  • Niche-Anchored Queries   │
  │  • Apify Instagram Search   │
  │  • OpenCV Video Download    │
  │  • Semantic Verification    │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │   ML Pattern Discovery      │
  │  (K-Means & Feature Matrix) │
  └──────────────┬──────────────┘
                 │
                 ▼
  ┌─────────────────────────────┐
  │     Streamlit Glassmorphic  │
  │           Dashboard         │
  └─────────────────────────────┘
```

---

## 2. Major Capabilities

### 2.1 Multimodal Video Understanding
- **Twelve Labs Pegasus 1.5 Integration**: Generates structured deep video summaries, entity identification, niche classification, semantic hook evaluations, emotional tone categorization, and pacing audits.
- **Multimodal Video Preprocessing**: Automatically downsamples high-resolution 4K/60fps video to optimal 720p/10fps budgets for fast API transmission and OpenCV frame-level feature extraction.

### 2.2 Laboratory-Derived Visual Attention & Spatial Saliency
> **Scientific Disclosure**: The attention models implemented in ViralLens AI provide a measure of **visual-attention potential** and **visual saliency** based on laboratory eye-tracking protocols (trained on the NEMAR laboratory gaze dataset and validated against DHF1K benchmark saliency datasets). These models do **not** claim to predict Instagram viewer retention or platform watch-time metrics.
- **Temporal Visual-Attention Potential**: Predicts human gaze concentration and visual engagement potential over 0.5-second windows across 14 extracted visual and acoustic features (motion magnitude, scene cuts, brightness, contrast, visual complexity, face/text presence, audio RMS, silence ratio).
- **Shadow Spatial Saliency Service**: Runs an independent, CPU-optimized TinySalNet spatial model in shadow mode to analyze spatial dispersion, Shannon entropy, and inter-frame gaze shifts without altering primary production predictions.

### 2.3 Audio, Speech & On-Screen OCR
- **Acoustic Profiling (Librosa)**: Measures RMS energy, silence distribution, and spectral centroid dynamics across the audio track.
- **Speech Transcription (OpenAI Whisper)**: Extracts speech timing and transcript dynamics.
- **Optical Character Recognition (pytesseract)**: Identifies on-screen text overlays, title cards, and subtitle timing.

### 2.4 Multi-Agent Analytical Architecture
ViralLens AI delegates qualitative analysis to dedicated specialist agents coordinated by a Master Orchestrator:
- **Attention Agent**: Evaluates visual saliency curves and peak attention zones.
- **Hook Agent**: Inspects the opening 3-second frame dynamics and semantic hook strength.
- **Behavior & Emotion Agent**: Assesses host delivery, expressions, and emotional tonality.
- **Editing & Pacing Agent**: Analyzes cut density, visual complexity, and audio-video sync.
- **Script & Messaging Agent**: Reviews narrative progression and clarity.
- **Strategy & Benchmark Agent**: Synthesizes empirical gaps against verified competitors.

### 2.5 Instagram Competitor Discovery & Semantic Verification
- **Niche-Anchored Multi-Word Queries**: Constructs tailored search expressions combining entity, niche, and format cues.
- **Apify Instagram Discovery**: Scrapes genuine Instagram Reels matching target criteria using the official Apify client.
- **Automated Validation & Download**: Validates container headers and frame decodability via OpenCV.
- **Semantic Relevance Verification**: Each downloaded competitor is analyzed by Twelve Labs to confirm true topical and structural alignment; off-topic candidates are flagged as `analyzed_not_relevant` and excluded from benchmarking.

### 2.6 Unsupervised Machine Learning Pattern Discovery
- **Feature Standardization & Clustering**: Applies `StandardScaler` and `KMeans` across verified competitors when a statistically sound sample (≥ 4 verified competitors) is present.
- **Empirical Gap Identification**: Pinpoints specific structural, pacing, and visual characteristics where the target Reel deviates from top-performing competitor cohorts.

### 2.7 Interactive Streamlit Dashboard
- **Glassmorphic UI**: High-contrast, dark-mode dashboard with interactive metric dials, attention timeline visualizations, multi-agent collapsible panels, competitor cards, and downloadable JSON/Markdown audit reports.

---

## 3. Technology Stack

- **Frontend & App Server**: Streamlit (Python)
- **Computer Vision & Video**: OpenCV (`opencv-python`), FFmpeg
- **Audio Processing**: Librosa, SoundFile, OpenAI Whisper
- **Text & OCR**: PyTesseract, Tesseract OCR Engine
- **Multimodal AI**: Twelve Labs API (Pegasus 1.5 engine)
- **Web Scraping & Social Discovery**: Apify Client (`apify-client`)
- **Machine Learning**: Scikit-Learn, NumPy, Pandas, Joblib
- **Testing**: Pytest

---

## 4. External Services & Credentials

> **Important**: External services require active third-party accounts and API credentials:
> 1. **Twelve Labs**: Requires an active API key with index access for the `pegasus1.5` engine.
> 2. **Apify**: Requires an active API token with access to the `apify/instagram-search-scraper` actor.
> 
> If external API tokens are omitted or depleted, the application degrades gracefully: target video feature extraction and laboratory attention modeling execute locally, while external discovery steps report their exact status transparently under the Zero Fabrication Policy.

---

## 5. Local Setup & Installation

### Prerequisites
- Python 3.12+
- FFmpeg installed and available on system `PATH`
- Tesseract OCR engine installed (optional, for on-screen text extraction)

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/virallens-ai.git
   cd virallens-ai
   ```

2. **Create and Activate Virtual Environment**:
   - **Linux / macOS**:
     ```bash
     python3.12 -m venv .venv
     source .venv/bin/activate
     ```
   - **Windows (PowerShell)**:
     ```powershell
     py -3.12 -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

3. **Install Dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your API credentials:
   ```env
   TWELVELABS_API_KEY=your_twelvelabs_api_key
   TWELVELABS_INDEX_ID=your_index_id
   APIFY_API_TOKEN=your_apify_api_token
   ```

5. **Run the Test Suite**:
   ```bash
   pytest -v
   ```

6. **Start the Streamlit Application**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## 6. Streamlit Community Cloud Deployment

ViralLens AI is configured for deployment on Streamlit Community Cloud:

1. **Repository Settings**:
   - Main file path: `app/streamlit_app.py`
   - Python version: `3.12`
2. **System Dependencies**:
   - System packages are declared in [packages.txt](packages.txt) (`ffmpeg`, `tesseract-ocr`, `libgl1`, `libglib2.0-0`).
3. **Secrets Management**:
   - In Streamlit Cloud Project Settings -> **Secrets**, paste:
     ```toml
     TWELVELABS_API_KEY = "your_twelvelabs_api_key"
     TWELVELABS_INDEX_ID = "your_twelvelabs_index_id"
     APIFY_API_TOKEN = "your_apify_api_token"
     ```

---

## 7. Zero Fabrication Policy

ViralLens AI strictly enforces a Zero Fabrication Policy:
- Metrics, view counts, engagement figures, and competitor data are never generated or simulated.
- When an Instagram scraping query is restricted, blocked, or yields zero results, the system reports the real status without fallback spoofing.
- Attention predictions reflect frozen laboratory regression models, not arbitrary synthetic scores.

---

## 8. License

This project is released under the MIT License.
