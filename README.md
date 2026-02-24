# 🩺 Continuum — Clinical AI Platform

> End-to-end clinical AI pipeline: patient intake → interview → AI triage → CXR diagnosis → AI-powered prescription safety → FHIR record export.

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| 📂 **Project Data Drive** | _[https://drive.google.com/file/d/1-lhgEEo1QOlXSdrkcU-E4PPM8QLTVOsT/view?usp=sharing]_ |
| ☁️ **Google Colab (run in browser)** | _[https://colab.research.google.com/drive/1TNyDWxWkE3MsK71ka1q_4DuZGprpXsV1?usp=drive_link]_ |
| 🤗 **Model — MedGemma 4b** | [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it) |

---

## Pipeline Flow

```
① Registration  →  ② Clinical Interview  →  ③ AI Triage
        ↓
④ CXR & Lab Diagnosis  →  ⑤ Prescription Safety  →  ⑥ Complete Records
```

### Flow Design Diagram

![Continuum Flow Design](Smart%20Front%20Door%20Patient-2026-02-24-071947.png)

---

## Notebook Files

| File | Purpose |
|------|---------|
| `Continuum.ipynb` | Primary production notebook — run this |

---

## Setup

### 1. Clone & install dependencies

```bash
git clone <repo-url>
```

### 2. Set HuggingFace token (for model download)

```bash
export HF_TOKEN=hf_...
```

Or place it in a `.env` file:
```
HF_TOKEN=hf_...
```

### 3. Run

Open `Continuum.ipynb` in Jupyter / VS Code and **Run All Cells**, then:

```python
launch_gradio_ui()   # Opens at http://localhost:7860
```

---

## Data Structure

```
Continuum/
├── DAta/master/
│   ├── global_patient_registry.csv    # Patient demographics & gene profiles
│   └── global_provider_registry.csv   # Doctors, departments, available slots
├── Mod1/                              # Intake / conversation storage
│   └── data/conversations/            # Per-visit JSON, FHIR bundles, lab JSON
├── Mod4/
│   └── Prescription_KG_Data/
│       └── drug_knowledge.json        # DDI × 37, Allergy × 10, Contrain × 18, PGx × 6
└── Mod3/
    └── .lancedb/                      # CXR vector embeddings (auto-created on first run)
```

---

## Key Features

- **MedGemma 4b** — multimodal clinical LLM (text + CXR image)
- **AI Triage** — department routing with provisional clinic assignment card
- **Longitudinal CXR** — dual-image comparative analysis for returning patients
- **Two-layer drug safety** — KB rule engine (37 DDI, PGx, allergy) + AI dosage / indication review
- **FHIR R4** — Patient / Condition / Observation / MedicationRequest bundle per visit
- **Offline** — no external APIs beyond HuggingFace model download

---

## Requirements

See [requirements.txt](requirements.txt) — key dependencies:

```
transformers>=4.40
torch>=2.1
gradio>=4.0
lancedb
sentence-transformers
pandas
Pillow
python-dotenv
```

---

## License

See [LICENSE](LICENSE).
