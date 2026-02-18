# Safety Engine

A medication safety checking engine that validates drug-drug interactions (DDI), allergies, and contraindications.

## Project Structure

```
safety-engine/
├── api/
│   └── main.py              # FastAPI application — all routes
├── engine/
│   ├── safety_engine.py     # Core check logic (DDI, allergy, contraindication)
│   └── patient_store.py     # Patient context retrieval
├── data/
│   ├── patients.json        # Mock patient DB (5 patients)
│   └── drug_knowledge.json  # Interaction rules (DrugBank-modeled)
├── tests/
│   └── test_safety_engine.py
├── requirements.txt
└── README.md
```

## Data Gdrive Link

[KG / Patient records](https://drive.google.com/drive/folders/1vNK9AA94jyQ39FpAX46XWcBJI3-s_R1Q?usp=drive_link)

## Setup

```bash
cd safety-engine
pip install -r requirements.txt
```

## Running

```bash
uvicorn api.main:app --reload
```

## API Endpoints

- `POST /check` - Check medication safety for a patient
- `GET /patients` - List all patients
- `GET /patients/{patient_id}` - Get patient details

## Testing

```bash
pytest tests/
```

