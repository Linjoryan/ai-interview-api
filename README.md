# Student Performance Prediction API

A REST API built with FastAPI for predicting student performance using a logistic regression model.

## Features

- ✅ Real-time student performance predictions
- ✅ Confidence scores and probability distributions
- ✅ Input validation and error handling
- ✅ Interactive API documentation
- ✅ Health check endpoints

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Your trained logistic regression model saved as `model.pkl`

## Installation

### 1. Clone or Download the Project

```bash
cd ~/Desktop/ai\ interview
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn pydantic numpy scikit-learn
```

Or create a `requirements.txt` file with:

```
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
numpy==1.24.3
scikit-learn==1.3.2
```

Then install:

```bash
pip install -r requirements.txt
```

### 4. Place Your Model File

Make sure your trained model file `model.pkl` is in the same directory as `main.py`:

```
ai interview/
├── main.py
├── model.pkl
└── requirements.txt
```

## Running the API

### Start the Server

```bash
uvicorn main:app --reload
```

You should see output like:

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using WatchFiles
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Root Endpoint
**GET** `/`

Returns API information and available endpoints.

```bash
curl http://localhost:8000/
```

### 2. Health Check
**GET** `/health`

Check if the API and model are loaded correctly.

```bash
curl http://localhost:8000/health
```

### 3. Predict Student Performance
**POST** `/predict`

Make predictions based on student features.

**Request Body:**

```json
{
  "sex": 0,
  "age": 17,
  "Medu": 3,
  "Fedu": 3,
  "famrel": 4,
  "freetime": 3,
  "goout": 2,
  "Dalc": 1,
  "Walc": 2,
  "health": 4,
  "absences": 4
}
```

**Feature Descriptions:**

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| sex | int | 0-1 | Student's sex (0 or 1) |
| age | int | 15-22 | Student's age |
| Medu | int | 0-4 | Mother's education level |
| Fedu | int | 0-4 | Father's education level |
| famrel | int | 1-5 | Quality of family relationships |
| freetime | int | 1-5 | Free time after school |
| goout | int | 1-5 | Going out with friends |
| Dalc | int | 1-5 | Workday alcohol consumption |
| Walc | int | 1-5 | Weekend alcohol consumption |
| health | int | 1-5 | Current health status |
| absences | int | 0+ | Number of school absences |

**Example using cURL:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sex": 0,
    "age": 17,
    "Medu": 3,
    "Fedu": 3,
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 2,
    "health": 4,
    "absences": 4
  }'
```

**Response:**

```json
{
  "prediction": "Pass",
  "prediction_label": 1,
  "confidence_score": 0.8523,
  "probability_pass": 0.8523,
  "probability_fail": 0.1477
}
```

## Testing with Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "sex": 0,
    "age": 17,
    "Medu": 3,
    "Fedu": 3,
    "famrel": 4,
    "freetime": 3,
    "goout": 2,
    "Dalc": 1,
    "Walc": 2,
    "health": 4,
    "absences": 4
}

response = requests.post(url, json=data)
print(response.json())
```

## Error Handling

The API includes comprehensive error handling:

- **422 Unprocessable Entity**: Invalid input data (out of range, wrong type)
- **500 Internal Server Error**: Model prediction errors
- **503 Service Unavailable**: Model not loaded

Example error response:

```json
{
  "detail": "Invalid input data: Value cannot be None"
}
```

## Configuration

### Change the Port

```bash
uvicorn main:app --reload --port 8080
```

### Run in Production

For production deployment, remove `--reload` and consider using workers:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Change Model Path

Edit the `MODEL_PATH` variable in `main.py`:

```python
MODEL_PATH = "path/to/your/model.pkl"
```

## Troubleshooting

### Model Not Loading

**Error**: `Model file not found at model.pkl`

**Solution**: Ensure `model.pkl` is in the same directory as `main.py`, or update `MODEL_PATH`

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**: Install dependencies:
```bash
pip install -r requirements.txt
```

### Port Already in Use

**Error**: `Address already in use`

**Solution**: Use a different port:
```bash
uvicorn main:app --reload --port 8080
```

Or kill the process using port 8000:
```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

## Development

### View Logs

Logs are printed to the console when running with `--reload`. Check for:
- Model loading status
- Prediction results
- Error messages

### Interactive Documentation

Access Swagger UI at http://localhost:8000/docs to:
- Test endpoints directly in the browser
- View request/response schemas
- Try different input values

## License

This project is for educational purposes.

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify your model file is compatible
3. Ensure all input values are within valid ranges
4. Visit the interactive docs at `/docs` for testing
