# module4# BOAMP Data Extractor

This is a FastAPI application that extracts and analyzes public procurement data from BOAMP API.

## Deployment on Render

The app is deployed on Render and can be accessed at [https://boamp-fastapi.onrender.com](https://boamp-fastapi.onrender.com).

## Local Development

1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (on Windows: `venv\Scripts\activate`)
4. Install dependencies: `pip install -r requirements.txt`
5. Run the app: `uvicorn main:app --reload`

Then open http://localhost:8000 in your browser.