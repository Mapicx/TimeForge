# TimeForge: GNSS Clock Bias Prediction & Analysis System

**TimeForge** is a full-stack application designed to monitor, analyze, and predict GNSS (Global Navigation Satellite System) clock biases and orbit errors. It utilizes a Transformer-based Deep Learning model for high-precision time series forecasting and features an interactive 3D frontend with an integrated AI agent for natural language data analysis.

---

## 🚀 Key Features

* **Deep Learning Engine**: Custom Transformer Encoder model trained on satellite clock data to predict future clock bias corrections.
* **Interactive 3D Visualization**: Real-time rendering of satellite constellations (GPS, GLONASS, Galileo, BeiDou) using Three.js and React.
* **Data Analysis Dashboard**: Visual charts for clock error and orbit error predictions using Recharts/Plotly.
* **AI Agent Assistant**: Built-in AI chat interface (powered by LangChain & Google GenAI) to query satellite performance, compare errors, and generate summaries.
* **Multi-Constellation Support**: Filters and visualization for specific satellite groups.

---

## 🛠 Tech Stack

### **Frontend**
* **Framework**: React (Vite)
* **Styling**: Tailwind CSS, PostCSS
* **Visualization**: Three.js (`three-globe`), Lucide React Icons
* **HTTP Client**: Axios

### **Backend**
* **Server**: FastAPI (Python)
* **Data Processing**: Pandas, NumPy
* **AI/LLM Integration**: LangChain, Google Gemini API (GenerativeLanguage)

### **Deep Learning (DL)**
* **Framework**: PyTorch
* **Model**: Transformer Encoder with learned Satellite Embeddings
* **Utilities**: Scikit-Learn (metrics), Astropy (time conversion)

---

## 📂 Project Structure

```text
TimeForge/
├── backend/                # FastAPI Server and AI Logic
│   ├── agent_prototype/    # Advanced AI Agent modules (LangChain)
│   ├── data/               # CSV Data storage (e.g., suyash_gandu.csv)
│   ├── server.py           # Main entry point for the API
│   └── requirements.txt    # Backend dependencies
├── frontend/               # React + Vite User Interface
│   ├── src/
│   │   ├── components/     # UI Components (Galaxy, Globe, Charts)
│   │   ├── App.jsx         # Main Application Logic
│   │   └── index.css       # Global Styles
│   └── package.json        # Frontend dependencies
└── DL/                     # Deep Learning Model Training
    ├── model/              # Transformer model architecture
    ├── utils/              # Data loaders and metrics
    ├── train.py            # Training script
    └── requirements.txt    # ML dependencies
```

---

## ⚙️ Installation & Setup

### 1. Deep Learning Module (Optional - for training)
If you wish to retrain the model or generate new `.pth` weights:

```bash
cd DL
# Install scientific computing dependencies
pip install -r requirements.txt
# Run the training script
python train.py --epochs 10 --batch_size 32
```

**Note:** The script saves the model as `gnss_transformer_model_with_embeddings.pth`.

---

### 2. Backend Setup
The backend serves the API and the AI agent.

```bash
cd backend
# Install backend dependencies
pip install -r requirements.txt

# Create a .env file
touch .env
```

**Configuration:** Ensure you place your data file (e.g., `suyash_gandu.csv`) inside `backend/data/` as referenced in `server.py`.

**Run the Server:**
```bash
# Starts the FastAPI server
python server.py
# OR using uvicorn directly
uvicorn server:app --reload --port 8000
```

---

### 3. Frontend Setup
The user interface is built with Vite.

```bash
cd frontend
# Install Node dependencies
npm install  # or yarn install

# Start the development server
npm start    # Runs 'craco start' or 'vite' depending on config
```

The frontend will launch at `http://localhost:3000` by default.

---

## 🖥 Usage Guide

**Dashboard:** Upon loading, you will see a 3D globe. Click on any satellite (dots on the globe or list on the left) to view its specific details.

**Analysis Panel:** Clicking a satellite opens the right panel showing:
- **Summary:** Average clock error and peak orbit error.
- **Charts:** Click on "Clock Error Prediction" or "Orbit Error Prediction" to expand the graphs.

**AI Agent:**
- Click the "AI Agent" button in the header or the "Ask AI Agent" button in the satellite panel.
- Type queries like:
  - `"Compare G01 vs G12 orbit error"`
  - `"Find satellites with peak clock error > 2m"`

**Filtering:** Use the dropdown in the header to filter satellites by constellation (GPS, GLONASS, Galileo, BeiDou).

---

## 🤖 AI Agent Configuration

The system uses LangChain and Google Generative AI. To make the AI features work fully in the backend:

1. Obtain a Google API Key.
2. Add it to your `backend/.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## 📜 License
MIT
