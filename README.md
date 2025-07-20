Jingly AI
Basic Details
Team Name: Mavericks

Team Members:  

Abhijith N P 
Akshay Sasi 
Anjitha S 
Arsha Venkitakrishnan

Track: AI-Powered Audio SolutionsProblem Statement: Creating engaging, brand-specific jingles is time-consuming and requires musical expertise, making it challenging for businesses to produce high-quality audio advertisements quickly and cost-effectively.Solution: Jingly AI automates the creation of custom jingles using AI-driven music and vocal generation, allowing users to input brand details and receive a tailored 30-second jingle in minutes.Project Description: Jingly AI is a web-based application that generates professional jingles for brands. Users input details like brand name, motto, product brief, and music genre through a React-based frontend. The backend, powered by Python and AI models (MusicGen and Bark), generates instrumental music and vocals, combining them into a downloadable WAV file. The system is designed for ease of use, enabling businesses to create marketing jingles without musical expertise.
Technical Details
Tech Stack and Libraries Used

Frontend:
React 18.2.0: JavaScript library for building the user interface.
Vite 5.2.0: Fast build tool and development server.
Axios 1.6.8: For making HTTP requests to the backend API.
Tailwind CSS: For styling the UI (assumed based on class names like bg-indigo-600).


Backend:
Python 3.10: Core programming language.
FastAPI: Web framework for building the API.
Uvicorn: ASGI server for running FastAPI.
MusicGen (facebook/musicgen-small): AI model for generating instrumental music.
Bark (suno/bark-small): AI model for generating vocals.
Google Generative AI (gemini-1.5-flash): For generating music prompts and lyrics.
PyDub: For audio processing and combining music/vocals.
PyTorch: For running AI models (CPU-only in current setup).
SciPy: For saving audio files.
Hugging Face Hub: For downloading AI models.
Python-dotenv: For managing environment variables.


Other: Git for version control, npm for frontend package management.

Implementation

Frontend (jingly-frontend):
Built with React and Vite for a fast, responsive UI.
JingleForm.jsx: Collects user inputs (brand name, motto, product brief, genre, optional reference audio) and sends them to the backend via a multipart/form-data POST request to /generate-jingle.
JingleResult.jsx: Displays the generated jingle with an audio player and download link.
LoadingScreen.jsx: Shows a loading state during jingle generation.
Proxy setup in vite.config.js forwards /api requests to http://localhost:8000.


Backend (jingly-AI-new):
FastAPI server handles /generate-jingle (POST) to process inputs and generate jingles.
Uses Gemini API to generate a music prompt and lyrics based on user inputs.
MusicGen generates a 30-second instrumental track (jingle_music.wav).
Bark generates vocals from lyrics (jingle_voice.wav).
PyDub combines music and vocals into a final jingle (final_jingle.wav).
Serves audio files via /outputs/{filename} (GET).
Models are stored in models/ and cached in ~/.cache/suno/bark_v0 for Bark.
Logs are saved to logs/jingle_generator.log for debugging.



Installation and Execution Instructions
Prerequisites

Node.js (v16 or later): For running the frontend.
Python (3.10.11+ recommended): For the backend.
Git: For cloning the repository.
At least 80GB free disk space: For AI models (4GB for Bark, ~2GB for MusicGen) and outputs (300MB).
NVIDIA GPU (optional): For faster AI model processing with CUDA-enabled PyTorch.

Backend Setup (jingly-AI-new)

Clone the Repository (if not already cloned):git clone <repository-url>
cd jingly-AI-new


Set Up Virtual Environment:python -m venv venv
.\venv\Scripts\activate


Install Dependencies:pip install audiocraft bark pydub google-generativeai huggingface_hub torch scipy python-dotenv fastapi uvicorn

Optional (GPU): Install CUDA-enabled PyTorch for faster processing:pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install xformers


Set Up Environment Variables:
Create a .env file in jingly-AI-new:echo GEMINI_API_KEY=your_gemini_api_key > .env

Replace your_gemini_api_key with your Google Generative AI key.


Verify Models:
Ensure models/bark-small contains text_2.pt, coarse_2.pt, fine_2.pt.
Ensure models/musicgen-small exists or will be downloaded.
Check with:dir models\bark-small
dir models\musicgen-small




Run the Backend:uvicorn main:app --host 0.0.0.0 --port 8000


The server should run on http://localhost:8000.



Frontend Setup (jingly-frontend)

Navigate to Frontend Directory:cd ../jingly-frontend


Install Dependencies:npm install


Run the Development Server:npm run dev


Access the app at http://localhost:5173.
If port 5173 is in use, check for conflicts:netstat -aon | findstr :5173
taskkill /PID <pid> /F

Or change the port in vite.config.js to 3000 and access http://localhost:3000.



Execution

Start Backend:
Ensure the backend is running (uvicorn main:app --host 0.0.0.0 --port 8000).


Start Frontend:
Run npm run dev in jingly-frontend.


Use the App:
Open http://localhost:5173 (or http://localhost:3000 if changed).
Fill out the form (e.g., brand_name: Krawings, motto: Crave Healthy, Live Happy, product_brief: Healthy snacks, genre: Upbeat Pop).
Submit to generate a jingle.
Wait ~10-20 minutes (CPU) or less (GPU) for the jingle to generate.
Play or download the jingle from the result page.


Check Logs:
Backend logs: jingly-AI-new/logs/jingle_generator.log.
Frontend errors: Browser console (F12).



Troubleshooting

404 Error on http://localhost:5173:
Ensure npm run dev is running in jingly-frontend.
Reinstall dependencies:rm -rf node_modules package-lock.json
npm install
npm run dev




API Errors:
Ensure backend is running on http://localhost:8000.
Test API:curl -X POST http://localhost:8000/generate-jingle -F "brand_name=Krawings" -F "motto=Crave Healthy, Live Happy" -F "product_brief=Healthy snacks" -F "genre=Upbeat Pop"




Slow Generation: Use a GPU with CUDA-enabled PyTorch or test with shorter lyrics.
Model Issues: Verify text_2.pt, coarse_2.pt, fine_2.pt in models/bark-small. Download from https://huggingface.co/suno/bark-small if missing.

Screenshots
(Add screenshots here after generating them)  

Home Page: [Placeholder for screenshot of the Jingly AI form]
Loading Screen: [Placeholder for screenshot of the loading animation]
Result Page: [Placeholder for screenshot of the audio player with download link]

To generate screenshots:

Run the app (npm run dev and uvicorn main:app).
Take screenshots of the form, loading screen, and result page.
Save as screenshot1.png, screenshot2.png, etc., in jingly-frontend.
Update this section with:![Home Page](screenshot1.png)
![Loading Screen](screenshot2.png)
![Result Page](screenshot3.png)



Project Demo Video (Optional)
(Add link to demo video if available)  

Upload a video showcasing the app (form submission, loading, result) to YouTube or another platform.
Update this section with:Watch the demo video: [Link to video]



License
This project is licensed under the MIT License. See the LICENSE file for details.