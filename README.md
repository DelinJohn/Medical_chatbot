
# Medical Prescription Assistant

The Medical Prescription Assistant is an AI-powered tool that provides personalized prescription recommendations based on patient history and symptom analysis. Using Langchain, FAISS, Groq, and NVIDIA technologies, the system quickly retrieves relevant medical information and formulates tailored treatment plans.


## Authors

- [@delin_shaji_john](https://github.com/DelinJohn)


## Table of Contents
- Overview
- Features
-  Architecture
-  Usage
- Run Locally
- Sample Input and Output
- Strengths and Limitations
- Use Cases
- Workflow Flowchart


###  Overview
The Medical Prescription Assistant is designed to streamline personalized healthcare by using a patient’s medical history and current symptoms to recommend relevant medications. It is built for scalability and efficiency, with modular vector storage and support for rapid embedding-based similarity search.
### Features
- Personalized Treatment Recommendations: Uses vectorized patient histories and symptoms for highly customized prescriptions.
- Efficient Data Retrieval: Utilizes FAISS for quick, efficient similarity search, enabling real-time recommendations.
- Lightweight Embeddings: Optimized with Nvidia embeddings for speed and reduced computational load.
- Configurable and Secure: API keys and sensitive information managed via environment variables, ensuring data privacy and security.
###  Architecture
The system integrates multiple components:

- Langchain: Orchestrates the processing pipeline and handles language modeling interactions.
- FAISS: Provides fast vector-based retrieval, storing vectorized patient histories for efficient querying.
- Groq and NVIDIA: Enable high-performance computing and model efficiency.
- Nvidia embeddings: Used for creating lightweight embeddings, balancing accuracy and computational demands.


### Usage
- #### Input

    The system takes in patient_id and symptoms. Here’s an      
    example:

        "patient_id": "[1-7500]",
        "symptoms": ["fever", "cough", "headache"]

- #### Output
   
    - "medications": 
    
            {"name": "Ibuprofen", "dosage": "400 mg", "frequency": "every 8 hours"},
            {"name": "Cough Syrup", "dosage": "10 ml", "frequency": "twice daily"}
            { "additional_instructions": "Take plenty of fluids and rest. Reassess in 3 days if symptoms persist."}
  
## Run Locally

Clone the project

```bash
  git clone https://github.com/DelinJohn/Medical_chatbot
```

Go to the project directory

```bash
  cd Medical_chatbot
```

Install dependencies

```bash
  pip Install -r requirements.txt
```

Start the server

```bash
  streamlit run app.py
```


### Strengths and Limitations
- #### Strengths
    - Personalization: Tailored treatment based on patient history and current symptoms.
    - Speed: Fast similarity-based retrieval with FAISS, suitable for real-time applications.
     - Lightweight and Deployable: Designed for efficient embeddings, making it lightweight and scalable.


- #### Limitations
     - Data Privacy: Compliance with data protection regulations is essential when handling sensitive patient data.
    - Connectivity Requirement: Relies on external APIs and internet access for Hugging Face Spaces, which could impact availability.

### Use Cases
- Clinics and Hospitals: Assists healthcare professionals by suggesting medications based on symptom input, saving time in patient care.
- Telemedicine Applications: Integrates into virtual health platforms to offer preliminary prescription suggestions based on symptom descriptions.
- Pharmaceutical Consultation: Supports pharmacists in checking medication compatibility and dosage based on patient-specific factors.
### Workflow-Chart
    https://drive.google.com/file/d/1RHF35U1nYEWNzsREilqH6_Lc9RgJ8CIB/view?usp=sharing
