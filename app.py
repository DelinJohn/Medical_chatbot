# Importing required libraries


from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.pydantic_v1 import BaseModel,Field
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
import ast
import streamlit as st

 

# Accessing the values
LANGCHAIN_TRACING_V2=st.secrets['LANGCHAIN_TRACING_V2']
LANGCHAIN_API_KEY=st.secrets['LANGCHAIN_API_KEY']
LANGCHAIN_PROJECT=st.secrets['LANGCHAIN_PROJECT']  
groq_api_key = st.secrets["GROQ_API_KEY"]
langchain_endpoint = st.secrets["LANGCHAIN_ENDPOINT"]
NVIDIA_API=st.secrets['NVIDIA_API']



# Loading LLM
def load_llm(model):
    llm = ChatGroq(model=model, api_key=groq_api_key,temperature=0.5) 
    return llm


#Loading the database
db=SQLDatabase.from_uri('sqlite:///patients.db')


# Embeddings
embeddings=NVIDIAEmbeddings(model="nvidia/nv-embedqa-mistral-7b-v2",api_key=NVIDIA_API)


# Loading vector stores 
Antimicrobial_Regimen_Selection = FAISS.load_local("Antimicrobial_Regimen_Selection", embeddings, allow_dangerous_deserialization=True)
Central_Nervous_System_Infections = FAISS.load_local("Central_Nervous_System_Infections", embeddings, allow_dangerous_deserialization=True)
Endocarditis = FAISS.load_local("Endocarditis", embeddings, allow_dangerous_deserialization=True)
Fungal_Infections = FAISS.load_local("Fungal_Infections", embeddings, allow_dangerous_deserialization=True)
Gastrointestinal_Infections = FAISS.load_local("Gastrointestinal_Infections", embeddings, allow_dangerous_deserialization=True)
Human_Immunodeficiency_Virus = FAISS.load_local("Human_Immunodeficiency_Virus", embeddings, allow_dangerous_deserialization=True)
Influenza = FAISS.load_local("Influenza", embeddings, allow_dangerous_deserialization=True)
Intra_Abdominal_Infections = FAISS.load_local("Intra_Abdominal_Infections", embeddings, allow_dangerous_deserialization=True)
Respiratory_Tract_Infections_Lower = FAISS.load_local("Respiratory_Tract_Infections_Lower", embeddings, allow_dangerous_deserialization=True)
Respiratory_Tract_Infections_Upper = FAISS.load_local("Respiratory_Tract_Infections_Upper", embeddings, allow_dangerous_deserialization=True)
Sepsis_and_Septic_Shock = FAISS.load_local("Sepsis_and_Septic_Shock", embeddings, allow_dangerous_deserialization=True)
Sexually_Transmitted_Diseases = FAISS.load_local("Sexually_Transmitted_Diseases", embeddings, allow_dangerous_deserialization=True)
Skin_and_Soft_Tissue_Infections = FAISS.load_local("Skin_and_Soft_Tissue_Infections", embeddings, allow_dangerous_deserialization=True)
Surgical_Prophylaxis = FAISS.load_local("Surgical_Prophylaxis", embeddings, allow_dangerous_deserialization=True)
Tuberculosis = FAISS.load_local("Tuberculosis", embeddings, allow_dangerous_deserialization=True)
Urinary_Tract_Infections = FAISS.load_local("Urinary_Tract_Infections", embeddings, allow_dangerous_deserialization=True)


## Vector stores


vectorstores = {
    
    "Central_Nervous_System_Infections": Central_Nervous_System_Infections,
    "Endocarditis": Endocarditis,
    "Fungal_Infections": Fungal_Infections,
    "Gastrointestinal_Infections": Gastrointestinal_Infections,
    "Human_Immunodeficiency_Virus": Human_Immunodeficiency_Virus,
    "Influenza": Influenza,
    "Intra_Abdominal_Infections": Intra_Abdominal_Infections,
    "Respiratory_Tract_Infections_Lower": Respiratory_Tract_Infections_Lower,
    "Respiratory_Tract_Infections_Upper": Respiratory_Tract_Infections_Upper,
    "Sepsis_and_Septic_Shock": Sepsis_and_Septic_Shock,
    "Sexually_Transmitted_Diseases": Sexually_Transmitted_Diseases,
    "Skin_and_Soft_Tissue_Infections": Skin_and_Soft_Tissue_Infections,
    "Surgical_Prophylaxis": Surgical_Prophylaxis,
    "Tuberculosis": Tuberculosis,
    "Urinary_Tract_Infections": Urinary_Tract_Infections}

## Router LLM

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal[
                        'Central_Nervous_System_Infections',
                        'Endocarditis',
                        'Fungal_Infections',
                        'Gastrointestinal_Infections',
                        'Human_Immunodeficiency_Virus',
                        'Influenza',
                        'Intra_Abdominal_Infections',
                        'Respiratory_Tract_Infections_Lower',
                        'Respiratory_Tract_Infections_Upper',
                        'Sepsis_and_Septic_Shock',
                        'Sexually_Transmitted_Diseases',
                        'Skin_and_Soft_Tissue_Infections',
                        'Surgical_Prophylaxis',                                   
                        'Tuberculosis',
                        'Urinary_Tract_Infections'] = Field(
        ...,
        description="Given different vectorstores, choose the correct vector store based on the symptoms."
    )









## Function for the Router Query

def router(symptoms,llm):



    """
    Routes the given symptoms to the appropriate vector store based on the likely infection.

    This function leverages a language model to analyze the input symptoms and determine the corresponding 
    infection. It selects the appropriate vector store, where each store corresponds to a specific disease 
    or infection. The router uses a prompt to query the model and returns both the symptoms and the selected 
    vector store.

    Parameters:
    ----------
    symptoms : str
        A string representing the symptoms provided by the user, which will be used to identify the likely infection.

    Returns:
    -------
    tuple:
        - symptoms (str): The same symptoms that were passed as input.
        - answer.datasource (str or None): The name of the vector store that corresponds to the infection 
          (based on the symptoms). If an exception occurs, returns `None` for the vector store.
    """

    structured_llm_router = llm.with_structured_output(RouteQuery)


    system="""you are an expert at understanding what sypmtpoms corresponds to which infection based on your
      understanding select the right vector store
    . Each vector store correstponds to the disease as given in the vectorstore name """

    try:
        route_prompt=ChatPromptTemplate.from_messages(


            [
                ("system",system),
                ("human","{question}")
            ]
        )
        question_router=route_prompt | structured_llm_router
        
        answer=question_router.invoke(
        {'question':symptoms}
        )
        return symptoms,answer.datasource
    except Exception as e:
        return symptoms,None




## This LLM is resposible for providing the first context  for the  ChatBot for Antimicrobial Regimen Selection

def context_llm(symptom,patient_history,llm):



    """
    Retrieves the basic treatment criteria for a given infection based on the provided symptoms and patient history.

    This function interacts with a language model to retrieve an initial context that includes recommended 
    medication, regimens, and instructions for treatment. It uses the patient's history and the symptoms 
    as input to generate a contextual foundation for a more specific model, which will refine the treatment 
    recommendation.

    Parameters:
    ----------
    symptom : str
        A string representing the symptoms presented by the patient.
    patient_history : str
        A string containing the patient's medical history, which may include past treatments, conditions, 
        and other relevant medical information.

    Returns:
    -------
    str:
        A string containing the initial treatment recommendation, including basic regimens, medications, and 
        instructions, which can be used as the context for a more specific LLM model.
    """



    
    retriver=Antimicrobial_Regimen_Selection.as_retriever()
    prompt = ChatPromptTemplate.from_template(
    """
    Fetch the short and required regimen, medication, and instructions based on the following input:



**Patient History:** 
{patient_history}

**Context:**
{context}

    """
    )
    doc_chain=create_stuff_documents_chain(llm,prompt)
    chain=create_retrieval_chain(retriver,doc_chain)
    result=chain.invoke({'input':symptom,'patient_history':patient_history})
    return result['answer']




# For retriving patient history

def patient_history(patient_id):


    """
    Retrieves patient details based on the given patient ID and returns them as a formatted string.

    This function queries the patient history from the database using the provided patient ID. It extracts 
    relevant details such as ethnicity, gender, age, and the medications the patient is taking for diabetes, 
    and returns this information as a descriptive string.

    Parameters:
    ----------
    patient_id : int or str
        The unique identifier for the patient whose history is to be retrieved.

    Returns:
    -------
    str or None:
        A string summarizing the patient's details (ethnicity, gender, age, medications) if successful. 
        Returns None if an error occurs during the query or processing.
    """




    try:
        query = f'SELECT * FROM patient_history WHERE patient_Id = {patient_id}'
        data = db.run(query)
        
        
        data = ast.literal_eval(data)
        patient_id, ethinicity, gender, age, *medication = data[0]
        a = [i for i in medication if i is not None]
        
        return f"A {gender} of age {age} of {ethinicity} ethnicity is taking {', '.join(a)} for diabetes"
    
    
    except Exception as e:
        
        return None
    
# Core Chat bot funtion    



def chat_bot(symptoms, patient_Id,model):



    """
    The main chatbot function that processes patient symptoms and history to provide appropriate medical recommendations.

    This function integrates multiple components to analyze patient data and symptoms, fetch relevant patient history,
    retrieve a suitable vector store (infection-related data), and generate a set of recommended medications using an 
    LLM-based model. It ensures that the suggested medications are compatible with the patient's medical history 
    and provides detailed reasoning for each recommendation.

    Parameters:
    ----------
    symptoms : str
        A description of the patient's symptoms which will be analyzed to determine the infection.
    
    patient_Id : int or str
        The unique identifier of the patient for retrieving their medical history.

    Returns:
    -------
    None
        The function outputs the medical recommendations directly via a Streamlit interface (`st.write`).
        It may also provide error messages if symptoms or patient details are invalid.

    Exceptions:
    -----------
    ValueError:
        Raised if there is an issue with the symptoms or patient ID not matching the expected criteria.
    
    Exception:
        A general exception that handles unexpected errors during the process.
    """



    try:
        llm=load_llm(model=model)
        cause, store = router(symptoms,llm)
        patient_details = patient_history(patient_Id)
        
        if llm is None :
            raise ValueError("Entered api Key is wrong")
        

        if (store is None) and (patient_details is None):
            raise ValueError("Both patient details and symptoms do not match the given criteria.")
        
        if store is None:
            raise ValueError("The symptoms you mentioned do not correspond to any infection.")
        
        if patient_details is None:
            raise ValueError("The requested patient ID is not in the database.")

        retriever = vectorstores[store].as_retriever() if store else None
        general_details = context_llm(symptoms, patient_details,llm=llm)

        
        prompt = ChatPromptTemplate.from_template(
        """
You are a doctor’s assistant. Based on the following details, suggest 2-3 appropriate medications for the patient. Ensure that the first medication is the first line of treatment, followed by an alternative option, and a third option if needed. Include specific dosages, duration, rationale for each, and any potential interactions. Make sure to consider the patient's history and indicate any medications that should be avoided.

<patient history>
{patient_history}
</patient history>

<general_details>
{general_details}
</general_details>

<context>
{context}
</context>

**Important Notes:**
- Review the patient history carefully to avoid any medications that may interact with pre-existing conditions or current medications.
- Clearly state any medications that should be avoided based on the patient history.
- Consider the information in `general_details`.
- Avoid making assumptions; provide details strictly based on the data given.
- No hallucination or external data fetching should occur.

### Output Structure:
1. **Medication Recommendations:** (Dosages should consider the age of the patient from the patient history)
    - **First Line of Treatment:** Medication 1: [Name], [Dosage], [Frequency], [Duration]  
      *(Refer to the context for the first line of treatment based on the given symptoms)*
    - **Alternative Option:** Medication 2: [Name], [Dosage], [Frequency], [Duration]  
      *(Refer to the context for the alternative medication)*
    - **Additional Option (if needed):** Medication 3: [Name], [Dosage], [Frequency], [Duration]

2. **Rationale for Each Medication:** (Consider patient history)
    - Provide an explanation for why each medication is chosen, focusing on its effectiveness and safety for the patient.

3. **Medications to Avoid:** (Strictly based on the condition from patient history)
    - List any medicine (not just prescribed but any) should be avoided based on the patient's history and potential interactions.

4. **Patient Instructions:** (Consider patient history)
    - Provide detailed guidance on how to take each medication (e.g., with food, time of day).
    - Include reminders about completing the full course of treatment.

5. **Monitoring and Safety Recommendations:** (Also consider patient history)
    - Offer specific guidance on what to monitor in the patient (e.g., side effects, periodic tests).
    - Recommend follow-up actions in case of adverse reactions or lack of improvement.

Now, generate the recommendations.
"""

        )

        
       
        doc_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(retriever, doc_chain)
        st.write(f"Fetching recommendations for patient {patient_Id} with symptoms: {symptoms}")
        st.write(f"Using model: {model}")
        # Invocation of the chain with the right variables
        answer= chain.invoke({
            'input': cause,
            'patient_history': patient_details,
            'general_details': general_details
        })
        
        # You can display a loading message or spinner while fetching the response
        # Example:
        with st.spinner('Processing...'):
            st.write(patient_details)
            st.write(store)
            st.write(answer['answer'])
       

    except ValueError as ve:
        reason =(f"Error: {str(ve)}")
        st.write(reason)
    except Exception as e:
        reason=(f"An error occurred: {str(e)}")
        st.write(reason)   
    

def main():

        st.title("Medical Prescription Assistant")
        st.write("Enter patient details and symptoms to receive appropriate medication recommendations.")



        model = st.selectbox(
            "Choose a model:",
            options=["llama3-groq-70b-8192-tool-use-preview", "mixtral-8x7b-32768", "gemma-7b-it"]
        )
        patient_id = st.text_input("Enter Patient ID")
        symptoms = st.text_area("Enter symptoms")

        # Button to get recommendations
        if st.button("Get Medication Recommendation"):
            if symptoms and patient_id:
                chat_bot(symptoms, patient_id, model)
            else:
                st.error("Please provide both symptoms and patient ID.")

if __name__ == "__main__":
    main()