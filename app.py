# Importing required libraries



from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from pydantic import BaseModel,Field
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
import streamlit as st
import ast



# Accessing the values
groq_api_key = st.secrets["GROQ_API_KEY"]
langchain_endpoint = st.secrets["LANGCHAIN_ENDPOINT"]




# Loading LLM
def load_llm(model):
    llm = ChatGroq(model=model, api_key=groq_api_key,temperature=0.5) 
    return llm


#Loading the database
db=SQLDatabase.from_uri('sqlite:///patients.db')


# Embeddings
embeddings=OllamaEmbeddings(model='llama3.1')


# Loading vector stores 
abdominal_infection_store = FAISS.load_local(
    "abdominal_infection_store", embeddings, allow_dangerous_deserialization=True
)

central_nervous_system_infetions_store = FAISS.load_local(
    "central_nervous_system_infections_store", embeddings, allow_dangerous_deserialization=True
)

endocratis_store = FAISS.load_local(
    "endocratis_store", embeddings, allow_dangerous_deserialization=True
)

Fungal_Infections_Invasive_store = FAISS.load_local(
    "Fungal_Infections_Invasive_store", embeddings, allow_dangerous_deserialization=True
)

Gastrointestinal_Infections_store = FAISS.load_local(
    "Gastrointestinal_Infections_store", embeddings, allow_dangerous_deserialization=True
)

Human_Immunodeficiency_Virus_Infection_store = FAISS.load_local(
    "Human_Immunodeficiency_Virus_store", embeddings, allow_dangerous_deserialization=True
)

influenza_store = FAISS.load_local(
    "influenza_store", embeddings, allow_dangerous_deserialization=True
)

respiratory_tract_infection_Lower_store = FAISS.load_local(
    "respiratory_tract_infection_Lower_store", embeddings, allow_dangerous_deserialization=True
)

respiratory_tract_infection_Upper_store = FAISS.load_local(
    "respiratory_tract_infection_Upper_store", embeddings, allow_dangerous_deserialization=True
)

Sepsis_and_septic_shock_store = FAISS.load_local(
    "Sepsis_and_septic_shock_store", embeddings, allow_dangerous_deserialization=True
)

skin_infection_store = FAISS.load_local(
    "skin_infection_store", embeddings, allow_dangerous_deserialization=True
)

STDS_infection_store = FAISS.load_local(
    "STDS_infection_store", embeddings, allow_dangerous_deserialization=True
)

surgical_prolaxix_store = FAISS.load_local(
    "surgical_prolaxix_store", embeddings, allow_dangerous_deserialization=True
)

tuberclosis_store = FAISS.load_local(
    "tuberclosis_store", embeddings, allow_dangerous_deserialization=True
)

Urinary_infection_store = FAISS.load_local(
    "Urinary_infection_store", embeddings, allow_dangerous_deserialization=True

)

Infection_criteria_store=FAISS.load_local(
     "infection_criteria",embeddings,allow_dangerous_deserialization=True
)



## Vector stores


vector_store={
        'abdominal_infection_store':abdominal_infection_store,
        'central_nervous_system_infections_store':central_nervous_system_infetions_store,
        'endocarditis_store':endocratis_store,
        'fungal_infections_invasive_store':Fungal_Infections_Invasive_store,
        'gastrointestinal_infections_store':Gastrointestinal_Infections_store,
        'respiratory_tract_infection_lower_store':respiratory_tract_infection_Lower_store,
        'human_immunodeficiency_virus_infection_store':Human_Immunodeficiency_Virus_Infection_store,
        'sepsis_and_septic_shock_store':Sepsis_and_septic_shock_store,
        'skin_infection_store':skin_infection_store,
        'stds_infection_store':STDS_infection_store,
        'surgical_prolaxix_store':surgical_prolaxix_store,
        'tuberclosis_store':tuberclosis_store,
        'Urinary_infection_store':Urinary_infection_store,
        'influenza_store':influenza_store
}


## Router LLM

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    datasource: Literal[
        'abdominal_infection_store',
        'central_nervous_system_infections_store',
        'endocarditis_store',
        'fungal_infections_invasive_store',
        'gastrointestinal_infections_store',
        'respiratory_tract_infection_lower_store',
        'human_immunodeficiency_virus_infection_store',
        'sepsis_and_septic_shock_store',
        'skin_infection_store',
        'stds_infection_store',
        'surgical_prolaxix_store',
        'tuberclosis_store',
        'Urinary_infection_store',
        'influenza_store',
        
    ] = Field(
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



    
    retriver=Infection_criteria_store.as_retriever()
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

        retriever = vector_store[store].as_retriever() if store else None
        general_details = context_llm(symptoms, patient_details,llm=llm)

        
        prompt = ChatPromptTemplate.from_template(
        """
    You are a doctor’s assistant. Based on the following details, suggest 2-3 appropriate medications, along with specific dosages, duration, rationale for each, and any potential interactions. Ensure the medications do not interact with the patient's history.

    <patient history>
    {patient_history}
    </patient history>

    <general_details>
    {general_details}
    </general_details>

    <context>
    {context}
    </context>

    

    Output Structure:
    1. **Medication Recommendations:** 
        - Medication 1(Should Be first Choice if possible go With the Broad Spectrum): [Name], [Dosage], [Frequency], [Duration]   
        - Medication 2: [Name], [Dosage], [Frequency], [Duration]
        - Medication 3 (if needed): [Name], [Dosage], [Frequency], [Duration]

    2. **Rationale for Each Medication:**
        - Explanation for why each medication is chosen, with a focus on effectiveness and patient safety.

    3. **Patient Instructions:**
        - Detailed guidance on how to take the medication (e.g., with food, time of day).
        - Reminders about completing the full course of treatment.

    4. **Monitoring and Safety Recommendations:**
        - Specific guidance on what to monitor in the patient (e.g., side effects, periodic tests).
        - Recommendations for follow-up actions in case of adverse reactions or lack of improvement.

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