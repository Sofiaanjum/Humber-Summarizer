import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.chains.summarize import load_summarize_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import base64
import pyautogui

import docx #pip install python-docx
import PyPDF2
import markdown
from PIL import Image


# Load model directly
english_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
english_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

french_tokenizer = AutoTokenizer.from_pretrained("moussaKam/barthez")
french_model = AutoModelForSeq2SeqLM.from_pretrained("moussaKam/barthez-orangesum-abstract")


def generate_french(text, sum_range):
   inputs = french_tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
   input_ids = inputs.input_ids
   attention_mask = inputs.attention_mask
   
   output = french_model.generate(input_ids, attention_mask=attention_mask, 
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=sum_range[1],
                                 min_length=sum_range[0],
                                 no_repeat_ngram_size=3)
   return french_tokenizer.decode(output[0], skip_special_tokens=True)


def generate_english(text, sum_range):
    input_tokens = english_tokenizer.batch_encode_plus([text], return_tensors='pt', truncation=True)['input_ids']
    encoded_ids = english_model.generate(input_tokens,
                                 num_beams=4,
                                 length_penalty=2.0,
                                 max_length=sum_range[1],
                                 min_length=sum_range[0],
                                 no_repeat_ngram_size=3)
    return english_tokenizer.decode(encoded_ids.squeeze(), skip_special_tokens=True)


# file loader and preprocessing to extract text from pdf 
def file_preprocessing(file, doctype): #return str
    if doctype == '.pdf':
        loader = PyPDFLoader(file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        texts = text_splitter.split_documents(pages)
        final_texts = ""
        for text in texts:
            final_texts = final_texts + text.page_content
        #return final_texts
    elif doctype == '.docx':
        output_markdown = convert_docx_to_markdown(file)
        final_texts = str(output_markdown)
        #return final_texts
    elif doctype == '.txt':
        final_texts = str(file)

    return final_texts


# LLM pipeline
def llm_pipeline(filepath, language, doctype, sum_range=(50, 500)):
    input_text = file_preprocessing(filepath, doctype)
    if language == 'eng':
        result = generate_english(input_text, sum_range)
    else:
        result = generate_french(input_text, sum_range)
    return result


# display PDF
@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # embed PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    # display the file
    st.markdown(pdf_display, unsafe_allow_html=True)


# display DOC in markdown format
def convert_docx_to_markdown(input_file): #, output_file):
    doc = docx.Document(input_file)
    paragraphs = [p.text for p in doc.paragraphs]
    markdown_text = '\n'.join(paragraphs)

    return markdown_text


# Streamlit
st.set_page_config(layout='wide')


def main():
    st.image('humber_logo_copy.png',width = 180)
    st.title('Document Summarization')
    st.markdown("""The Humber Summarizer tool can accept plain-text, PDF, 
                or word documents. To begin, enter the text or document 
                below and then specify the appropriate language, 
                either English or French.\n\n L'outil Humber Summarizer peut accepter du
                texte brut, PDF ou Word. Pour commencer, entrez le texte ou le document ci-dessous, 
                puis spécifiez la langue appropriée, soit en Anglais ou soit en Français.""")
    text_input = st.text_input(label='Please enter text here / Entrez votre texte ici')

    uploaded_file = st.file_uploader("OR Upload a File (PDF or Word) / OU Téléchargez un Fichier (PDF ou Word)") #, type=['pdf'])
    lan_code = ''

    if (uploaded_file is not None) or len(text_input) > 0:
        if st.button("Summarize (English)", key="send_english_button"):
            lan_code = 'eng'
        if st.button("Résumer (Français)", key="send_french_button"):
            lan_code = 'fr'

        sum_range = st.slider("Select min and max summarization tokens", 25, 700, (50, 500), step=25)
        col1, col2 = st.columns(2)

        if len(text_input) > 0:
            # summarize the text input
            with col1:
                st.info("Uploaded Text")
                st.markdown(text_input)
            with col2:
                st.info("Summarization")
                if lan_code != '':
                    summary = llm_pipeline(text_input, language=lan_code, doctype='.txt', sum_range=sum_range)
                    st.markdown(summary)
                else:
                    st.markdown('Please select a language / Choisissez une langue')
        else:
            filepath = uploaded_file.name
            with open(filepath, 'wb') as temp_file:
                temp_file.write(uploaded_file.read())

            if filepath.endswith('.pdf'):

                with col1:
                    st.info("Uploaded PDF / PDF téléchargé")
                    pdf_viewer = displayPDF(filepath)
                with col2:
                    st.info("Summarization")
                    if lan_code != '':
                        summary = llm_pipeline(filepath, language=lan_code, doctype='.pdf', sum_range=sum_range)
                        st.markdown(summary)
                    else:
                        st.markdown('Please select a language / Choisissez une langue')
            elif filepath.endswith('.docx'):
                with col1:
                    st.info("Uploaded DOC")
                    #pdf_viewer = displayPDF(filepath)
                    output_markdown = convert_docx_to_markdown(filepath)
                    st.markdown(output_markdown, unsafe_allow_html=True)
                with col2:
                    st.info("Summarization / Récapitulation")
                    if lan_code != '':
                        summary = llm_pipeline(filepath, language=lan_code, doctype='.docx', sum_range=sum_range)
                        st.markdown(summary)
                    else:
                        st.markdown('Please select a language / Choisissez une langue')
            else:
                with col1:
                    st.info("Unsupported file format / Le format de fichier n'est pas pris en charge")


if __name__ == '__main__':
    main()