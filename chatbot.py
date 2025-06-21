from transformers import pipeline
import nltk
nltk.download('punkt')

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

def get_response(user_input):
    result = chatbot(user_input, max_length=50, clean_up_tokenization_spaces=True)
    return result[0]['generated_text']
