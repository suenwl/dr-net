# Text processing methods go here
from invoice2data import extract_data
from invoice2data.extract.loader import read_templates

def convert_text_to_result (text):

    result = None # TODO: Use invoice2data or other means to obtain results using text from invoice

    return result

def convert_text_based_pdf_to_result(invoice):
    templates = read_templates("./templates")
    
    # TODO: Implement data extraction using invoice2data

    return None