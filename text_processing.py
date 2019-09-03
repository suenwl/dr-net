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

# Use pytesseract to perform OCR and obtain text from an invoice, which is a PDF-like object
def convert_image_based_invoice_to_text(invoice):
    config = r'--psm "4"' # tesseract config parameter, psm 4 treats the PDF as one block of text
    raw_OCR_output = pytesseract.image_to_data(None, config=config, output_type="data.frame") #Replace None object with an image-like object

    return raw_OCR_output

# Clean the rather dirty raw OCR output to remove meaningless tokens
def clean_OCR_output(raw_OCR_output):
    
    UNECESSARY_CHARACTERS = [" ", "(", ")", "&"]
    
    without_null = raw_OCR_output.loc[text["text"].isnull() == False]
    without_unecessary_characters = without_null.loc[~text["text"].isin(UNECESSARY_CHARACTERS)] # Not in unecessary characters

    
    return without_unecessary_characters