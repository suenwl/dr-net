# OCR toolkit
import pytesseract

# Use pytesseract to perform OCR and obtain text from an invoice, which is a PDF-like object
def convert_image_based_invoice_to_text(invoice):
    config = r'--psm "4"' # tesseract config parameter, psm 4 treats the PDF as one block of text
    text = pytesseract.image_to_string(None, config=config) #Replace None object with an image-like object

    return text