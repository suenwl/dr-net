# import methods defined in the other files
from OCREngine import OCREngine
from Invoice import Invoice
from Token import Token
from Classifier import Classifier
from util import *

# This method manages all the various methods we use to obtain data from invoices
# Our OCR, regex, and text model methods are all called from this method.
# This method is called once per invoice file that needs to be processed
def obtain_output(invoice):

    # Determine if invoice is text or image based
    IS_TEXT_BASED = is_invoice_text_based(invoice)

    # OCR step
    if not IS_TEXT_BASED:
        text = convert_image_based_invoice_to_text(invoice)
        result = convert_text_to_result(text)

    else:
        result = convert_text_based_pdf_to_result(invoice)

    # For unrecognised invoices, use a word/topic model to obtain classifications of text within invoice

    # TODO: Build and implement word model


if __name__ == "__main__":
    # Load invoices in specific folder
    invoice = Invoice(
        "/Users/suenwailun/Sync Documents/University/Y4S1/BT3101 Business Analytics Capstone Project/Tech demo/starhub.pdf"
    )
    page = invoice.get_page(1)
    page.do_OCR()
    classifier = Classifier()
    for token in page.tokens:
        print(token, classifier.create_features(token, page))
    # for i, block in page.get_tokens_by_block().items():
    #     print(block)
    #     print(" ")

    # Iterate through invoices and obtain output
    # for invoice in invoices:
    #     obtain_output(invoice)

    # Write to excel/csv file
