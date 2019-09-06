import fitz
from PIL import Image


def convert_pdf_to_image(PDF_path):
    document = fitz.open(PDF_path, width=1653, height=2339)
    pages = []
    for page in document:
        zoom = 3
        mat = fitz.Matrix(
            zoom, zoom
        )  # This is done because getPixmap would otherwise pick the default resolution
        pix = page.getPixmap(matrix=mat)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


# Util function to load invoices
def load_invoices():

    # TODO: Load the invoices, store in list

    return None  # Eventually return a list of invoices


# Util function to determine if an invoice is scanned or text based
def is_invoice_text_based(invoice):

    # TODO: add code to determine if an invoice is text or image based

    return True  # Eventually return result

