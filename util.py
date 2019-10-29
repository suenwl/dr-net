import fitz
from PIL import Image
from IPython.display import clear_output, display


def convert_pdf_to_image(PDF_path):
    document = fitz.open(PDF_path, width=1653, height=2339)
    pages = []
    num_pages = len(document)
    for i, page in enumerate(document):
        if i not in [0, 1, num_pages - 2, num_pages - 1]:
            pages.append(None)
            continue
        zoom = 3
        mat = fitz.Matrix(
            zoom, zoom
        )  # This is done because getPixmap would otherwise pick the default resolution
        pix = page.getPixmap(matrix=mat)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        pages.append(img)
    return pages


def print_progress(current, total, text):
    percentage = round(current / total * 100, 1)
    done = "#" * int(float(str(percentage / 2)))
    todo = "-" * (50 - int(float(str(percentage / 2))))
    string = (
        "{text}<{display}>".format(text=text, display=done + todo)
        + " "
        + str(percentage)
        + "%"
    )
    clear_output(wait=True)
    print(string, end="")


def print_tokens_and_features_with_category(invoices, category, selected_features):
    from FeatureEngine import FeatureEngine

    print("{:30.25} {:15}".format("Invoice name", "Token text"))

    for invoice in invoices:
        for page in invoice.pages:
            for token in page.grouped_tokens:
                if token.category == category:
                    generated_features = FeatureEngine.create_features(token, page)
                    features_to_print = []
                    for feature in selected_features:
                        features_to_print.append(generated_features[feature])
                    print(
                        "{:30.25} {:15}".format(invoice.readable_name, token.text),
                        *features_to_print
                    )


def display_invoice(invoices, name_of_invoice, page):
    for invoice in invoices:
        if invoice.readable_name == name_of_invoice:
            invoice.get_page(page).draw_bounding_boxes()
