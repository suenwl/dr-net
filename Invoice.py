from pdf2image import convert_from_path


class Invoice:
    def __init__(self, PDF_path):
        self.pages = convert_from_path(
            PDF_path, 500
        )  # Each of the individual pages in the PDF is converted to images
        self.is_text_based = (
            False
        )  # TODO: Need to implement way to check if PDF is text based

    def length(self):
        return len(self.pages)


def convert_text_to_result(text):

    result = (
        None
    )  # TODO: Use invoice2data or other means to obtain results using text from invoice

    return result


def convert_text_based_pdf_to_result(invoice):
    templates = read_templates("./templates")

    # TODO: Implement data extraction using invoice2data

    return None
