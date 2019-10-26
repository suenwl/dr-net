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


currencies = [
    "SGD",
    "HKD",
    "JPY",
    "USD",
    "US$",
    "SG$",
    "$SG",
    "$US",
    "S$",
    "SINGAPORE DOLLAR",
    "$",
    "dollar",
    "¥",
]

category_mappings = {
    "Others": 0,
    "Account number": 1,
    "Consumption period": 2,
    "Country of consumption": 3,
    "Currency of invoice": 4,
    "Date of invoice": 5,
    "Invoice number": 6,
    "Name of provider": 7,
    "PO Number": 8,
    "Tax": 9,
    "Total amount": 10,
    0: "Others",
    1: "Account number",
    2: "Consumption period",
    3: "Country of consumption",
    4: "Currency of invoice",
    5: "Date of invoice",
    6: "Invoice number",
    7: "Name of provider",
    8: "PO Number",
    9: "Tax",
    10: "Total amount",
}


features_to_use = [
    # "char_count",
    # "word_count",
    # "height",
    # "width",
    "rel_dist_top",
    # "rel_dist_left",
    # "dist_bottom",
    # "dist_right",
    # "dist_top_outer",
    # "dist_left_outer",
    # "dist_bottom_outer",
    # "dist_right_outer",
    # "rel_size_page_x",
    # "rel_size_page_y",
    # "average_dist_neighbours_pixel",
    # "average_dist_neighbours_rel",
    # "average_dist_N_nearest_neighbours_pixel",
    # "average_dist_N_nearest_neighbours_rel",
    # "percentile_width",
    # "percentile_height",
    "contains_date",
    "contains_currency",
    "contains_specific_currency",
    "contains_date_range",
    "contains_address",
    # "contains_num_label", #
    # "contains_total_label", #
    # "contains_amount_label", #
    # "contains_date_label", #
    # "contains_date_of_invoice_label", #
    "contains_digit",
    "contains_company",
    # "contains_tax_label",
    # "vert_align_to_cell_w_date",
    "vert_align_to_cell_w_currency",
    # "vert_align_to_cell_w_address",
    "vert_align_to_cell_w_datelabel",
    "vert_align_to_cell_w_dateofinvoicelabel",
    # "vert_align_to_cell_w_numlabel",
    "vert_align_to_cell_w_totallabel",
    "vert_align_to_cell_w_amountlabel",
    # "vert_align_to_cell_w_digit",
    "vert_align_to_cell_w_invoicenum_label",
    "vert_align_to_cell_w_accountnum_label",
    "vert_align_to_cell_w_ponum_label",
    "vert_align_to_cell_w_tax_label",
    # "hori_align_to_cell_w_date",
    "hori_align_to_cell_w_currency",
    # "hori_align_to_cell_w_address",
    "hori_align_to_cell_w_datelabel",
    "hori_align_to_cell_w_dateofinvoicelabel",
    # "hori_align_to_cell_w_numlabel",
    "hori_align_to_cell_w_totallabel",
    "hori_align_to_cell_w_amountlabel",
    # "hori_align_to_cell_w_digit",
    "hori_align_to_cell_w_invoicenum_label",
    "hori_align_to_cell_w_accountnum_label",
    "hori_align_to_cell_w_ponum_label",
    "hori_align_to_cell_w_tax_label",
    # "rel_dist_nearest_cell_w_date",
    "rel_dist_nearest_cell_w_currency",
    # "rel_dist_nearest_cell_w_address",
    "rel_dist_nearest_cell_w_datelabel",
    "rel_dist_nearest_cell_w_invoicedatelabel",
    # "rel_dist_nearest_cell_w_numlabel",
    "rel_dist_nearest_cell_w_invoicenumlabel",
    "rel_dist_nearest_cell_w_accnumlabel",
    "rel_dist_nearest_cell_w_ponumlabel",
    "rel_dist_nearest_cell_w_totallabel",
    "rel_dist_nearest_cell_w_amountlabel",
    # "rel_dist_nearest_cell_w_digit",
    "rel_dist_nearest_cell_w_tax_label",
]
