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


def calc_min_dist(t1, t2):
    # get bounding outer rectangle
    outer_rect_left = min(t1.coordinates["x"], t2.coordinates["x"])
    outer_rect_top = min(t1.coordinates["y"], t2.coordinates["y"])
    outer_rect_bottom = max(
        (t1.coordinates["y"] + t1.coordinates["height"]),
        (t2.coordinates["y"] + t2.coordinates["height"]),
    )
    outer_rect_right = max(
        (t1.coordinates["x"] + t1.coordinates["width"]),
        (t2.coordinates["x"] + t2.coordinates["width"]),
    )
    outer_rect_width = outer_rect_right - outer_rect_left
    outer_rect_heigth = outer_rect_bottom - outer_rect_top
    inner_rect_width = max(
        0, outer_rect_width - (t1.coordinates["width"] + t2.coordinates["width"])
    )
    inner_rect_height = max(
        0, outer_rect_heigth - (t1.coordinates["height"] + t2.coordinates["height"])
    )
    pixel_dist = math.sqrt(inner_rect_width ** 2 + inner_rect_height ** 2)
    return pixel_dist


def is_vert_aligned(t1, t2, moe):
    """Returns true if t2 is vertically aligned with t1 and is below t1"""
    t2_below_t1 = t1.coordinates["y"] - t2.coordinates["y"] < 0
    if not t2_below_t1:
        return False

    if abs(t1.coordinates["x"] - t2.coordinates["x"]) < moe:
        return True
    if (
        abs(
            (t1.coordinates["x"] + t1.coordinates["width"])
            - (t2.coordinates["x"] + t2.coordinates["width"])
        )
        < moe
    ):
        return True
    t1_midpt_x = t1.coordinates["x"] + (t1.coordinates["width"] / 2)
    t2_midpt_x = t2.coordinates["x"] + (t2.coordinates["width"] / 2)
    if abs(t1_midpt_x - t2_midpt_x) < moe:
        return True
    return False


# checks if two tokens are aligned horizontally within a margin of error (checks midpoint, top boundary, bottom boundary)
def is_hori_aligned(t1, t2, moe):
    """Returns true if t2 is horizontally aligned with t1 and is to the right of t1"""
    t2_to_right_of_t1 = t1.coordinates["x"] - t2.coordinates["x"] < 0
    if not t2_to_right_of_t1:
        return False

    if abs(t1.coordinates["y"] - t2.coordinates["y"]) < moe:
        return True
    if (
        abs(
            (t1.coordinates["y"] + t1.coordinates["height"])
            - (t2.coordinates["y"] + t2.coordinates["height"])
        )
        < moe
    ):
        return True
    t1_midpt_y = t1.coordinates["y"] + (t1.coordinates["height"] / 2)
    t2_midpt_y = t2.coordinates["y"] + (t2.coordinates["height"] / 2)
    if abs(t1_midpt_y - t2_midpt_y) < moe:
        return True
    return False


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


def missing_fields_percentage(invoice_predictions):
    total = len(invoice_predictions)
    missing_counts = {k: 0 for (k, v) in invoice_predictions[0].items()}
    for prediction in invoice_predictions:
        for field in missing_counts.keys():
            if not prediction[field][0]:
                missing_counts[field] += 1

    return {k: v / total for (k, v) in missing_counts.items()}
