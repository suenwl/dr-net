#%%
from PIL import Image
import pytesseract

#%%
print("Starting...")
INVOICE_PATH = "/Users/lxg/Documents/Semester Modules/BT3101 Capstone Project/PDF Invoices/OCR Test.jpg"
invoice = Image.open(INVOICE_PATH)
print(invoice.bits, invoice.size, invoice.format)
vanil_OCR_output = pytesseract.image_to_data(invoice, output_type="data.frame")
raw_OCR_output = pytesseract.image_to_data(invoice, output_type="data.frame", config='pitsync_linear_version==6, textord_noise_rejwords==0， textord_noise_rejrows==0')
#print(page)
#print(test_OCR(page))

#%%
