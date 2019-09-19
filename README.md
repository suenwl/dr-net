# dr-net

A project to extract information from invoices

## Setup instructions
#### Clone repository
```git clone (copy link from git repo)```

#### Setup virtual environment
Change directory into project folder before executing the following ```cd dr-net```
```
pip install virtualenv
virtualenv env
source ./env/bin/activate
pip install -r requirements.txt
```
###### Additional notes for Windows users
If you don't see the virtual environemnt being activated (ie. no (env) in the command prompt), change directories to env/bin where u should see the activate.bat file. Simply run ```source activate``` or just ```activate```, and then return to the main project folder directory.

#### Install brew packages
tesseract is required for pytesseract
```
brew install tesseract
```
###### Additional notes for Windows users
If this doesn't work.  Instead, one can try downloading tesseract directly from https://pypi.org/project/pytesseract/#files. If there are still issues, it is likely because tesseract isn’t in your PATH. One way to resolve this is to change the “tesseract_cmd” variable. This can be done in the OCREngine file, where there is a commented out line with "pytesseract.pytesseract.tesseract_cmd =". Simply change this to wherever Tesseract OCR is installed on your system.

---

#### Download nltk data
```
python -m nltk.downloader all
```

#### After adding any packages, update the requirements.txt file before pushing
```pip freeze > requirements.txt```

### To run on Mac:
- Open terminal, run: ```source env/bin/activate```
- Then run: ```pip install -r requirements.txt```
- Run ```python main.py```

### To run on Windows:
- Open command prompt, run: ```source env/bin/activate``` (if this doesn't work refer to instructions above)
- Then run: ```pip install -r requirements.txt```
- Run ```python main.py```

#### Notes on VScode
We are using "black" as the official linter for code
