# dr-net

A project to extract information from invoices

## Setup instructions
#### Clone repository
```git clone (copy link from git repo)```

#### Setup virtual environment
```
pip install virtualenv
virtualenv env
source .env/bin/activate
pip install -r requirements.txt
```

#### Install brew packages
poppler is required for pdf2image to work
tesseract is required for pytesseract
```
brew install poppler
brew install tesseract
```

#### After adding any packages, update the requirements.txt file before pushing
```pip freeze > requirements.txt```
