# dr-net

A project to extract information from invoices

## Setup instructions
#### Clone repository
```git clone (copy link from git repo)```

#### Setup virtual environment
Change directory into project folder before executing the following
```
pip install virtualenv
virtualenv env
source ./env/bin/activate
pip install -r requirements.txt
```

#### Install brew packages
tesseract is required for pytesseract
```
brew install tesseract
```

#### Download nltk data
```
python -m nltk.downloader all
```

#### After adding any packages, update the requirements.txt file before pushing
```pip freeze > requirements.txt```


#### Notes on VScode
We are using "black" as the official linter for code
