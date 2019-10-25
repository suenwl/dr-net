# OCR toolkit
import pytesseract
import pandas as pd
from PIL import Image
from typing import Dict
from math import sqrt
import re
import datetime


class Token:
    def __init__(
        self,
        text: str,
        coordinates: Dict[str, int],
        confidence: int,
        token_structure: Dict[str, int],
        category: str = "Others",
    ):
        self.text = text
        self.coordinates = coordinates
        self.confidence = confidence
        self.token_structure = token_structure
        self.category = category

        # feature related fields
        self.date_values = self.get_date()
        self.currency = self.get_currency()
        self.consumption_period = self.get_period()
        self.address = self.get_address()
        self.num_label = self.get_num_label()
        self.total_label = self.get_total_label()
        self.amount_label = self.get_amount_label()
        self.date_label = self.get_date_label()
        self.period_label = self.get_period_label()
        self.company = self.get_company()
        self.contains_digit = self.get_contains_digits()
        self.tax_label = self.get_tax_label()

    def __repr__(self):
        return self.text if self.text else str(self.token_structure)

    def __str__(self):
        return self.text if self.text else str(self.token_structure)

    # returns true if has at least one digit, false otherwise
    def get_contains_digits(self):
        if self.text:
            for c in self.text:
                if c.isdigit():
                    return True
            return False

    def get_tax_label(self):
        kw = ["gst", "tax"]
        negative_kw = ["excl","with","incl"]
        if self.text:
            words = self.text.lower().split(" ")
            no_negative_keywords = not any(word in self.text.lower() for word in negative_kw)
            keywords_exist = any(word in self.text.lower() for word in kw)
            if len(words)<4 and no_negative_keywords and keywords_exist:
                return True

    def get_date_label(self):
        kw = ["invoice", "bill", "issued", "receipt"]
        if self.text:
            if "date" in self.text.lower():
                if any(word in self.text.lower() for word in kw):
                    return (
                        self.text.lower()
                    )  # Return the entire text if it is specified that it is a invoice date
                else:
                    return "date"  # else just return date

    def get_period_label(self):
        kw = ["period"]
        if self.text:
            return any(word in self.text.lower() for word in kw)

    def get_company(self):
        kw = ["limited", "limited.", "ltd", "ltd."]
        if self.text:
            if any(word in self.text.lower() for word in kw):
                return self.text

    # tries to extract address from token
    def get_address(self):
        kw = [
            "drive",
            "dr",
            "road",
            "rd",
            "lane",
            "ln",
            "ave",
            "avenue",
            "street",
            "jalan",
            "jln",
            "boulevard",
            "blvd",
            "way",
            "park",
            "gate",
            "crescent",
            "cres",
            "singapore",
            "grove",
            "grv",
            "lorong",
            "lor",
            "valley",
            "view",
            "bukit",
            "tanjong",
        ]
        if self.text:
            text_array = self.text.lower().split(" ")
            for w in kw:
                for t in text_array:
                    if re.search("^" + w, t) and len(text_array) < 10:
                        return self.text.lower()

    # returns the text if "total" or some variant is contained in text and group is fewer than 5 words
    def get_total_label(self):
        kw = ["total"]
        if self.text:
            for w in kw:
                if w in self.text.lower() and len(self.text.split(" ")) < 5:
                    return self.text.lower()
    
    def get_amount_label(self):
        kw = ["amount","amt"]
        if self.text:
            for w in kw:
                if w in self.text.lower() and len(self.text.split(" ")) < 5:
                    return self.text.lower()

    # returns string for description of number, eg. account number, invoice number
    def get_num_label(self):
        kw = ["no", "no.", "no:", "no.:", "number", "num", "#", "#:"]
        if self.text:
            text_array = self.text.lower().split(" ")
            if any(word in text_array for word in kw):
                return self.text.lower()

    # returns a dictionary of {cur: <prefix> , value: <dollar amt> }
    # eg. {cur: $ , value: 5.00 }
    def get_currency(self):
        currencies = ["$", "¥", "dollar", "SGD", "USD", "US$", "SG$", "$SG", "$US"]
        out = {}
        for cur in currencies:
            if self.text and cur in self.text:
                out["cur"] = cur
                start = self.text.index(cur) + len(cur)
                for i in range(start, len(self.text)):
                    char = self.text[i]
                    if not char.isdigit() and not (char in [".", ","]):
                        break
                try:
                    out["val"] = float(self.text[start : i + 1])
                    return out
                except:
                    return None

    """     # checks if token is a date token
    def get_dates(self):
        # TODO parse range of dates and into date objects
        dates = []
        text = self.text
        if type(text) is str:
            month_names = set(
                [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ]
            )

            # checks for numerical months using regex
            text_nospaces = text.replace(" ", "")
            # matches d/dd or d-dd
            re_date = re.search("\d[/|-]\d\d", text_nospaces)
            if re_date:
                dates.append(text_nospaces)

            text_list = text.split(" ")

            # if token has multiple words
            if len(text_list) > 1:
                # checks for named months
                for index, word in enumerate(text_list):
                    for month in month_names:
                        if month in word.lower():
                            date = []
                            if index > 0:
                                date.append(text_list[index - 1])

                            date.append(text_list[index])

                            if index < len(text_list) - 1:
                                date.append(text_list[index + 1])

                            if len(date) > 1:
                                dates.append(" ".join(date))

            # token has only one word
            else:
                for month in month_names:
                    if month in text_list[0]:
                        dates.append(text_list[0])
        return dates """

    #matches 1-2 digits for date & month and 2-4 digits for year d/m/y or d-m-y
    #assumed to be in date then month then year format
    def regex_date_check(text_nospaces):
        re_date = re.search("\d{1,2}[/|-]\d{1,2}[/|-]\d{2,4}", text_nospaces)
        
        if re_date:
            fulldate = re_date.group(0)
        
            if len(fulldate.split("-"))>1:
                datelist = fulldate.split("-")
            else:
                datelist = fulldate.split("/")
            
            datelist = [ int(x) for x in datelist ]
    
             #convert separated dates (1st entry for date / mth, 2nd for date / month followed by year to a date object)
            day = datelist [0]
            month = datelist[1]
            year = datelist[2]
            if datelist[1]>12 and datelist[0]<=12:
                month = datelist[0]
                day = datelist [1]
            if datelist[2]<100:
                year = datelist[2]+2000 #adjust year for 2 digit representation
            date = datetime.datetime.strptime(str(day)+str(month)+str(year), "%d%m%Y").date()
            return date
    
     #Assumes if date is available, it is in 1 token
     #Creates date objects for consistency of formats
    def get_date(selftext):
        dates = []
        text = selftext
        if type(text) is str:
            month_names = [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ]
    
            # checks for numerical months using regex
            text_nospaces = text.replace(" ", "")
            if regex_date_check(text_nospaces):
                return regex_date_check(text_nospaces)
    
            text_list = text.split(" ")
            if len(text_list) >=1:
                # checks for named months
                for index, word in enumerate(text_list):
                    for month in month_names:
                        if month in word.lower():
                            mth = month_names.index(month)+1
                            day = 0
                            year = 0
                            if len(word)>3: #continue searching if it is in 1 token
                                re_date = re.search("\d{1,2}[/|-][a-zA-Z]{3}[/|-]\d{2,4}", word)
                                if re_date:
                                    fulldate = re_date.group(0)
                                    if len(fulldate.split("-"))>1:
                                        datelist = fulldate.split("-")
                                    else:
                                        datelist = fulldate.split("/")
                                    day = int(datelist[0])
                                    year = int(datelist[2])
                                    if int(datelist[2])<100:
                                        year = int(datelist[2])+2000 #adjust year for 2 digit representation
                                    date = datetime.datetime.strptime(str(day)+str(mth)+str(year), "%d%m%Y").date()
                                    return date
                            else: #for cases like '09', 'Aug', '2018'
                                for index2, word2 in enumerate(text_list):
                                    if word2!= word:
                                        if word2.isnumeric():
                                            if day == 0:
                                                if int(word2)<= 31:
                                                    day = int(word2)
                                            else:
                                                if int(word2) <100:
                                                    year = int(word2) + 2000
                                                else:
                                                    year = int(word2)
                                date = datetime.datetime.strptime(str(day)+str(mth)+str(year), "%d%m%Y").date()
                                return date
                
                        #last case to catch cases like '30JUL19', 'JUL3019', '30JUL2019','JUL302019'
                            if re.search("\d{1,2}[a-zA-Z]{3}\d{2,4}", word):
                                fulldate = re.search("\d{1,2}[a-zA-Z]{3}\d{2,4}", word).group(0).lower()
                                day = int(fulldate[:fulldate.find(month)])
                                year = int(fulldate[fulldate.find(month)+3:])
                                
                            elif re.search("[a-zA-Z]{3}\d{1,2}\d{2,4}", word):
                                fulldate = re.search("[a-zA-Z]{3}\d{1,2}\d{2,4}", word).group(0).lower()
                                if len(fulldate)<8:
                                    year = int(fulldate[-2:])
                                    day = int(fulldate[3:-2])
                                else:
                                    year = int(fulldate[-4:])
                                    day = int(fulldate[3:-4])
                            if year<100:
                                year+=2000
                                
                            date = datetime.datetime.strptime(str(day)+str(mth)+str(year), "%d%m%Y").date()
                            return date

    # checks if token itself is a consumption period and returns a boolean
    def is_consumption_period(self):
        text = self.text
        boolean = False
        if type(text) is str:
            month_names = set(
                [
                    "jan",
                    "feb",
                    "mar",
                    "apr",
                    "may",
                    "jun",
                    "jul",
                    "aug",
                    "sep",
                    "oct",
                    "nov",
                    "dec",
                ]
            )
             # checks for numerical months using regex
            text_nospaces = text.replace(" ", "")
            #possibilities -> same month or 2 diff months
        return boolean
        


    # returns a dict of 2 dates {earlier_date: date object, later_date: date object }
    # to be implemented
    def get_period(self):
                try:
                    return None
                except:
                    return None

    def set_category(self, category: str):
        self.category = category

    def is_horizontally_aligned_with(self, token):
        token_vertical_midpoint = (
            token.coordinates["y"] + token.coordinates["height"] / 2
        )
        return (
            self.coordinates["y"] < token_vertical_midpoint
            and token_vertical_midpoint
            < self.coordinates["y"] + self.coordinates["height"]
        )

    def is_vertically_aligned_with(self, token):
        token_horizontal_midpoint = (
            token.coordinates["x"] + token.coordinates["width"] / 2
        )
        return (
            self.coordinates["x"]
            < token_horizontal_midpoint
            < self.coordinates["x"] + self.coordinates["width"]
        )

    def is_aligned_with(self, token):
        return self.is_horizontally_aligned_with(
            token
        ) or self.is_vertically_aligned_with(token)

    def get_distance_to(self, token):
        token_vertical_midpoint = (
            token.coordinates["y"] + token.coordinates["height"] / 2
        )
        token_horizontal_midpoint = (
            token.coordinates["x"] + token.coordinates["width"] / 2
        )

        self_vertical_midpoint = self.coordinates["y"] + self.coordinates["height"] / 2
        self_horizontal_midpoint = self.coordinates["x"] + self.coordinates["width"] / 2

        return sqrt(
            (self_vertical_midpoint - token_vertical_midpoint) ** 2
            + (self_horizontal_midpoint - token_horizontal_midpoint) ** 2
        )

    def get_aligned_tokens(self, token_list, option="all"):
        if option == "all":
            return [
                token
                for token in token_list
                if self.is_horizontally_aligned_with(token)
                or self.is_vertically_aligned_with(token)
            ]
        elif option == "vertical":
            return [
                token for token in token_list if self.is_vertically_aligned_with(token)
            ]
        elif option == "horizontal":
            return [
                token
                for token in token_list
                if self.is_horizontally_aligned_with(token)
            ]
        else:
            raise Exception("Invalid option provided for get_aligned_tokens")

    def get_percentage_overlap(self, coordinates, image_size):
        image_width = image_size[0]
        image_height = image_size[1]

        token_x1 = self.coordinates["x"] / image_width
        token_x2 = token_x1 + self.coordinates["width"] / image_width
        token_y1 = self.coordinates["y"] / image_height
        token_y2 = token_y1 + self.coordinates["height"] / image_height
        token_area = abs(token_x1 - token_x2) * abs(token_y1 - token_y2)

        rect_x1 = coordinates["x"]
        rect_x2 = rect_x1 + coordinates["width"]
        rect_y1 = coordinates["y"]
        rect_y2 = rect_y1 + coordinates["height"]
        rect_area = abs(rect_x1 - rect_x2) * abs(rect_y1 - rect_y2)

        x1 = max(min(token_x1, token_x2), min(rect_x1, rect_x2))
        y1 = max(min(token_y1, token_y2), min(rect_y1, rect_y2))
        x2 = min(max(token_x1, token_x2), max(rect_x1, rect_x2))
        y2 = min(max(token_y1, token_y2), max(rect_y1, rect_y2))
        if x1 < x2 and y1 < y2:  # If there is an overlap
            overlap_area = abs(x2 - x1) * abs(y2 - y1)
            return overlap_area / (rect_area + token_area - overlap_area)
        else:
            return 0
