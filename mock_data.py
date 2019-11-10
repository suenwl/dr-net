import pandas as pd
import random
from datetime import datetime
import time
from random import choice

dateList = pd.date_range('2018-08-01','2019-08-01', freq='MS').strftime("%Y-%m-%d").tolist() + ['2018-12-04']

telco = ["Singtel", "StarHub Ltd", "M1", "CSL Mobile Limited", "NTT"]
hardware = ["Canon Singapore Pte Ltd", "Dell Singapore Pte Ltd"]
software = ["Oracle Corporation", "NCS Pte Ltd"]
consultants = ["BCG Pte Ltd", "Mckinsey & Company"]

allServices = telco + hardware + software + consultants

def generateNumber(typeOfNumber):
    
    if typeOfNumber == "Invoice":
        
        output = "00"
        
        for i in range(0,6):
            
            otherDigit = random.randrange(0,9)
            
            output += str(otherDigit)
            
    elif typeOfNumber == "Account":
        
        output = ""
        
        for i in range(0,8):
            
            otherDigit = random.randrange(0,9)
            
            output += str(otherDigit)
            
    elif typeOfNumber == "Confidence":
        
        output = str(random.uniform(0.8,0.9999999999))
        
    elif typeOfNumber == "PO Number":
        
        output = ""
        
        for i in range(0,5):
            
            otherDigit = random.randrange(0,9)
            
            output += str(otherDigit)
        
    return output


def getWarePriceRange(ware):
    
    totalCost = 0
    tax = 0
    
    if ware == "Singtel":
        
        priceRange = range(550,750)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    elif ware == "Starhub":
        
        priceRange = range(650,850)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    elif ware == "M1":
        
        priceRange = range(550,650)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    elif ware == "CSL Mobile Limited":
        
        priceRange = range(350,450)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0, 2)
        
    elif ware == "NTT":
        
        priceRange = range(950,1050)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    elif ware == "Canon Singapore Pte Ltd":
        
        priceRange = range(950,1250)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    elif ware == "Dell Singapore Pte Ltd":
        
        priceRange = range(950,1500)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    elif ware == "Oracle Corporation":
        
        priceRange = range(2050,2550)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    elif ware == "NCS Pte Ltd":
        
        priceRange = range(2550,3050)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    elif ware == "BCG Pte Ltd":
        
        priceRange = range(5000, 5500)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
    
    elif ware == "Mckinsey & Company":
        
        priceRange = range(7500, 8000)
        
        totalCost = round(choice(priceRange), 2)
        tax = round(0.07*totalCost, 2)
        
    return [str(totalCost), str(tax)]

dataFile = []

for i in dateList:
    
    for j in allServices:
        
        resultTemp = {}
        
        # Creating account number & invoice number, PO Number
        accountNumber = generateNumber("Account")
        confidenceNumberA = generateNumber("Confidence")
        account = [accountNumber, confidenceNumberA]
        invoiceNumber = generateNumber("Invoice")
        confidenceNumberB = generateNumber("Confidence")
        invoice = [invoiceNumber, confidenceNumberB]
        PONumber = generateNumber("PO Number")
        confidenceNumberC = generateNumber("Confidence")
        PO = [PONumber, confidenceNumberC]
        
        # Creating consumption period & date of invoice
        confidenceNumber2a = generateNumber("Confidence")
        consumption = [str(i), confidenceNumber2a]
        confidenceNumber2b = generateNumber("Confidence")
        date = [str(i), confidenceNumber2b]
        
        # Creating Country of Consumption & Currency, tax, total
        totalAmt = getWarePriceRange(j)
        total = totalAmt[0]
        tax = totalAmt[1]
        confidenceNumberD = generateNumber("Confidence")
        taxAmt = [tax, confidenceNumberD]
        confidenceNumberE = generateNumber("Confidence")
        totalAmt = [total, confidenceNumberE]
        confidenceNumberF = generateNumber("Confidence")
        provider = [str(j), confidenceNumberF]
        if j == "CSL Mobile Limited":
            confidenceNumber3a = generateNumber("Confidence")
            country = ["Hong Kong", confidenceNumber3a]
            confidenceNumber3b = generateNumber("Confidence")
            currency = ["HKD", confidenceNumber3b]
            confidenceNumber3c = generateNumber("Confidence")
            taxAmt = ["0.00", confidenceNumber3c]
        elif j == "NTT":
            confidenceNumber4a = generateNumber("Confidence")
            country = ["Japan", confidenceNumber4a]
            confidenceNumber4b = generateNumber("Confidence")
            currency = ["JPY", confidenceNumber4b]
        else:
            confidenceNumber5a = generateNumber("Confidence")
            country = ["Singapore", confidenceNumber5a]
            confidenceNumber5b = generateNumber("Confidence")
            currency = ["SGD", confidenceNumber5b]
            
        resultTemp = {"Account number": account, "Consumption period": consumption, "Country of consumption": country,
                      "Currency of invoice": currency, "Date of invoice": date, "Invoice number": invoice, 
                      "Name of provider": provider, "PO Number": PO, "Tax": taxAmt, "Total amount": totalAmt}
        
        dataFile += [resultTemp]
    