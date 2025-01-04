from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import pdfplumber
import os

def extract_text_from_pdf(pdf_path, poppler_path = None):
    extracted_text = []

    #get text from main textual content
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text.append(text)
    except Exception as e:
        print(f"Error reading PDF with PyPDF2: {e}")

    #get text from tables
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                for table in tables:
                    table_text = "\n".join(["\t".join([str(cell) if cell else "" for cell in row])for row in table])
                    extracted_text.append(table_text)
    except Exception as e:
        print(f"Error reading tables with pdfplumber: {e}")

    #get text from images
    try:
        images = convert_from_path(pdf_path, poppler_path = poppler_path)
        for image in images:
            text_from_image = pytesseract.image_to_string(image)
            extracted_text.append(text_from_image)
    except Exception as e:
        print(f"Error extracting text from image: {e}")

    return "\n".join(extracted_text)
        

