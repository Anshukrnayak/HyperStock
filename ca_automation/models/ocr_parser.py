import pytesseract
from PIL import Image
import io
import cv2
import numpy as np


def preprocess_image(img):
   img = np.array(img)
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
   return Image.fromarray(thresh)


def parse_invoice(file):
   try:
      img = Image.open(file)
      img = preprocess_image(img)
      text = pytesseract.image_to_string(img)
      description = text.split('\n')[0].strip() if text else "Unknown"
      amount = 0.0
      for line in text.split('\n'):
         if any(word in line.lower() for word in ['amount', 'total', 'rs', '₹']):
            try:
               amount = float(''.join(filter(str.isdigit, line)) or 0)
            except ValueError:
               continue
      return description, amount
   except Exception as e:
      print(f"Error in parse_invoice: {e}")
      return None, None