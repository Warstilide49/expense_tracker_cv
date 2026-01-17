import time
import cv2
import os
import easyocr
import re

# Receipt parser
def parse_receipt(results):
    text_lines = [item[1] for item in results]
    
    # This regex looks for digits, a comma, and exactly 2 digits at the end
    price_pattern = r'\d+[,]\d{2}'
    
    found_prices = []
    
    for line in text_lines:
        line = line.replace(' ', '')  # Remove spaces
        matches = re.findall(price_pattern, line)
        for match in matches:
            price = float(match.replace(',', '.'))
            found_prices.append(price)

    print(f"Found prices: {found_prices}")
    # This is with a hope that no other numbers are decimals except the prices
    max_price = max(found_prices) if found_prices else None    
    return max_price

def get_ocr_output(img, target_width=800):
    reader = easyocr.Reader(['de', 'en'], gpu=False)

    # Resizing image for faster processing
    scale_ratio = target_width / img.shape[1]
    new_height = int(img.shape[0] * scale_ratio)

    img_small = cv2.resize(img, (target_width, new_height))
    return reader.readtext(img_small)


imgs = [
    cv2.imread('samples/crumpled.jpg'),
    cv2.imread('samples/normal.jpg')
]

outputs = [get_ocr_output(img) for img in imgs]
totals = [parse_receipt(output) for output in outputs]
for i, total in enumerate(totals):
    print(f"Detected Total for image {i+1}: {total}")