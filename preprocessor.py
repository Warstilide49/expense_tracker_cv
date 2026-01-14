import cv2
import pytesseract
import matplotlib.pyplot as plt

image_path = 'samples/normal.jpg' 
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
else:
    print("Image loaded successfully!")

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # BW Simple visualization
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Conversion Test')
    plt.show(block=False)
    plt.pause(0.001)

    # Raw OCR Output
    try:
        text = pytesseract.image_to_string(gray)
        print("--- Raw OCR Output ---")
        print(text)
        print("----------------------")
    except Exception as e:
        print(f"Tesseract Error: {e}")