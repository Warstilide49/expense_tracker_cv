import cv2
import pytesseract
import matplotlib.pyplot as plt

image_path = 'samples/normal.jpg' 
image = cv2.imread(image_path)

# Adaptive thresholding
# Description: local thresholding, im using Gaussian method for better results on varying lighting conditions
# Parameters: blockSize=11, C=2
def adaptive_threshold(img, block_size, C):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)

def preprocess(img):
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    print("Image loaded successfully!")

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur
    img = cv2.medianBlur(gray,5)

    thres_with_blur = adaptive_threshold(img, 21, 4)
    
    # Visualization
    plt.imshow(thres_with_blur, cmap='gray')
    plt.title("AT with Blur")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)
    return thres_with_blur

def print_ocr_output(img):
    try:
        text = pytesseract.image_to_string(img)
        print("--- Raw OCR Output ---")
        print(text)
        print("----------------------")
    except Exception as e:
        print(f"Tesseract Error: {e}")

preprocessed_image = preprocess(image)
if preprocessed_image is not None:
    print_ocr_output(preprocessed_image)