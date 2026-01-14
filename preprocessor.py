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

    # Adaptive thresholding - 
    # Description: local thresholding, im using Gaussian method for better results on varying lighting conditions
    # Parameters: blockSize=11, C=2
    thres1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    thres2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,21,4)
    thres3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,5,1)

    # Visualization
    titles = ['BW', 'Adaptive Thres 1', 'Adaptive Thres 2', 'Adaptive Thres 3']
    images = [gray, thres1, thres2, thres3]
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("plot.png", dpi=300)

    # Raw OCR Output
    try:
        text = pytesseract.image_to_string(thres2)
        print("--- Raw OCR Output ---")
        print(text)
        print("----------------------")
    except Exception as e:
        print(f"Tesseract Error: {e}")