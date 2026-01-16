import cv2
import pytesseract
import matplotlib.pyplot as plt

image_path = 'samples/normal.jpg' 
image = cv2.imread(image_path)
print(f"Original Image Size: {image.shape}")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Adaptive thresholding
# Description: local thresholding, im using Gaussian method for better results on varying lighting conditions
# Parameters: blockSize=11, C=2
def adaptive_threshold(img, block_size, C):
    return cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block_size, C)

tinkered_values = {
    'block_size': 95,
    'C': 13,
    'blur': 3,
    'kernel_size': 10
}
def preprocess(img, tinkered_values = tinkered_values):
    if image is None:
        print(f"Error: Could not load image at {image_path}")
        return None
    
    print("Image loaded successfully!")

    # Extracting tinkered values
    block_size = tinkered_values['block_size']
    C = tinkered_values['C']
    blur = tinkered_values['blur']
    kernel_size = tinkered_values['kernel_size']

    # Blur
    img = cv2.medianBlur(gray, blur)
    # Adaptive Thresholding
    img = adaptive_threshold(img, block_size, C)
    # Morphological operation
    img = apply_morphology(img, kernel_size=10)

    # Visualization
    plt.imshow(img, cmap='gray')
    plt.title("Blur, AT, Morphology")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig("plot_morphed.png", dpi=300)
    return img

def nothing(x):
    pass

def cv2_tinkering(img):
    cv2.namedWindow('Tuner')
    cv2.resizeWindow('Tuner', 300, 300)
    cv2.createTrackbar('BlockSize', 'Tuner', 10, 100, nothing) 
    cv2.createTrackbar('C', 'Tuner', 4, 20, nothing)
    cv2.createTrackbar('Blur', 'Tuner', 3, 15, nothing)
    cv2.createTrackbar('Right', 'Tuner', 800, 4624, nothing)
    cv2.createTrackbar('Down', 'Tuner', 800, 3468, nothing)
    cv2.createTrackbar('Morph Kernel', 'Tuner', 2, 30, nothing)
    while True:
        bs_val = cv2.getTrackbarPos('BlockSize', 'Tuner')
        c_val = cv2.getTrackbarPos('C', 'Tuner')
        blur_val = cv2.getTrackbarPos('Blur', 'Tuner')
        # odd number constraint for BlockSize
        real_bs = bs_val if bs_val % 2 == 1 else bs_val + 1
        if real_bs < 3: real_bs = 3

        # odd number constraint for median blur kernel size
        blur_val = blur_val if blur_val % 2 == 1 else blur_val + 1
        if blur_val < 1: blur_val = 1

        right = cv2.getTrackbarPos('Right', 'Tuner')
        down = cv2.getTrackbarPos('Down', 'Tuner')
        right = 0 if right==None else right
        down = 0 if down==None else down
        cropped = img[down:down+500, right:right+800]
        cropped = cv2.medianBlur(cropped, blur_val)

        kernel_size = cv2.getTrackbarPos('Morph Kernel', 'Tuner')
        kernel_size = kernel_size if kernel_size>2 else 2
        morphed = apply_morphology(cropped, kernel_size)

        th = cv2.adaptiveThreshold(morphed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, real_bs, c_val)

        cv2.imshow('Tuner', th)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def print_ocr_output(img):
    try:
        custom_config = r'-l deu --oem 3 --psm 6'
        text = pytesseract.image_to_string(img, config=custom_config)
        print("--- Raw OCR Output ---")
        print(text)
        print("----------------------")
    except Exception as e:
        print(f"Tesseract Error: {e}")

# Morphological operation
# Description: removes noise by performing opening(erosion + dilation) operation
def apply_morphology(img, kernel_size=3):
    # line below is the same as np.ones((kernel_size, kernel_size), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opened

preprocessed_image = preprocess(image)
if preprocessed_image is not None:
    print_ocr_output(preprocessed_image)
# cv2_tinkering(gray) 