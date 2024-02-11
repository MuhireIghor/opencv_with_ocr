import cv2
import pytesseract
import numpy as np

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Binarize the image using adaptive thresholding
    bin_img = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    return bin_img

def detect_roi(bin_img):
    # Perform contour detection to find text regions
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get bounding boxes of text regions
    rois = [cv2.boundingRect(cnt) for cnt in contours]
    print(f'got {len(contours)} rois')

    return rois

def perform_ocr(image, rois, lang='eng'):
    # Configure Tesseract OCR
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    custom_config = r'--oem 3 --psm 6'  # Page segmentation mode and OEM mode

    extracted_text = []

    # Perform OCR on each ROI
    for roi in rois:
        x, y, w, h = roi
        roi_image = image[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi_image, lang=lang, config=custom_config)
        if  text.strip() != "":

            extracted_text.append(text.strip())

    return extracted_text

def post_process_text(text):
    # Perform post-processing steps such as spell checking, punctuation correction, etc.
    # Example: Text cleaning using regex or spell checking libraries
    # Example: Error correction based on dictionary lookup or contextual information
    return text

def main():
    # Image preprocessing
    bin_img = preprocess_image('reading.jpg')

    # ROI detection
    rois = detect_roi(bin_img)

    # Perform OCR
    extracted_text = perform_ocr(bin_img, rois)

    # Post-processing
    # corrected_text = [post_process_text(text) for text in extracted_text]

    # Validation and verification (Manual inspection or comparison with ground truth)

    # Print the extracted and corrected text
    #filter the extracted_text array to return only array elements with some content
    final_text=" ".join(extracted_text)
    print(final_text)


    # for i, text in enumerate(corrected_text):
    #     print(f"Extracted Text from ROI {i+1}: {text}")

if __name__ == "__main__":
    main()
