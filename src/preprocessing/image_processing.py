import cv2
import numpy as np

def preprocess_image(image_np, target_size=(512, 512)):
    # Resize
    img_resized = cv2.resize(image_np, target_size)
    
    # RGB to YCbCr
    img_ycbcr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2YCrCb)
    
    # Separate channels (Y is luma)
    y, cr, cb = cv2.split(img_ycbcr)
    
    # Median filtering on Y channel
    y_filtered = cv2.medianBlur(y, 5)
    
    # Contrast stretching (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    y_stretched = clahe.apply(y_filtered)
    
    # Merge back
    img_preprocessed_ycbcr = cv2.merge((y_stretched, cr, cb))
    img_preprocessed_rgb = cv2.cvtColor(img_preprocessed_ycbcr, cv2.COLOR_YCrCb2RGB)
    
    return img_preprocessed_rgb, y_stretched
