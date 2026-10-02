import cv2
import numpy as np

def extract_lesion_features(y_channel):
    # Thresholding for dark lesions (red) and bright lesions
    # y_channel is CLAHE enhanced. 
    # Bright lesions (exudates) have high intensity. Dark (hemorrhages) have low.
    
    # Simple thresholding for demonstration
    _, bright_mask = cv2.threshold(y_channel, 200, 255, cv2.THRESH_BINARY)
    _, dark_mask = cv2.threshold(y_channel, 50, 255, cv2.THRESH_BINARY_INV)
    
    def get_contour_features(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count = len(contours)
        if count == 0:
            return [0, 0, 0, 0]
        
        areas = []
        diameters = []
        solidities = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas.append(area)
            
            # Equivalent diameter
            diameter = np.sqrt(4 * area / np.pi)
            diameters.append(diameter)
            
            # Solidity
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = float(area) / hull_area
            else:
                solidity = 0
            solidities.append(solidity)
            
        return [
            count,
            np.mean(areas) if count > 0 else 0,
            np.mean(diameters) if count > 0 else 0,
            np.mean(solidities) if count > 0 else 0
        ]

    red_features = get_contour_features(dark_mask)
    bright_features = get_contour_features(bright_mask)
    
    # return 8 features
    return red_features + bright_features
