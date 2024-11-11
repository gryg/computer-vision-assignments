import cv2
import numpy as np
from skimage import measure
from skimage.morphology import binary_dilation, binary_erosion
import os
from datetime import datetime
import glob

class SkinSegmentation:
    def __init__(self, input_dir="input_images", output_dir="output"):
        self.input_dir = input_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        self.create_output_directories()
        
        # Research-based color thresholds for skin detection
        # Based on: "Face Detection Using Color-based Segmentation and Template Matching"
        self.lower_hsv = np.array([0, 30, 60])
        self.upper_hsv = np.array([20, 150, 255])
        
        # Based on: "Human Skin Detection Using RGB, HSV and YCbCr Color Models"
        self.lower_ycbcr = np.array([0, 135, 85])
        self.upper_ycbcr = np.array([255, 180, 135])

    def create_output_directories(self):
        """Create directories for intermediate results"""
        subdirs = ["1_original", "2_skin_mask", "3_cleaned", "4_morphology", "5_final"]
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def get_input_images(self):
        """Get exactly 10 images from input directory"""
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(self.input_dir, ext)))
        
        if len(image_files) < 10:
            raise ValueError(f"Found only {len(image_files)} images. Need exactly 10.")
        return sorted(image_files)[:10]

    def detect_skin(self, image):
        """
        Detect skin using multiple color space thresholding
        Based on research papers that combine HSV and YCbCr for robust skin detection
        """
        # Convert to HSV and YCbCr color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Create masks in both color spaces
        mask_hsv = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask_ycbcr = cv2.inRange(ycbcr, self.lower_ycbcr, self.upper_ycbcr)
        
        # Combine masks using AND operation for stricter skin detection
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycbcr)
        
        return skin_mask

    def remove_noise(self, mask):
        """
        Remove small noisy components using connected component analysis
        """
        # Label connected components
        labels = measure.label(mask)
        props = measure.regionprops(labels)
        
        if not props:
            return np.zeros_like(mask)
        
        # Calculate areas and set threshold dynamically
        areas = [prop.area for prop in props]
        max_area = max(areas)
        min_area = max_area * 0.01  # Components smaller than 1% of largest are noise
        
        # Remove small components
        clean_mask = np.zeros_like(mask)
        for prop in props:
            if prop.area >= min_area:
                clean_mask[labels == prop.label] = 1
                
        return clean_mask

    def apply_morphology(self, mask):
        """
        Apply morphological operations as specified in the assignment
        """
        # Create kernel for morphological operations
        kernel = np.ones((5,5), np.uint8)
        
        # Dilate first to connect nearby components
        dilated = binary_dilation(mask, kernel)
        
        # Then erode to remove small protrusions
        result = binary_erosion(dilated, kernel)
        
        return result

    def detect_faces(self, mask):
        """
        Detect face regions using blob properties
        Based on typical face aspect ratios and roundness measures
        """
        labels = measure.label(mask)
        props = measure.regionprops(labels)
        
        faces = []
        for prop in props:
            # Calculate roundness using perimeter and area
            perimeter = prop.perimeter
            area = prop.area
            roundness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Check if blob has face-like properties
            if (roundness > 0.4 and  # Fairly round
                prop.eccentricity < 0.8 and  # Not too elongated
                prop.extent > 0.5):  # Reasonably solid
                
                # Get ellipse parameters
                y0, x0 = prop.centroid
                orientation = prop.orientation
                major_axis = prop.major_axis_length
                minor_axis = prop.minor_axis_length
                
                faces.append({
                    'center': (int(x0), int(y0)),
                    'axes': (int(major_axis/2), int(minor_axis/2)),
                    'angle': np.degrees(orientation)
                })
        
        return faces

    def process_image(self, image_path, index):
        """Process a single image through all steps"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        filename = f"{index:02d}_{os.path.splitext(os.path.basename(image_path))[0]}"
        
        # 1. Original image
        cv2.imwrite(os.path.join(self.output_dir, "1_original", f"{filename}.jpg"), image)
        
        # 2. Skin detection
        skin_mask = self.detect_skin(image)
        cv2.imwrite(os.path.join(self.output_dir, "2_skin_mask", f"{filename}.jpg"), 
                   skin_mask.astype(np.uint8) * 255)
        
        # 3. Noise removal
        clean_mask = self.remove_noise(skin_mask)
        cv2.imwrite(os.path.join(self.output_dir, "3_cleaned", f"{filename}.jpg"), 
                   clean_mask.astype(np.uint8) * 255)
        
        # 4. Morphological operations
        morphed_mask = self.apply_morphology(clean_mask)
        cv2.imwrite(os.path.join(self.output_dir, "4_morphology", f"{filename}.jpg"), 
                   morphed_mask.astype(np.uint8) * 255)
        
        # 5. Face detection and visualization
        faces = self.detect_faces(morphed_mask)
        result = image.copy()
        for face in faces:
            cv2.ellipse(result, 
                       face['center'], 
                       face['axes'], 
                       face['angle'], 
                       0, 360, 
                       (0, 255, 0), 
                       2)
        
        cv2.imwrite(os.path.join(self.output_dir, "5_final", f"{filename}.jpg"), result)
        
        return faces

    def process_all_images(self):
        """Process all images in the input directory"""
        image_paths = self.get_input_images()
        results = []
        
        for idx, path in enumerate(image_paths):
            print(f"Processing image {idx+1}/10: {path}")
            faces = self.process_image(path, idx)
            if faces is not None:
                results.append((path, faces))
        
        return results

def main():
    processor = SkinSegmentation("input_images", "output")
    results = processor.process_all_images()
    print(f"\nProcessing complete! Results saved in: {processor.output_dir}")

if __name__ == "__main__":
    main()