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
        self.base_output_dir = output_dir
        self.output_dir = os.path.join(output_dir, f"run_{self.timestamp}")
        self.create_output_directories()

    def create_output_directories(self):
        subdirs = ["original", "skin_mask", "cleaned", "final", "results"]
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def get_input_images(self):
        """Get exactly 10 images as per requirements"""
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.input_dir, ext)))
        
        # Ensure exactly 10 images
        if len(image_files) < 10:
            raise ValueError(f"Found only {len(image_files)} images. Need exactly 10 images.")
        return sorted(image_files)[:10]

    def analyze_blob_shape(self, prop):
        """Enhanced face shape analysis for fashion photography"""
        min_row, min_col, max_row, max_col = prop.bbox
        height = max_row - min_row
        width = max_col - min_col
        
        # Minimum size requirements (adjusted for fashion photos)
        if height < 50 or width < 50:
            return False
        
        # Stricter aspect ratio for faces
        aspect_ratio = width / height if height > 0 else 0
        if not (0.9 <= aspect_ratio <= 1.4):
            return False
        
        # Stricter size constraints relative to image
        relative_size = prop.area / (prop.image.shape[0] * prop.image.shape[1])
        if relative_size < 0.02 or relative_size > 0.3:  # Face should be between 2% and 30% of image
            return False
        
        # Higher solidity threshold
        if prop.solidity < 0.85:  # Face regions should be very solid
            return False
        
        # Stricter eccentricity threshold
        if prop.eccentricity > 0.6:  # Face should be more circular
            return False
        
        # Check convexity
        if prop.extent < 0.65:  # Face should be relatively convex
            return False
        
        return True

    def skin_detection(self, image):
        """Basic but reliable skin detection using proven color ranges"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Standard HSV ranges for skin detection (from research papers)
        lower_hsv = np.array([0, 30, 60])
        upper_hsv = np.array([20, 150, 255])
        
        # Standard YCrCb ranges for skin detection
        lower_ycrcb = np.array([0, 135, 85])
        upper_ycrcb = np.array([255, 180, 135])
        
        # Create masks
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        
        # Basic cleanup with conservative parameters
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return mask

    def remove_small_components(self, binary_image, min_size=2000):
        """Simple component filtering with fixed threshold"""
        labels = measure.label(binary_image)
        props = measure.regionprops(labels)
        
        mask = np.zeros_like(binary_image)
        for prop in props:
            if prop.area >= min_size:
                mask[labels == prop.label] = 1
        
        return mask

    def process_image(self, image_path, index):
        """Simplified processing pipeline"""
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{index:03d}_{base_filename}"
        
        # Read and resize image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Keep original size but ensure maximum dimension doesn't exceed 1024
        max_dim = 1024
        height, width = image.shape[:2]
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        # Basic skin detection
        skin_mask = self.skin_detection(image)
        
        # Remove small components
        cleaned_mask = self.remove_small_components(skin_mask)
        
        # Simple morphological operations
        kernel = np.ones((5,5), np.uint8)
        final_mask = cv2.morphologyEx(cleaned_mask.astype(np.uint8), 
                                    cv2.MORPH_CLOSE, 
                                    kernel, 
                                    iterations=2)
        
        # Save all steps
        self.save_image(image, "original", f"{filename}_original.jpg")
        self.save_image(skin_mask, "skin_mask", f"{filename}_skin_mask.jpg", True)
        self.save_image(cleaned_mask, "cleaned", f"{filename}_cleaned.jpg", True)
        self.save_image(final_mask, "final", f"{filename}_final_mask.jpg", True)
        
        # Visualize the mask on the original image
        result = image.copy()
        result[final_mask == 0] = 0
        self.save_image(result, "results", f"{filename}_result.jpg")
        
        return {
            'filename': filename,
            'original_path': image_path,
            'mask_generated': True
        }


    def save_image(self, image, subfolder, filename, is_mask=False):
        """Save image to specified subfolder"""
        if image is None:
            return
            
        save_path = os.path.join(self.output_dir, subfolder, filename)
        
        if is_mask:
            # Convert binary mask to visible image
            save_image = image.astype(np.uint8) * 255
        else:
            save_image = image
            
        cv2.imwrite(save_path, save_image)
        return save_path

    def visualize_results(self, image, faces):
        """Improved visualization with better face proportions"""
        result = image.copy()
        for face in faces:
            center = face['center']
            axes = face['axes']
            angle = face['angle']
            
            # Draw ellipse
            cv2.ellipse(result, 
                        center,
                        axes,
                        angle,
                        0, 360,
                        (0, 255, 0),
                        2)
            
            # Draw center point
            cv2.circle(result, center, 3, (0, 0, 255), -1)
        
        return result


    def process_multiple_images(self, image_paths):
        """Process multiple images and save results"""
        results = []
        for idx, path in enumerate(image_paths):
            print(f"Processing image {idx + 1}/{len(image_paths)}: {path}")
            result = self.process_image(path, idx)
            if result:
                results.append(result)

        # Save summary report
        self.save_summary_report(results)
        return results

    def save_summary_report(self, results):
        """Save a summary report of the processing results"""
        report_path = os.path.join(self.output_dir, "processing_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"Processing Report - {self.timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            for result in results:
                f.write(f"Image: {result['filename']}\n")
                f.write(f"Faces detected: {result['faces_detected']}\n")
                f.write("Face details:\n")
                for i, face in enumerate(result['faces']):
                    f.write(f"  Face {i+1}:\n")
                    f.write(f"    Center: {face['center']}\n")
                    f.write(f"    Axes: {face['axes']}\n")
                    f.write(f"    Angle: {face['angle']:.2f} degrees\n")
                f.write("\n")

# Example usage
def main():
    try:
        processor = SkinSegmentation(
            input_dir="input_images",
            output_dir="skin_segmentation_output"
        )
        
        image_paths = processor.get_input_images()
        results = processor.process_multiple_images(image_paths)
        
        print(f"\nProcessing complete!")
        print(f"Results saved in: {processor.output_dir}")
        print(f"Processed {len(results)} images successfully")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")

if __name__ == "__main__":
    main()