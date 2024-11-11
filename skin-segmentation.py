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

    def get_texture_mask(self, image, window_size=15):
        """Generate texture mask using a simpler and more robust approach"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate local variance using a more efficient method
        blur = cv2.GaussianBlur(gray, (window_size, window_size), 0)
        squared_diff = cv2.absdiff(gray, blur)
        variance = cv2.GaussianBlur(squared_diff, (window_size, window_size), 0)
        
        # Create mask based on variance
        texture_mask = np.zeros_like(gray)
        texture_mask[(variance > 5) & (variance < 50)] = 255
        
        return texture_mask

    def skin_detection(self, image):
        """Enhanced skin detection with lighting compensation and texture analysis"""
        # Apply lighting compensation
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Perform CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab = cv2.merge([l, a, b])
        balanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(balanced_image, cv2.COLOR_BGR2YCrCb)
        
        # Strict HSV ranges for skin
        lower_hsv = np.array([0, 30, 60])
        upper_hsv = np.array([20, 180, 250])
        
        # Strict YCrCb ranges for skin
        lower_ycrcb = np.array([80, 135, 85])
        upper_ycrcb = np.array([235, 180, 135])
        
        # Create masks
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Get texture mask
        texture_mask = self.get_texture_mask(balanced_image)
        
        # Combine all masks
        mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
        mask = cv2.bitwise_and(mask, texture_mask)
        
        # Clean up the mask
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return mask

    def analyze_blob_shape(self, prop):
        """Enhanced face shape analysis"""
        min_row, min_col, max_row, max_col = prop.bbox
        height = max_row - min_row
        width = max_col - min_col
        
        # Size requirements
        if height < 60 or width < 60:
            return False
        
        # Aspect ratio
        aspect_ratio = width / height if height > 0 else 0
        if not (0.95 <= aspect_ratio <= 1.3):
            return False
        
        # Size relative to image
        relative_size = prop.area / (prop.image.shape[0] * prop.image.shape[1])
        if relative_size < 0.03 or relative_size > 0.25:
            return False
        
        # Shape checks
        if prop.solidity < 0.9:
            return False
        
        if prop.eccentricity > 0.5:
            return False
        
        if prop.extent < 0.7:
            return False
        
        return True

    def remove_small_components(self, binary_image, min_size=1000):
        """Remove small components and apply additional filtering"""
        labels = measure.label(binary_image)
        props = measure.regionprops(labels)
        
        mask = np.zeros_like(binary_image)
        for prop in props:
            if (prop.area >= min_size and
                prop.extent >= 0.4):
                mask[labels == prop.label] = 1
                
        return mask
    
    def process_image(self, image_path, index):
        """Enhanced image processing with better visualization"""
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{index:03d}_{base_filename}"
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Standardize image size while maintaining quality
        target_width = 1024  # Increased for better detail
        aspect_ratio = image.shape[1] / image.shape[0]
        target_height = int(target_width / aspect_ratio)
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply enhanced skin detection
        skin_mask = self.skin_detection(image)
        
        # Remove noise with larger minimum size
        cleaned_mask = self.remove_small_components(skin_mask, min_size=1000)
        
        # More aggressive morphological operations
        kernel = np.ones((7,7), np.uint8)  # Larger kernel
        dilated = binary_dilation(cleaned_mask, kernel)
        final_mask = binary_erosion(dilated, kernel)
        
        # Enhanced face detection
        labels = measure.label(final_mask)
        props = measure.regionprops(labels)
        
        faces = []
        for prop in props:
            if self.analyze_blob_shape(prop):
                y0, x0 = prop.centroid
                # Adjusted face dimensions
                face_width = prop.major_axis_length * 0.7
                face_height = face_width * 1.3
                
                faces.append({
                    'center': (int(x0), int(y0)),
                    'axes': (int(face_width/2), int(face_height/2)),
                    'angle': np.degrees(prop.orientation)
                })
        
        # Enhanced visualization
        result_image = self.visualize_results(image, faces)
        
        # Save intermediate results
        self.save_image(image, "original", f"{filename}_original.jpg")
        self.save_image(skin_mask, "skin_mask", f"{filename}_skin_mask.jpg", True)
        self.save_image(cleaned_mask, "cleaned", f"{filename}_cleaned.jpg", True)
        self.save_image(final_mask, "final", f"{filename}_final_mask.jpg", True)
        self.save_image(result_image, "results", f"{filename}_result.jpg")
        
        return {
            'filename': filename,
            'original_path': image_path,
            'faces_detected': len(faces),
            'faces': faces
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