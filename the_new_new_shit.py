import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.measure import regionprops, label
import logging
from typing import Tuple, List
import sys
from datetime import datetime

# Logging setup remains the same
log_folder = Path('logs')
log_folder.mkdir(exist_ok=True)
log_file = log_folder / f'skin_segmentation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file)
    ]
)
def segment_skin(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment skin using YCrCb color space with better thresholds
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Split channels
        y = ycrcb[:,:,0]   # Luminance
        cr = ycrcb[:,:,1]  # Red difference
        cb = ycrcb[:,:,2]  # Blue difference
        
        # Define skin mask with stricter bounds for each channel
        skin_mask = np.logical_and.reduce([
            y >= 50,    # Minimum brightness
            y <= 250,   # Maximum brightness
            cr >= 137,  # Minimum red difference
            cr <= 177,  # Maximum red difference
            cb >= 77,   # Minimum blue difference
            cb <= 127   # Maximum blue difference
        ])
        
        # Convert to uint8
        skin_mask = skin_mask.astype(np.uint8) * 255
        
        # Apply Gaussian blur to reduce noise
        skin_mask = cv2.GaussianBlur(skin_mask, (3,3), 0)
        
        # Apply binary threshold to get clean mask
        _, skin_mask = cv2.threshold(skin_mask, 127, 255, cv2.THRESH_BINARY)
        
        return skin_mask, image
    
    except Exception as e:
        logging.error(f"Error in segment_skin: {str(e)}")
        raise

def remove_small_components(binary_image: np.ndarray, min_size: int = 2000) -> np.ndarray:
    """
    Remove small connected components and fill holes
    """
    try:
        # Convert to binary
        binary = binary_image > 127
        
        # Label connected components
        labeled_image = label(binary)
        
        # Remove small components
        for region in regionprops(labeled_image):
            if region.area < min_size:
                binary[labeled_image == region.label] = 0
        
        # Fill holes in the remaining components
        from scipy import ndimage
        binary = ndimage.binary_fill_holes(binary)
        
        return binary.astype(np.uint8) * 255
    
    except Exception as e:
        logging.error(f"Error in remove_small_components: {str(e)}")
        raise

def morphological_operations(binary_image: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations with better sequence
    """
    try:
        # Ensure binary
        binary = binary_image > 127
        
        # Create structural elements
        small_element = disk(2)
        medium_element = disk(3)
        large_element = disk(5)
        
        # First do opening to remove small noise
        cleaned = binary_erosion(binary, small_element)
        cleaned = binary_dilation(cleaned, small_element)
        
        # Then do closing to fill holes
        cleaned = binary_dilation(cleaned, medium_element)
        cleaned = binary_erosion(cleaned, medium_element)
        
        # Final closing with larger element to merge nearby components
        cleaned = binary_dilation(cleaned, large_element)
        cleaned = binary_erosion(cleaned, large_element)
        
        return cleaned.astype(np.uint8) * 255
    
    except Exception as e:
        logging.error(f"Error in morphological_operations: {str(e)}")
        raise


def detect_faces(binary_image: np.ndarray, 
                original_image: np.ndarray,
                min_face_size: int = 1000) -> Tuple[np.ndarray, List]:
    """
    Detect faces with support for different poses and profiles
    """
    try:
        binary = binary_image > 127
        labeled_image = label(binary)
        regions = regionprops(labeled_image)
        
        faces = []
        result_image = original_image.copy()
        
        height, width = binary_image.shape
        image_area = height * width
        
        # Sort regions by area
        regions = sorted(regions, key=lambda r: r.area, reverse=True)
        
        for region in regions:
            bbox = region.bbox  # (min_row, min_col, max_row, max_col)
            bbox_height = bbox[2] - bbox[0]
            bbox_width = bbox[3] - bbox[1]
            aspect_ratio = bbox_height / bbox_width
            relative_area = region.area / image_area

            # Calculate position in image
            center_y = (bbox[0] + bbox[2]) / 2 / height
            
            # Different criteria for front and side profiles
            is_front_face = (
                region.area > min_face_size and
                0.02 < relative_area < 0.9 and
                0.8 < aspect_ratio < 1.8 and
                region.eccentricity < 0.8 and
                region.extent > 0.3 and
                region.solidity > 0.6
            )
            
            is_side_face = (
                region.area > min_face_size and
                0.02 < relative_area < 0.9 and
                0.6 < aspect_ratio < 2.2 and    # Allow more variation for side profiles
                region.eccentricity < 0.85 and  # Allow more elongated shapes
                region.extent > 0.25 and        # Side profiles might be less filled
                region.solidity > 0.5 and       # More tolerant for side views
                center_y < 0.8                  # Face should be in upper 80% of image
            )
            
            if is_front_face or is_side_face:
                faces.append(region)
                logging.info(f"Face detected - Area: {region.area}, Aspect ratio: {aspect_ratio:.2f}, "
                           f"Eccentricity: {region.eccentricity:.2f}")
                
                try:
                    y0, x0 = region.centroid
                    orientation = region.orientation if region.orientation is not None else 0
                    
                    # Adjust ellipse size based on face type
                    if is_side_face:
                        major_axis = bbox_height / 1.8  # Reduce height more for profile
                        minor_axis = bbox_width / 1.6   # Make narrower for profile
                    else:
                        major_axis = bbox_height / 2
                        minor_axis = bbox_width / 1.8
                    
                    # Additional adjustments based on aspect ratio
                    if aspect_ratio > 1.5:
                        major_axis *= 0.85
                    if aspect_ratio < 1:
                        minor_axis *= 0.85
                    
                    center = (int(x0), int(y0))
                    axes = (int(minor_axis), int(major_axis))
                    angle = np.degrees(orientation) + 90
                    
                    # Draw ellipse
                    cv2.ellipse(result_image, center, axes, angle, 0, 360, (0, 255, 0), 2)
                    
                    # Optional: Draw bounding box for debugging
                    cv2.rectangle(result_image, 
                                (bbox[1], bbox[0]), 
                                (bbox[3], bbox[2]), 
                                (0, 0, 255), 2)
                    
                    break
                    
                except Exception as e:
                    logging.warning(f"Failed to draw ellipse: {str(e)}")
                    continue
        
        if not faces:
            logging.warning("No faces detected in image")
            for region in regions[:3]:
                logging.debug(f"Region stats: area={region.area}, "
                            f"aspect_ratio={bbox_height/bbox_width:.2f}, "
                            f"eccentricity={region.eccentricity:.2f}")
        
        return result_image, faces
    
    except Exception as e:
        logging.error(f"Error in detect_faces: {str(e)}")
        raise
    
    

def save_visualization(original_image: np.ndarray,
                      skin_mask: np.ndarray,
                      processed_mask: np.ndarray,
                      result_image: np.ndarray,
                      output_path: Path,
                      filename: str) -> None:
    """
    Save visualization of the processing steps
    """
    try:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        
        plt.subplot(142)
        plt.imshow(skin_mask, cmap='gray')
        plt.title('Skin Mask')
        
        plt.subplot(143)
        plt.imshow(processed_mask, cmap='gray')
        plt.title('Processed Mask')
        
        plt.subplot(144)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Faces')
        
        plt.tight_layout()
        plt.savefig(str(output_path / f'analysis_{filename}.png'))
        plt.close()
        
    except Exception as e:
        logging.error(f"Error saving visualization: {str(e)}")
        raise

def process_images(input_folder: str) -> None:
    """
    Process all images in the input folder
    """
    try:
        input_path = Path(input_folder)
        if not input_path.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")
        
        # Create output folder for results
        output_path = input_path.parent / 'output'
        output_path.mkdir(exist_ok=True)
        
        # Get list of image files
        image_files = list(input_path.glob('*.jpg')) + list(input_path.glob('*.jpeg')) + list(input_path.glob('*.png'))
        
        if not image_files:
            logging.warning(f"No image files found in {input_folder}")
            return
        
        logging.info(f"Found {len(image_files)} images to process")
        
        # Process each image
        for i, image_file in enumerate(image_files, 1):
            try:
                logging.info(f"Processing image {i}/{len(image_files)}: {image_file.name}")
                
                # Segment skin
                skin_mask, original_image = segment_skin(image_file)
                
                # Remove small components
                cleaned_mask = remove_small_components(skin_mask)
                
                # Apply morphological operations
                processed_mask = morphological_operations(cleaned_mask)
                
                # Detect faces
                result_image, faces = detect_faces(processed_mask, original_image)
                
                # Save results
                output_file = output_path / f'processed_{image_file.name}'
                cv2.imwrite(str(output_file), result_image)
                
                # Save visualization
                save_visualization(
                    original_image, skin_mask, processed_mask, 
                    result_image, output_path, image_file.stem
                )
                
                logging.info(f"Successfully processed {image_file.name}")
                
            except Exception as e:
                logging.error(f"Failed to process {image_file.name}: {str(e)}")
                continue
                
    except Exception as e:
        logging.error(f"Error in process_images: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        input_folder = "input_images"
        logging.info("Starting skin segmentation process...")
        process_images(input_folder)
        logging.info("Processing completed successfully")
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        sys.exit(1)