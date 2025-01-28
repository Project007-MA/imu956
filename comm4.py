import matplotlib.pyplot as plt
import cv2

def show_predictions_with_detections(image_path, detections):
    """
    Displays an image with detection boxes, classification labels, and confidence scores.

    Parameters:
        image_path (str): Path to the image file.
        detections (list of dict): List of detection results. Each detection should be a dictionary with:
            - 'box' (tuple): Bounding box as (x_min, y_min, x_max, y_max).
            - 'label' (str): Classification label.
            - 'confidence' (float): Confidence score.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Plot the image
    plt.figure(figsize=(12, 7))
    plt.imshow(image)
    ax = plt.gca()

    # Draw detections
    for detection in detections:
        box = detection['box']
        label = detection['label']
        confidence = detection['confidence']
        
        # Draw bounding box
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                              linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label and confidence score
        text = f"{label}: {confidence:.2f}"
        ax.text(x_min, y_min - 10, text, color='red', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    plt.axis("off")
    plt.show()


def show_images_with_predictions(original_image_path, mask_image_path, result_image_path, detections):
    """
    Displays original, mask, and result images in subplots, 
    including detection boxes, classification labels, and confidence scores on the result image.

    Parameters:
        original_image_path (str): Path to the original image.
        mask_image_path (str): Path to the mask image.
        result_image_path (str): Path to the result image.
        detections (list of dict): List of detection results for the result image.
    """
    # Load images
    original_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(mask_image_path)
    result_image = cv2.imread(result_image_path)

    if original_image is None or mask_image is None or result_image is None:
        raise FileNotFoundError("One or more image paths are invalid.")
    
    # Convert BGR to RGB
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Plot images in subplots
    plt.figure(figsize=(15, 5))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    # plt.title("Original Image")
    plt.axis("off")
    
    # Mask Image
    plt.subplot(1, 3, 2)
    plt.imshow(mask_image)
    # plt.title("Mask Image")
    plt.axis("off")
    
    # Result Image with Detections
    plt.subplot(1, 3, 3)
    plt.imshow(result_image)
    ax = plt.gca()

    # Draw detections on the result image
    for detection in detections:
        box = detection['box']
        label = detection['label']
        confidence = detection['confidence']
        
        # Draw bounding box
        x_min, y_min, x_max, y_max = box
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                              linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label and confidence score
        text = f"{label}: {confidence:.2f}"
        ax.text(x_min, y_min - 10, text, color='red', fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.7))
    
    # plt.title("Result Image with Detections")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# Example usage
original_image_path = r"E:\Projects\Santhiya-Paper\santhiya-rework\immune_sys\Dataset\1\603.png"
mask_image_path = r"E:\Projects\Santhiya-Paper\santhiya-rework\immune_sys\Dataset1\test\207_mask.png"
result_image_path = r"E:\Projects\Santhiya-Paper\santhiya-rework\immune_sys\Dataset1\test\207.png"

# Example detections
detections = [
    {'box': (50,10, 170, 100), 'label': 'centromere;', 'confidence': 0.92}

]

show_images_with_predictions(original_image_path, mask_image_path, result_image_path, detections)
