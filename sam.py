import cv2
import numpy as np

# Load the original image and mask
image = cv2.imread('data/ICPR2012_Cells_Classification_Contest/test/010.png')
mask = cv2.imread('data/ICPR2012_Cells_Classification_Contest/test/010_mask.png', cv2.IMREAD_GRAYSCALE)

# Apply the mask to the image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Display the result
#cv2.imshow("Masked Image", masked_image)
#cv2.imshow("normal",image)
#cv2.imshow("emty_mask",mask)
cv2.imwrite("Masked Image010.jpg", masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
