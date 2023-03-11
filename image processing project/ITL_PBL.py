# Project Title: - " APPLYING CARTOON EFFECT FILTER ON IMAGES USING PYTHON "

# Developers:-
# 47_Avadhut Patil



# Step 1: - IMPORTING REQUIRED MODULES

import numpy as np                                  # Handling all arithmetic, trignometric, complex number function.
import cv2                                          # Processing the images
from tkinter.filedialog import *                    # Provide platforms to open the input images & save it
from matplotlib import pyplot as plt                # Plotting the output images

# Step 2: - TAKING INPUT IMAGE FROM USER & DISPLAYING IT

photo = askopenfilename()                           # For taking permission to open file
img = cv2.imread (photo)                            # To read the input image
               
plt.imshow(img)                                     # For displaying the output image
plt.show() 

# Step 3: - CONVERTING BYDEFAULT BGR IMAGE TO RGB IMAGE

image = cv2.cvtColor (img, cv2.COLOR_BGR2RGB)       # For converting BGR image to RGB image.
   
plt.figure(figsize= (10,10))                        # For plotting size of image 
plt.imshow(image)                                   # For displaying the RGB output image                            
plt.show()

# Step 4: - APPLYING BILATERAL FILTER

# 1. REDUCING THE PIXELS OF IMAGE

img_small = cv2.pyrDown(image)                      # Pyramid down reduces pixel                

# 2. APPLYING BILATERAL FILTER ON IMAGE

num_iter = 5                                        # Count for applying Bilateral filter                          
for _ in range(num_iter):
    img_small = cv2.bilateralFilter(img_small, d=9, sigmaColor=9, sigmaSpace=7)

# 3. INCREASING PIXELS OF IMAGE

img_rgb = cv2.pyrUp(img_small)                      # Pyramid up to increase the pixel of image        

# Displaying the output image
plt.imshow(img_rgb)
plt.show()

# Step 5: - CONVERTING IMAGE FROM RGB TO GRAY FOR IDENTIFYING EDGES BY APPLYING ADAPTIVE THRESHOLD

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)  # To convert the RGB image to GRAY image
img_blur = cv2.medianBlur(img_gray, 3)                # Blur the image for smoothening the output   
img_edge = cv2.adaptiveThreshold (img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)    # To detect the edges 
                                                 
plt.imshow(img_edge)                                   # Displaying the output image                            
plt.show()

# Step 6: - CONVERTING IMAGE FROM GRAY TO RGB  AND DISPLAYING THE EDGES

img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)  # To convert the GRAY image to RGB
plt.imshow(img_edge)                                   # Displaying the edges of the image
plt.show()

# Step 7: - COLOUR QUANTIZATION

def color_quantization (img, k):                       # Function definition            

# 1. Transform the image

  data = np.float32(img).reshape((-1, 3))              # Reshaping image from larger value set to smaller value set

# 2. Determine criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.01)

# 3. Implementing K-Means (Clustering)

  ret, label, center = cv2.kmeans (data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # Implement Quantizer Using K-Means

  center = np.uint8(center)                             # Converting datatype uint8 to float
  result = center[label.flatten()]                      # Flattening the image output
  result = result.reshape(img.shape)                    # Reshaping image top original image   
  return result

 
img_1 = color_quantization(img, 8 )                     # Calling the function
           
plt.imshow(img_1)                                       # Plotting the output image 
plt.show()

# Step 8: - CONVERTING COLOUR QUANTIZED BGR IMAGE TO RGB IMAGE.
   
img_2 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)         # Colour converter (BGR2RGB)

plt.figure(figsize= (10,10))                           # Plotting the size of output image                            
plt.imshow(img_2)                                      # Displayng output image                       
plt.show()

# Step 9: - APPLYING MEDIAN BLUR ON THE COLOUR QUANTIZED IMAGE.

blurred = cv2.medianBlur(img_2, 3)                     # Blur the image for smoothening the output
           
plt.imshow(blurred)                                    # Displaying blurred output image 
plt.show()

# Step 10: - COMBINING THE OUTPUTS USING THE BITWISE_AND OPERATOR.

array_1 = cv2.bitwise_and(blurred, img_edge)           # Combining blurred and edged output   

plt.figure(figsize= (10,10))                           # Plotting the size of output image                  
plt.imshow(array_1)                                    # Displaying final output image
plt.show()


