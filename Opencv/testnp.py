import cv2
image = cv2.imread("n2.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # grayscale
_,thresh = cv2.threshold(gray,150,255,cv2.THRESH_BINARY_INV) # threshold
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
erosion = cv2.erode(thresh,kernel,iterations = 0)
dilated = cv2.dilate(erosion,kernel,iterations = 7) # dilate"""
"""opening = cv2.morphologyEx(thresh,kernel,iterations = 7)"""
_,contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours

# for each contour found, draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x,y,w,h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h>50 and w>70:
        continue

    # discard areas that are too small

    if h<20 or w<20:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image,(x,y),(x+w,y+h),(150,85,255),2)

# write original image with added contours to disk  
cv2.imwrite("contoured.jpg", image)
cv2.imwrite("erode.jpg",erosion)
cv2.imwrite("dilate.jpg",dilated)