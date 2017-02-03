import cv2

image = cv2.imread("a.bmp")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)  # threshold
thresh1 = cv2.medianBlur(thresh, 3)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
erosion = cv2.erode(thresh1, kernel, iterations=1)
dilated = cv2.dilate(thresh1, kernel, iterations=2)  # dilate"""
close = cv2.erode(dilated, kernel, iterations=4)
open = cv2.dilate(erosion, kernel, iterations=3)
# opening = cv2.morphologyEx(thresh, kernel, MORPH_OPEN, iterations=3)

"""opening = cv2.morphologyEx(thresh,kernel,iterations = 7)"""
_, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

d = 0
size = (28, 28)
# for each contour found, draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # discard areas that are too large
    if h > 100 and w > 100:
        continue

    # discard areas that are too small

    if h < 15 or w < 15:
        continue

    # test = image[y - 30:y + h, x:x + w]
    # test1 = cv2.resize(test, size)
    # filename = "%c.bmp" % d
    # cv2.imwrite("aa", test1)
    # d = d + 1
    # draw rectangle around contour on original image2
    cv2.putText(image,'a',(x+w,y+h), 4, 1,(0,0,0),2)
    cv2.rectangle(image, (x, y - 15), (x + w, y + h ), (0, 0, 255), 2)
    print(x, y, w, h)

# write original image with added contours to disk
cv2.imwrite("contoured_a.jpg", image)
# cv2.imwrite("Binary Image.jpg", thresh)
# cv2.imwrite("medianFilter.jpg", thresh1)
# cv2.imwrite("erode_a.jpg", erosion)
# cv2.imwrite("dilate_a.jpg", dilated)
# cv2.imwrite("open_a.jpg", open)
# cv2.imwrite("close_a.jpg", close)
