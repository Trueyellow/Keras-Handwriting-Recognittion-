import cv2
import numpy as np

image = cv2.imread('Rutgers.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
cv2.imwrite('binary.jpg', binary)
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
eroded = cv2.erode(binary, kernel, iterations=5)
cv2.imwrite('eroded.jpg', eroded)
medianBlur = cv2.medianBlur(eroded, 5)
cv2.imwrite('medianBlur.jpg', medianBlur)

hist = np.zeros((gray.shape[1], 1))
for x in range(gray.shape[1]):
    counter = 0
    for i in range(gray.shape[0]):
        if medianBlur[i][x] == 0:
            counter += 1
    hist[x] = counter

edge_left = 0
edge_right = 0

for i in range(hist.shape[0]):
    if (hist[i] > 10):
        edge_left = i
        break

for i in reversed(range(gray.shape[1])):
    if (hist[i] > 10):
        edge_right = i
        break

window = 30
if (edge_left - window > 0 and edge_right < image.shape[1]):
    new_image = image[0:image.shape[0], edge_left - window:edge_right + window]
elif edge_left - window < 0 and edge_right < image.shape[1]:
    new_image = image[0:image.shape[0], 0:edge_right + window]
elif edge_left - window < 0 and edge_right > image.shape[1]:
    new_image = image[0:image.shape[0], 0:image.shape[1]]
elif edge_left - window > 0 and edge_right > image.shape[1]:
    new_image = image[0:image.shape[0], edge_left - window:image.shape[1]]

cv2.imwrite('orginal_cut.jpg', new_image)

step = 20
window_size=40

i=0
d=1
while i<new_image.shape[1]-window_size:
    subimage=new_image[0:new_image.shape[0],i:i+window_size]
    subimage=cv2.resize(subimage,(128,128))
    cv2.imwrite('{}.jpg'.format(d),subimage)
    d+=1
    i+=step
