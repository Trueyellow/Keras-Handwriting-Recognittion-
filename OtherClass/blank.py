import cv2

image = cv2.imread("blank.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale

print(gray.shape)

d = 1
slide1 = 10
slide2 = 10

i=0
j=0
while i < gray.shape[0] - 128:
    while j < gray.shape[1]-128:
        new_image = gray[i:i+128,j:j+128]
        filename = "%d.jpg" % d
        cv2.imwrite(filename, new_image)
        d = d + 1
        j=j+slide1
        print(j)
    i = i + slide2
    j=0
