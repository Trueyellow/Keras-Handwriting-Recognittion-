def cut(picture_name):
    image = cv2.imread(picture_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    hist_x = np.zeros((gray.shape[0], 1))
    hist_y = np.zeros((gray.shape[1], 1))

    for i in range(binary.shape[0]):
        counter = 0
        for j in range(binary.shape[1]):
            if binary[i][j] == 0:
                counter += 1
        hist_x[i] = counter

    for m in range(binary.shape[1]):
        counter = 0
        for n in range(binary.shape[0]):
            if binary[n][m] == 0:
                counter += 1
        hist_y[m] = counter

    left = right = up = down = 0

    for a in range(binary.shape[0]):
        if hist_x[a] != 0:
            up = a
            break

    for b in reversed(range(binary.shape[0])):
        if hist_x[b] != 0:
            down = b
            break

    for c in range(binary.shape[1]):
        if hist_y[c] != 0:
            left = c
            break

    for d in reversed(range(binary.shape[1])):
        if hist_y[d] != 0:
            right = d
            break

    print(up,down,left,right)
    if left - 3 >= 0 and right + 3 < binary.shape[1] and up - 3 >= 0 and down + 3 < binary.shape[0]:
        new_image = image[up - 3:down + 3, left - 3:right + 3]
    else:
        new_image = image[up:down, left:right]

    cv2.imwrite('{}.jpg'.format(picture_name), new_image)
