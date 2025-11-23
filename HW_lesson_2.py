import cv2
import numpy as np

image = cv2.imread('images/Absolute.jpg')
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image = cv2.Canny(image, 100, 100)

kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations = 1)
image = cv2.erode(image, kernel, iterations = 1)

cv2.imwrite('images/Absolute1.jpg', image)

cv2.imshow("Absolute", image)

image1 = cv2.imread('images/email.jpg')
image1 = cv2.resize(image1, (image1.shape[1] // 2, image1.shape[0] // 2))

image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image1 = cv2.Canny(image1, 200, 200)

cv2.imwrite('images/email1.jpg', image1)

cv2.imshow("email", image1)


cv2.waitKey(0)
cv2.destroyAllWindows()