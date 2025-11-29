import cv2
import numpy as np
from numpy.ma.core import filled

img = np.zeros((512, 512, 3), np.uint8)
# чоряний фон

# img[100:150,200:280] = 109, 248, 123 #конкретна частина по координатам
# img[:] = 109, 248, 123 #вся матриця

cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
# квадрат

cv2.line(img, (400, 100), (300, 150), (0, 0, 225), 2)
# лінія

print(img.shape)

cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2),  (255, 225, 0), 2) # лінія навпіл екрана

cv2.circle(img, (200, 200), 40, (255, 255, 0), -1)
# коло

cv2.putText(img, "name", (100, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1) # текст


cv2.imshow("primitiuv", img)
cv2.waitKey(0)
cv2.destroyAllWindows()