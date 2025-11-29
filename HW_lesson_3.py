import cv2
from numpy.ma.core import filled

img = cv2.imread('images/selfie.jpg')
img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))
cv2.rectangle(img, (120, 180), (360, 520), (0, 255, 0), 2)
cv2.rectangle(img, (120, 520), (260, 550), (0, 255, 0), -1)

cv2.putText(img, "Ivanko Nazarij", (125, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

# overlay = img.copy()
# cv2.rectangle(overlay, (120, 180), (360, 520), (0, 255, 0), -1)
# img = cv2.addWeighted(overlay, 0.2, img, 1 - 0.2, 0)

cv2.imshow("selfie", img)
cv2.waitKey(0)
cv2.destroyAllWindows()