import cv2
import numpy as np

img = cv2.imread('images/candies.jpg')
img_copy = img.copy()

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_orange = np.array([8, 120, 120])
upper_orange = np.array([25, 255, 255])

lower_purple = np.array([125, 60, 60])
upper_purple = np.array([155, 255, 255])

lower_green = np.array([35, 80, 80])
upper_green = np.array([90, 255, 255])

colors = [
    (lower_orange, upper_orange, (0, 165, 255), "orange"),
    (lower_purple, upper_purple, (180, 0, 180), "purple"),
    (lower_green,  upper_green,  (0, 255, 0),   "green"),
]

for lower, upper, color, name in colors:
    mask = cv2.inRange(img_hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.drawContours(img_copy, [cnt], -1, color, 3)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(name, font, font_scale, thickness)
            text_x = x + (w - text_w) // 2
            text_y = y + (h + text_h) // 2
            cv2.putText(img_copy, name, (text_x, text_y), font, font_scale, color, thickness)

cv2.imwrite('images/Konfetki.jpg', img_copy)

cv2.imshow('Candies', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
