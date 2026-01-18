import cv2
import numpy as np

img = cv2.imread('images/holodos.jpg')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5, 5), 1)

img = cv2.equalizeHist(img)

img_edges = cv2.Canny(img, 50, 150)

kernel = np.ones((2, 2), np.uint8)
contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #

magnit_count = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 400:
        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        if aspect_ratio < 5:
            magnit_count += 1

        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)

        magnit_count += 1

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_y = y - 10 if y - 10 > 20 else y + 20

        text = f'{magnit_count}'
        cv2.putText(img_copy, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f'Кількість знайдених магнітів: {magnit_count}')

cv2.imwrite('images/holodos1.jpg', img_copy)

cv2.imshow('image', img)
cv2.imshow('image2', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()