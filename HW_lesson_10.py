import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def generate_image(color, shape):
    img = np.zeros((200, 200, 3), np.uint8)

    if shape == "circle":
        cv2.circle(img, (100, 100), 50, color, -1)
    elif shape == "square":
        cv2.rectangle(img, (50, 50), (150, 150), color, -1)
    elif shape == "triangle":
        points = np.array([[100, 40], [40, 160], [160, 100]])
        cv2.drawContours(img, [points], 0, color, -1)
    return img

X = []
y = []

colors = {
    'red':    (0, 0, 255),
    'green':  (0, 255, 0),
    'blue':   (255, 0, 0),
    'yellow': (0, 255, 255),
    'white':  (255, 255, 255),
    'black':  (0, 0, 0),
    'pink':   (255, 0, 255),
    'violet': (128, 0, 128),
}

shapes = ['circle', 'square', 'triangle']

for color_name, bgr in colors.items():
    for shape in shapes:
        for _ in range(10):
            img = generate_image(bgr, shape)
            mean_color = cv2.mean(img)[:3]
            features = [mean_color[0], mean_color[1], mean_color[2]]

            X.append(features)
            y.append(f'{color_name}_{shape}')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f'Точність моделі: {round(accuracy * 100, 2)}%')

SMOOTH_FRAMES = 5
color_buffer = []

test_img = generate_image((255, 13, 240), 'triangle')

for _ in range(SMOOTH_FRAMES):
    mean_color = cv2.mean(test_img)[:3]
    color_buffer.append(mean_color)
    color_buffer = color_buffer[-SMOOTH_FRAMES:]

smoothed_features = np.mean(color_buffer, axis=0)

prediction  = model.predict([smoothed_features])
proba       = model.predict_proba([smoothed_features])[0]
confidence  = max(proba) * 100

print(f'Передбачення:  {prediction[0]}')
print(f'Впевненість:   {round(confidence, 2)}%')

cv2.imshow("test", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()