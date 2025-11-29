import cv2

# img = cv2.imread('images/images.jpeg')
# cv2.waitKey(0)
# img = cv2.resize(img, (500, 300))
#
# cv2.destroyAllWindows()

cap = cv2.VideoCapture('video/video1.mov')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (800, 300))

    cv2.imshow('video', resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
