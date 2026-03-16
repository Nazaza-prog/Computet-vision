import cv2
import os

net = cv2.dnn.readNetFromCaffe('data/MobileNet/mobilenet_deploy.prototxt', 'data/MobileNet/mobilenet.caffemodel')
classes = []

with open('data/MobileNet/synset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(" ", 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)


image_folder = 'images/MobileNet'
extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(extensions)]

class_counts = {}

for filename in image_files:
    path = os.path.join(image_folder, filename)
    image = cv2.imread(path)

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))

    net.setInput(blob)
    
    preds = net.forward()

    index = preds[0].argmax()
    label = classes[index] if index < len(classes) else "unknown"
    conf = float(preds[0][index].item()) * 100

    print(f'Файл: {filename}')
    print(f'  Клас: {label}')
    print(f'  Ймовірність: {round(conf, 2)}%')
    print()
    print()

    class_counts[label] = class_counts.get(label, 0) + 1

    text = label + ": " + str(int(conf)) + "%"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow(filename, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print(f'{"Клас":<25} {"К-сть":>5}')
print()
for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f'{label:<25} {count:>5}')