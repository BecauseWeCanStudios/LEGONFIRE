import re
import cv2
import time
import threading
import numpy as np
from randomcolor import RandomColor
from skimage.measure import find_contours

randomcolor = RandomColor()
color_regex = re.compile(r'(\d+)')

def random_colors(count):
	return list(map(lambda x: list(map(int, color_regex.findall(x))), 
		randomcolor.generate(count=count, format_='rgb')))

def apply_mask(image, mask, color=None, alpha=0.5):
	if not color:
		color = random_colors(1)[0]
	for i in range(3):
		image[:, :, i] = np.where(mask, image[:, :, i] * (1 - alpha) + alpha * color[i], image[:, :, i])
	return image

def mask_image(image, boxes, masks, class_ids, class_names, scores, colors=None):
    n = boxes.shape[0]
    if not colors:
        colors = random_colors(len(class_names))
    
    masked_image = np.array(image)
    for i in range(n):
        class_id = class_ids[i]
        color = colors[class_id]
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, thickness=2)

        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        cv2.putText(
            masked_image, 
            '{} {:.3f}'.format(class_names[class_id], scores[i]), 
            (x1 + 5, y1 + 16), 
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, 
            color
        )

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        for v in find_contours(padded_mask, 0.5):
            cv2.polylines(masked_image, (np.fliplr(v) - 1).reshape((-1, 1, 2)).astype(np.int32), True, color)
    return masked_image

class Camera:

    def __init__(self, colors, class_names, size):
        self.colors = colors
        self.class_names = class_names
        self.cap = cv2.VideoCapture(0)
        self.lock = threading.Lock()
        self.image = None
        self.mask = None
        self.size = size
        thread = threading.Thread(target=self.__async_show__)
        thread.start()
    
    def __async_show__(self):
        try:
            while True:
                self.lock.acquire()
                _, image = self.cap.read()
                image = cv2.resize(image, self.size)
                self.image = image
                mask = self.mask
                colors = self.colors
                class_names = self.class_names
                self.lock.release()
                if mask:
                    image = mask_image(image, mask['rois'], mask['masks'], 
                                mask['class_ids'], class_names, mask['scores'], colors)
                cv2.imshow('Masked image', cv2.resize(image, None, fx=3, fy=3))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(e)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    
    def get_image(self):
        self.lock.acquire()
        image = self.image
        self.lock.release()
        return image

    def set_mask(self, mask):
        self.lock.acquire()
        self.mask = mask
        self.lock.release()

def start(model, class_names, size):
    colors = random_colors(len(class_names))
    camera = Camera(colors, class_names, size)
    while True:
        image = camera.get_image()
        if image is None:
            continue
        t = time.time()
        result = model.detect(image, 0)
        t = time.time() - t
        print(t)
        camera.set_mask(result)
