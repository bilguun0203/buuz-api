import onnxruntime as ort
import cv2
import numpy as np


class Predictor:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, overlap_threshold: float = 0.5):
        self.session = ort.InferenceSession(model_path)
        self.confidence_threshold = confidence_threshold
        self.overlap_threshold = overlap_threshold

    def process_image(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w, _ = image.shape
        resized_h, resized_w = 640, 640
        if h > w:
            resized_w = int(640 * w / h)
        else:
            resized_h = int(640 * h / w)
        processed_image = cv2.resize(image, (resized_w, resized_h))
        pad_h = int((640 - resized_h) / 2)
        pad_w = int((640 - resized_w) / 2)
        processed_image = cv2.copyMakeBorder(
            processed_image, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        processed_image = (
            processed_image / 255.0).astype(np.float32).transpose(2, 0, 1).reshape(1, 3, 640, 640)
        return processed_image, pad_w, pad_h

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        x, pad_w, pad_h = self.process_image(image)
        outputs = self.session.run([output_name], {input_name: x})
        boxes = self.process_bounding_boxes(outputs, self.confidence_threshold)
        boxes = self.non_max_suppression(boxes, self.overlap_threshold)
        return boxes, pad_w, pad_h

    def process_bounding_boxes(self, boxes: list, confidence_threshold: float = 0.5) -> np.ndarray:
        bounding_boxes = []
        for output in boxes[0][0]:
            x, y, w, h = output[:4]
            cnf = output[4]
            cls = output[5]
            if cnf > confidence_threshold:
                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)
                bounding_boxes.append([x1, y1, x2, y2, cnf, cls])
        return np.array(bounding_boxes)

    def non_max_suppression(self, boxes: np.ndarray, overlapThresh: float = 0.5) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([])

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        cnf = boxes[:, 4]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))
        return boxes[pick]
