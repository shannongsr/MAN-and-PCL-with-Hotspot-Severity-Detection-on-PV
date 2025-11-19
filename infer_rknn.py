import cv2
import numpy as np
from rknnlite.api import RKNNLite

RKNN_MODEL = 'YOLOv11-M.rknn'
IMG_PATH = 'test.jpg'
INPUT_SIZE = 640   # YOLO 输入尺寸

# -----------------------------
# 图像预处理
# -----------------------------
def letterbox(img, new_shape=640):
    h, w = img.shape[:2]
    s = new_shape / max(h, w)
    new_w, new_h = int(w * s), int(h * s)

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - new_w
    pad_h = new_shape - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    img_padded = cv2.copyMakeBorder(
        img_resized, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    return img_padded, s, (pad_left, pad_top)


def preprocess(path):
    img0 = cv2.imread(path)
    if img0 is None:
        raise FileNotFoundError(f"Cannot load image: {path}")

    img, ratio, (dw, dh) = letterbox(img0, INPUT_SIZE)
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.astype(np.float32)# / 255.0
   # img = img.transpose(2, 0, 1)  # HWC→CHW
    img = np.expand_dims(img, 0)
    return img, img0, ratio, (dw, dh)


# -----------------------------
# NMS + IOU
# -----------------------------
def iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)


def nms(boxes, scores, th=0.5):
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs):
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < th]
    return keep


# -----------------------------
# YOLO 后处理（自动识别维度）
# -----------------------------
def decode(pred, ratio, dwdh):
    """ pred shape: (N, C, 8400) """
    pred = np.squeeze(pred)  # (C,8400) or (8400,C)

    # 自动判断是否需要转置
    if pred.shape[0] < pred.shape[1]:
        pred = pred.transpose(1, 0)  # (8400,C)

    nc = pred.shape[1] - 5  # 类别数
    print(f"[AUTO] Detected num_classes = {nc}")

    boxes = pred[:, :4]
    scores = np.max(pred[:, 4:],axis=1)
    classes= np.argmax(pred[:, 4:],axis=1)
    mask = scores > 0.25
    boxes, scores, classes = pred[mask, :4], scores[mask], classes[mask]
        
    # xywh → xyxy
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    # 反 letterbox 缩放
    dw, dh = dwdh
    xyxy[:, [0, 2]] -= dw
    xyxy[:, [1, 3]] -= dh
    xyxy /= ratio

    # NMS
    keep = nms(xyxy, scores, 0.5)

    return xyxy[keep], scores[keep]+ classes[keep]


# -----------------------------
# 主流程
# -----------------------------
def main():
    print("=> Load RKNN...")
    rknn = RKNNLite()
    rknn.load_rknn(RKNN_MODEL)
    rknn.init_runtime()

    img, img0, ratio, dwdh = preprocess(IMG_PATH)

    print("=> Infer...")
    out = rknn.inference(inputs=[img])[0]
    print("RKNN output shape:", out.shape)

    boxes, scores = decode(out, ratio, dwdh)

    # 绘制
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img0, f"{sc:.2f}", (x1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imwrite("result.jpg", img0)
    print("=> Saved result.jpg")

    rknn.release()


if __name__ == "__main__":
    main()
