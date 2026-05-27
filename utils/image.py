import cv2

def letterbox(img, new_shape=(512, 512), color=(114, 114, 114)):
    shape = img.shape[:2]

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    new_unpad = (
        int(round(shape[1] * r)),
        int(round(shape[0] * r))
    )

    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad)

    top, bottom = int(round(dh)), int(round(dh))
    left, right = int(round(dw)), int(round(dw))

    img = cv2.copyMakeBorder(
        img, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )

    return img, r, (left, top)