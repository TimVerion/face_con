import math
import cv2
import dlib
import numpy as np

predictor_path = "resources/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
COLOUR_CORRECT_BLUR = 0.5

MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))

POINTS = LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS
ALIGN_POINTS = POINTS
OVERLAY_POINTS = [POINTS]


def get_landmarks(img):
    rects = detector(img, 1)
    if len(rects) == 0:
        return -1
    return np.matrix([[p.x, p.y] for p in predictor(img, rects[0]).parts()])


def get_rotated_points(point, anchor, deg_angle):
    angle = math.radians(deg_angle)
    px, py = point
    ox, oy = anchor

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [int(qx), int(qy)]


def blend_w_transparency(face_img, overlay_image, alpha_img=None):
    # BGR
    overlay_img = overlay_image[:, :, :3]
    # A
    overlay_mask = overlay_image[:, :, 3:]

    background_mask = 255 - overlay_mask
    overlay_mask = cv2.cvtColor(overlay_mask, cv2.COLOR_GRAY2BGR)
    background_mask = cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR)

    face_part = (face_img * (1 / 255.0)) * (background_mask * (1 / 255.0))
    overlay_part = (overlay_img * (1 / 255.0)) * (overlay_mask * (1 / 255.0))
    # cast to 8 bit matrix
    if alpha_img is None:
        return np.uint8(cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0))
    else:
        a = np.where(background_mask == 255, 0, 1).astype(np.uint8)
        overlay_part = np.array(overlay_part * 255).astype(np.uint8)
        face_part = np.array(face_part * 255).astype(np.uint8)
        face_rou = np.array(face_img * a).astype(np.uint8)
        res = cv2.addWeighted(face_rou, 1 - alpha_img, overlay_part, alpha_img, 0)
        out = np.uint8(cv2.addWeighted(face_part, 1, res, 1, 0.0))
        return out


def glasses_filter(face, glasses, landmarks, should_show_bounds=False):
    glasses = cv2.resize(glasses, (620, 220))  # (666, 2813, 4)
    pts1 = np.float32([[0, 0], [599, 0], [0, 208], [599, 208]])
    if type(landmarks) is not int:
        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # get hypotenuse
        face_width = math.sqrt((right_face_extreme[0] - left_face_extreme[0]) ** 2 +
                               (right_face_extreme[1] - right_face_extreme[1]) ** 2)
        glasses_width = face_width * 1.0

        # top and bottom of left eye
        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height * 1.2

        # generate bounding box from the anchor points
        anchor_point = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, anchor_point, face_angle)

        tr = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, anchor_point, face_angle)

        bl = [int(anchor_point[0] - (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, anchor_point, face_angle)

        br = [int(anchor_point[0] + (glasses_width / 2)), int(anchor_point[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, anchor_point, face_angle)

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts)

        rotated = cv2.warpPerspective(glasses, m, (face.shape[1], face.shape[0]))
        result_2 = blend_w_transparency(face, rotated)

        if should_show_bounds:
            for p in pts:
                pos = (p[0], p[1])
                cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
                cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))

        # cv2.imshow("Glasses Filter", result_2)
        return rotated


def guoqi_filter(face, guoqi, landmarks, should_show_bounds=False):
    landmarks = get_landmarks(face)  # 获得68个特征点坐标
    h, w, t = guoqi.shape
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if type(landmarks) is not int:
        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]  # 做脸的初始位置
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]  # 右脸的结束位置
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]  # 脸宽
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]  # 脸部倾斜度

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # 得到斜边
        face_width = math.sqrt((x_diff_face) ** 2 + (y_diff_face) ** 2)
        guoqi_width = face_width * 0.25  # 胡子的占比

        # 左眼上下
        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height * 0.5  # 自己设计的眼镜的占比
        # 从锚点生成边界框
        brow_anchor = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(brow_anchor[0] - (guoqi_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, brow_anchor, face_angle)

        tr = [int(brow_anchor[0] + (guoqi_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, brow_anchor, face_angle)

        bl = [int(brow_anchor[0] - (guoqi_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, brow_anchor, face_angle)

        br = [int(brow_anchor[0] + (guoqi_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, brow_anchor, face_angle)

        # 在人中找到胡子的新位置
        top_philtrum_point = [landmarks[33, 0], landmarks[33, 1]]
        bottom_philtrum_point = [landmarks[51, 0], landmarks[51, 1]]
        philtrum_anchor = [(top_philtrum_point[0] + bottom_philtrum_point[0]) / 2,
                           (top_philtrum_point[1] + bottom_philtrum_point[1]) / 2]

        # determine distance from old origin to new origin and translate
        anchor_distance = [int(philtrum_anchor[0] - brow_anchor[0]), int(philtrum_anchor[1] - brow_anchor[1])]
        rot_tl[0] += anchor_distance[0] - int(face_width * 0.32)
        rot_tl[1] += anchor_distance[1]
        rot_tr[0] += anchor_distance[0] - int(face_width * 0.32)
        rot_tr[1] += anchor_distance[1]
        rot_bl[0] += anchor_distance[0] - int(face_width * 0.32)
        rot_bl[1] += anchor_distance[1]
        rot_br[0] += anchor_distance[0] - int(face_width * 0.32)
        rot_br[1] += anchor_distance[1]

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts)
        rotated = cv2.warpPerspective(guoqi, m, (face.shape[1], face.shape[0]))
        result_2 = blend_w_transparency(face, rotated)
        if should_show_bounds:
            for p in pts:
                pos = (p[0], p[1])
                cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
                cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))
            cv2.imshow("dd", result_2)
        return rotated


def moustache_filter(face, moustache, landmarks, should_show_bounds=False):
    landmarks = get_landmarks(face)
    h, w, t = moustache.shape
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if type(landmarks) is not int:
        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # get hypotenuse
        face_width = math.sqrt((right_face_extreme[0] - left_face_extreme[0]) ** 2 +
                               (right_face_extreme[1] - right_face_extreme[1]) ** 2)
        moustache_width = face_width * 0.7

        # top and bottom of left eye
        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height * 0.6

        # generate bounding box from the anchor points
        brow_anchor = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(brow_anchor[0] - (moustache_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, brow_anchor, face_angle)

        tr = [int(brow_anchor[0] + (moustache_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, brow_anchor, face_angle)

        bl = [int(brow_anchor[0] - (moustache_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, brow_anchor, face_angle)

        br = [int(brow_anchor[0] + (moustache_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, brow_anchor, face_angle)

        # locate new location for moustache on philtrum
        top_philtrum_point = [landmarks[33, 0], landmarks[33, 1]]
        bottom_philtrum_point = [landmarks[51, 0], landmarks[51, 1]]
        philtrum_anchor = [(top_philtrum_point[0] + bottom_philtrum_point[0]) / 2,
                           (top_philtrum_point[1] + bottom_philtrum_point[1]) / 2]

        # determine distance from old origin to new origin and translate
        anchor_distance = [int(philtrum_anchor[0] - brow_anchor[0]), int(philtrum_anchor[1] - brow_anchor[1])]
        rot_tl[0] += anchor_distance[0]
        rot_tl[1] += anchor_distance[1]
        rot_tr[0] += anchor_distance[0]
        rot_tr[1] += anchor_distance[1]
        rot_bl[0] += anchor_distance[0]
        rot_bl[1] += anchor_distance[1]
        rot_br[0] += anchor_distance[0]
        rot_br[1] += anchor_distance[1]

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts + 10)
        rotated = cv2.warpPerspective(moustache, m, (face.shape[1], face.shape[0]))
        result_2 = blend_w_transparency(face, rotated)

        # annotate_landmarks(result_2, landmarks)

        if should_show_bounds:
            for p in pts:
                pos = (p[0], p[1])
                cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
                cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))
        # cv2.imshow("Moustache Filter", result_2)
        return rotated


def saihong_filter(face, saihong, landmarks, should_show_bounds=False):
    landmarks = get_landmarks(face)
    h, w, t = saihong.shape
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if type(landmarks) is not int:
        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # get hypotenuse
        face_width = math.sqrt((right_face_extreme[0] - left_face_extreme[0]) ** 2 +
                               (right_face_extreme[1] - right_face_extreme[1]) ** 2)
        saihong_width = face_width * 0.9

        # top and bottom of left eye
        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height * 0.6

        # generate bounding box from the anchor points
        brow_anchor = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(brow_anchor[0] - (saihong_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, brow_anchor, face_angle)

        tr = [int(brow_anchor[0] + (saihong_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, brow_anchor, face_angle)

        bl = [int(brow_anchor[0] - (saihong_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, brow_anchor, face_angle)

        br = [int(brow_anchor[0] + (saihong_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, brow_anchor, face_angle)

        # locate new location for saihong on philtrum
        top_philtrum_point = [landmarks[33, 0], landmarks[33, 1]]
        bottom_philtrum_point = [landmarks[51, 0], landmarks[51, 1]]
        philtrum_anchor = [(top_philtrum_point[0] + bottom_philtrum_point[0]) / 2,
                           (top_philtrum_point[1] + bottom_philtrum_point[1]) / 2]

        # determine distance from old origin to new origin and translate
        anchor_distance = [int(philtrum_anchor[0] - brow_anchor[0]), int(0.7 * (philtrum_anchor[1] - brow_anchor[1]))]
        rot_tl[0] += anchor_distance[0]
        rot_tl[1] += anchor_distance[1]
        rot_tr[0] += anchor_distance[0]
        rot_tr[1] += anchor_distance[1]
        rot_bl[0] += anchor_distance[0]
        rot_bl[1] += anchor_distance[1]
        rot_br[0] += anchor_distance[0]
        rot_br[1] += anchor_distance[1]

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts)
        rotated = cv2.warpPerspective(saihong, m, (face.shape[1], face.shape[0]))
        result_2 = blend_w_transparency(face, rotated)

        # annotate_landmarks(result_2, landmarks)

        if should_show_bounds:
            for p in landmarks:
                pos = (p[0, 0], p[0, 1])
                cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
                # cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))
            cv2.imshow("saihong Filter", result_2)
        return rotated


def ear_filter(face, ear, landmarks, should_show_bounds=False):
    landmarks = get_landmarks(face)
    h, w, t = ear.shape
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if type(landmarks) is not int:
        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # get hypotenuse
        face_width = math.sqrt((right_face_extreme[0] - left_face_extreme[0]) ** 2 +
                               (right_face_extreme[1] - right_face_extreme[1]) ** 2)
        ear_width = face_width * 1.5

        # top and bottom of left eye
        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height

        # generate bounding box from the anchor points
        brow_anchor = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(brow_anchor[0] - (ear_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, brow_anchor, face_angle)

        tr = [int(brow_anchor[0] + (ear_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, brow_anchor, face_angle)

        bl = [int(brow_anchor[0] - (ear_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, brow_anchor, face_angle)

        br = [int(brow_anchor[0] + (ear_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, brow_anchor, face_angle)

        # locate new location for ear on philtrum
        top_philtrum_point = [landmarks[33, 0], landmarks[33, 1]]
        bottom_philtrum_point = [landmarks[51, 0], landmarks[51, 1]]
        philtrum_anchor = [(top_philtrum_point[0] + bottom_philtrum_point[0]) / 2,
                           (top_philtrum_point[1] + bottom_philtrum_point[1]) / 2]

        # determine distance from old origin to new origin and translate
        anchor_distance = [int(philtrum_anchor[0] - brow_anchor[0]), int(philtrum_anchor[1] - 1.7 * brow_anchor[1])]
        rot_tl[0] += anchor_distance[0]
        rot_tl[1] += anchor_distance[1]
        rot_tr[0] += anchor_distance[0]
        rot_tr[1] += anchor_distance[1]
        rot_bl[0] += anchor_distance[0]
        rot_bl[1] += anchor_distance[1]
        rot_br[0] += anchor_distance[0]
        rot_br[1] += anchor_distance[1]

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts)
        rotated = cv2.warpPerspective(ear, m, (face.shape[1], face.shape[0]))
        result_2 = blend_w_transparency(face, rotated)

        # annotate_landmarks(result_2, landmarks)

        if should_show_bounds:
            for p in landmarks:
                pos = (p[0, 0], p[0, 1])
                cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
                # cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))
            cv2.imshow("ear Filter", result_2)
        return rotated


def nose_filter(face, nose, landmarks, should_show_bounds=False):
    landmarks = get_landmarks(face)
    h, w, t = nose.shape
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if type(landmarks) is not int:
        left_face_extreme = [landmarks[0, 0], landmarks[0, 1]]
        right_face_extreme = [landmarks[16, 0], landmarks[16, 1]]
        x_diff_face = right_face_extreme[0] - left_face_extreme[0]
        y_diff_face = right_face_extreme[1] - left_face_extreme[1]

        face_angle = math.degrees(math.atan2(y_diff_face, x_diff_face))

        # get hypotenuse
        face_width = math.sqrt((right_face_extreme[0] - left_face_extreme[0]) ** 2 +
                               (right_face_extreme[1] - right_face_extreme[1]) ** 2)
        nose_width = face_width * 0.4

        # top and bottom of left eye
        eye_height = math.sqrt((landmarks[19, 0] - landmarks[28, 0]) ** 2 +
                               (landmarks[19, 1] - landmarks[28, 1]) ** 2)
        glasses_height = eye_height

        # generate bounding box from the anchor points
        brow_anchor = [landmarks[27, 0], landmarks[27, 1]]
        tl = [int(brow_anchor[0] - (nose_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tl = get_rotated_points(tl, brow_anchor, face_angle)

        tr = [int(brow_anchor[0] + (nose_width / 2)), int(brow_anchor[1] - (glasses_height / 2))]
        rot_tr = get_rotated_points(tr, brow_anchor, face_angle)

        bl = [int(brow_anchor[0] - (nose_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_bl = get_rotated_points(bl, brow_anchor, face_angle)

        br = [int(brow_anchor[0] + (nose_width / 2)), int(brow_anchor[1] + (glasses_height / 2))]
        rot_br = get_rotated_points(br, brow_anchor, face_angle)

        # locate new location for nose on philtrum
        top_philtrum_point = [landmarks[33, 0], landmarks[33, 1]]
        bottom_philtrum_point = [landmarks[51, 0], landmarks[51, 1]]
        philtrum_anchor = [(top_philtrum_point[0] + bottom_philtrum_point[0]) / 2,
                           (top_philtrum_point[1] + bottom_philtrum_point[1]) / 2]

        # determine distance from old origin to new origin and translate
        anchor_distance = [int(philtrum_anchor[0] - brow_anchor[0]), int(philtrum_anchor[1] - 1.1 * brow_anchor[1])]
        rot_tl[0] += anchor_distance[0]
        rot_tl[1] += anchor_distance[1]
        rot_tr[0] += anchor_distance[0]
        rot_tr[1] += anchor_distance[1]
        rot_bl[0] += anchor_distance[0]
        rot_bl[1] += anchor_distance[1]
        rot_br[0] += anchor_distance[0]
        rot_br[1] += anchor_distance[1]

        pts = np.float32([rot_tl, rot_tr, rot_bl, rot_br])
        m = cv2.getPerspectiveTransform(pts1, pts)
        rotated = cv2.warpPerspective(nose, m, (face.shape[1], face.shape[0]))
        result_2 = blend_w_transparency(face, rotated)

        # annotate_landmarks(result_2, landmarks)

        if should_show_bounds:
            for p in landmarks:
                pos = (p[0, 0], p[0, 1])
                cv2.circle(result_2, pos, 2, (0, 0, 255), 2)
                # cv2.putText(result_2, str(p), pos, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 0, 0))
            cv2.imshow("nose Filter", result_2)
        return rotated
