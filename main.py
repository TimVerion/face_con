from flask import Flask, jsonify, render_template, request
import dlib, numpy
from face_snapchat import *
import base64
import cv2
import json


def change_face(types, should_show_bounds=False):
    ros = []
    frame = types["base64_11"][:, :, :3]
    landmar = get_landmarks(frame)
    if landmar is -1:
        return 0
    # 加上眼镜的图 和眼镜图
    if types["glasses"] != str(0):
        glasses = cv2.imread("resources/glasses/" + types["glasses"] + ".png", -1)
        ros.append(glasses_filter(frame, glasses, should_show_bounds))
    # 脸一侧国旗
    if types["guoqi"] != str(0):
        guoqi = cv2.imread("resources/cheek/" + types["guoqi"] + ".png", -1)
        ros.append(guoqi_filter(frame, guoqi, should_show_bounds))
    # 加上胡子的图 和胡子图
    if types["moustache"] != str(0):
        moustache = cv2.imread("resources/moustache/" + types["moustache"] + ".png", -1)
        ros.append(moustache_filter(frame, moustache, should_show_bounds))
    # 脸两侧 腮红
    if types["face"] != str(0):
        saihong = cv2.imread("resources/face/sh5.png" + types["face"] + ".png", -1)
        ros.append(saihong_filter(frame, saihong, should_show_bounds))
    # 耳朵
    if types["ear"] != str(0):
        ear = cv2.imread("resources/ear/" + types["ear"] + ".png", -1)
        ros.append(ear_filter(frame, ear, should_show_bounds))
    # 鼻子
    if types["nose"] != str(0):
        nose = cv2.imread("resources/nose/" + types["nose"] + ".png", -1)
        ros.append(nose_filter(frame, nose, should_show_bounds))
    for r in ros:
        frame = blend_w_transparency(frame, r)
    # return frame
    cv2.imwrite("static/img/yk_.jpg", frame)
    return 1


############################

# 定义接口
app = Flask(__name__)


@app.route('/api/face_add', methods=['post'])
def face_add():
    types = request.json.copy()
    base64_1 = types["base64_11"]
    imgdata = base64.b64decode(base64_1)
    img_array = np.fromstring(imgdata, np.uint8)  # 转换np序列
    img = cv2.imdecode(img_array, cv2.COLOR_BGR2RGB)  # 转换Opencv格式
    cv2.imwrite("static/img/yk.jpg", img)
    types["base64_11"] = img
    return jsonify(results=[change_face(types)])


# 绑定前台页面
@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000, debug='True')
