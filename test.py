from face_snapchat import *

if __name__ == '__main__':
    should_show_bounds = False
    glasses = cv2.imread('resources/glasses/glasses.png', -1)
    guoqi = cv2.imread("resources/cheek/China.png", -1)
    moustache = cv2.imread("resources/moustache/hz3.png", -1)
    saihong = cv2.imread("resources/face/sh5.png", -1)
    ear = cv2.imread("resources/ear/ed2.png", -1)
    nose = cv2.imread("resources/nose/bz3.png", -1)
    frame = cv2.imread('static/img/yk.jpg', -1)[:, :, :3]
    landmar = get_landmarks(frame)
    if landmar == -1:
        exit()
    ros = []
    # 加上眼镜的图 和眼镜图
    ros.append(glasses_filter(frame, glasses, landmar, should_show_bounds))
    print(ros)
    # 加上胡子的图 和胡子图
    ros.append(moustache_filter(frame, moustache, landmar, should_show_bounds))
    # 脸一侧国旗
    ros.append(guoqi_filter(frame, guoqi, landmar, should_show_bounds))
    # 脸两侧 腮红
    ros.append(saihong_filter(frame, saihong, landmar, should_show_bounds))
    # 耳朵
    ros.append(ear_filter(frame, ear, landmar, should_show_bounds))
    # 鼻子
    ros.append(nose_filter(frame, nose, landmar, should_show_bounds))
    for r in ros:
        frame = blend_w_transparency(frame, r)
    cv2.imshow("resources", frame)
    cv2.waitKey()
    cv2.destroyAllWindows()
