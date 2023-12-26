import cv2
import utlis

webcam = False
path = 'np.jpg'
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)
scale = 3
wp = 210 * scale
hp = 297 * scale

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)

    imgContours, conts = utlis.getContours(img, minArea=50000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        imgWarp = utlis.warpImg(img, biggest, wp, hp)

        imgContours2, conts2 = utlis.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)

        if len(conts2) != 0:
            for obj in conts2:
                nPoints = utlis.reorder(obj[2])
                nW = round((utlis.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale) / 10), 1)
                nH = round((utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale) / 10), 1)

                x, y, w, h = obj[3]
                cv2.putText(img, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

                cv2.putText(img, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)

        cv2.imshow('A4', imgContours2)

    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    utlis.getContours(img, cThr=[30, 120], showCanny=True)
    cv2.imshow('Original', img)
    cv2.waitKey(1)




