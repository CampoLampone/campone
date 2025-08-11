import campone
import cv2


video = campone.Video(30, (1280, 720))

cap = video.cap

while True:
    img = video.get_frame() #WHY IS IT NOT WORKING

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    video.show(img, img_gray)
    if video.exit():
        break

video.release()
