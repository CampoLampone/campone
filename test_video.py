import campone

video = campone.Video(30, (1280, 720))

cap = video.cap

if not video.is_streaming:
    print("Failed to start streaming")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    video.transfer(frame)
    if video.exit():
        break

video.release()
