from campone import Video, process_lines

video = Video(30, (1280, 720))

cap = video.cap

while True:
    frame = video.get_frame()
    error = process_lines(frame)

    video.show(frame)
    if video.exit():
        break

video.release()
