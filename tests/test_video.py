from campone import Video, process, process_lines, is_intersection

video = Video(30, (1280, 720))

cap = video.cap

while True:
    frame = video.get_frame()
    filtered_yellow, filtered_white = process(frame)
    error = process_lines(filtered_yellow, filtered_white)

    if is_intersection(filtered_yellow):
        print("Is intersection!")

    video.show(frame)
    if video.exit():
        break

video.release()
