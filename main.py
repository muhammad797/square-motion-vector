import cv2
import numpy as np

cap = cv2.VideoCapture('./data/square.mp4')

initialSearchWindow = []

start = [55, 55]
end = [104, 104]
searchWindowSize = [end[0] - start[0], end[1] - start[1]]
print("Search window: " + str(searchWindowSize[0]) + " " + str(searchWindowSize[1]))

motion_vectors = []


def motion_vector(fr, startingPosition):
    block = np.zeros((searchWindowSize[0], searchWindowSize[1]))
    block.fill(28)

    bl = block.tolist()
    fr = fr.tolist()
    has_found = False
    i = 0
    j = 0
    for fx in range(startingPosition[0], len(fr)):
        if has_found:
            break

        for fy in range(startingPosition[1], len(fr[fx])):
            if has_found:
                break

            diff = 0
            for bx in range(0, len(bl)):
                if has_found or bx + fx > len(bl):
                    break

                for by in range(0, len(bl[bx])):
                    if has_found or by + fy > len(bl):
                        break

                    i = fx
                    j = fy

                    diff = diff + abs(fr[fx + bx][fy + by] - bl[bx][by])

                    if diff == 0:
                        has_found = True

    print("")

    done_count = len(motion_vectors)
    if done_count == 0:
        result = [i, j]
        motion_vectors.append(result)

    else:
        result = [i - motion_vectors[done_count - 1][0], j - motion_vectors[done_count - 1][1]]
        motion_vectors.append(result)

    return result


startingPosition = [0, 0]
hasPrinted = False
while cap.isOpened():
    ret, frame = cap.read()

    current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', current)
    startingPosition = motion_vector(current, startingPosition)
    print(motion_vectors)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(motion_vectors)
