import cv2
import numpy as np

cap = cv2.VideoCapture('./data/square.mp4')

initialSearchWindow = []

start = [55, 55]
end = [104, 104]
searchWindowSize = [end[0] - start[0], end[1] - start[1]]
print("Search window: " + str(searchWindowSize[0]) + " " + str(searchWindowSize[1]))

motion_vectors = []


def motion_vector(fr, starting_position):
    block = np.zeros((searchWindowSize[0], searchWindowSize[1]))
    block.fill(28)
    current_frame = fr

    bl = block.tolist()
    fr = fr.tolist()
    has_found = False
    i = 0
    j = 0

    for fx in range(starting_position[0], len(fr) - 1):
        if has_found:
            break

        for fy in range(starting_position[1], len(fr[fx]) - 1):
            if has_found:
                break

            for bx in range(0, len(bl)):
                if bx + fx >= len(fr) or has_found:
                    # print("Breaking at: X: " + str(bx + fx))
                    break

                for by in range(0, len(bl[bx])):
                    if by + fy > len(fr[bx + fx]) or has_found:
                        # print("Breaking at: Y: " + str(by + fy) + " in row: " + str(bx + fx))
                        break

                    # v = current_frame[fx + bx][fy + by]
                    # current_frame[fx + bx][fy + by] = 255
                    # cv2.imshow("frame", current_frame)
                    # name = "fr" + str(fx + bx) + "," + str(fy + by) + ".png"
                    # cv2.imwrite(name, current_frame)
                    # current_frame[fx + bx][fy + by] = v

                    # try:

                    i = fx + bx
                    j = fy + by

                    if i >= len(fr) or j >= len(fr[i]):
                        break

                    v1 = fr[i][j]
                    v2 = bl[bx][by]

                    diff = abs(v1 - v2)

                    if diff == 0:
                        has_found = True


    current[i, j] = 255
    cv2.imshow("frame", current)
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
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        break

    current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    startingPosition = motion_vector(current, startingPosition)

    print(motion_vectors)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# print(motion_vectors)
