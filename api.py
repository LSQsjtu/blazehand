import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For static images:
root = 'D:/CVlabortory/hand_detection'
path = os.path.join(root, 'pics')
filenames = os.listdir(path)
file_list = [os.path.join(path, filename) for filename in filenames]

hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
)
nonFingerId = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]

f = open(path + "/datalist.txt", "w")
for idx, file in enumerate(file_list):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    # print("Handedness:", results.multi_handedness[0].classification[0])

    if not results.multi_hand_landmarks:
        f.write("0 0,0 0,0,0,0 ")
        for i in range(20):
            f.write("0,0,0,")
        f.write("0,0,0 ")
        f.write("0 0,0 0,0,0,0 ")
        for i in range(20):
            f.write("0,0,0,")
        f.write("0,0,0 ")
        continue
    image_hight, image_width, _ = image.shape
    # annotated_image = image.copy()
    num = len(results.multi_hand_landmarks)
    if num == 1:
        j = 0
        hand_landmarks = results.multi_hand_landmarks[0]
        xmin = ymin = 1
        xmax = ymax = 0
        for i in nonFingerId:
            xmin = min(hand_landmarks.landmark[i].x, xmin)
            xmax = max(hand_landmarks.landmark[i].x, xmax)
            ymin = min(hand_landmarks.landmark[i].y, ymin)
            ymax = max(hand_landmarks.landmark[i].y, ymax)

        # handDownx = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
        # handUpx = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
        # handDowny = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
        # handUpy = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
        # angleRad = math.atan2(handDownx - handUpx, handDowny - handUpy)
        xscale = xmax - xmin
        yscale = ymax - ymin
        x_center = xmin + xscale * 0.5
        y_center = ymin + yscale * 0.5
        yrescale = yscale / 2 * 2.1
        xrescale = xscale / 2 * 2.1
        min_x = min(xrescale, x_center)
        min_y = min(yrescale, y_center)
        w = min_x + min(min_x, 1 - x_center)
        h = min_y + min(min_y, 1 - y_center)

        if results.multi_handedness[j].classification[0].label == "Right":
            f.write(
                str(results.multi_handedness[j].classification[0].score)
                + " "
                + str(image_width)
                + ","
                + str(image_hight)
                + " "
                + str(x_center - min_x)
                + ","
                + str(y_center - min_y)
                + ","
                + str(w)
                + ","
                + str(h)
                + " "
            )
            for i in range(20):
                f.write(
                    str(hand_landmarks.landmark[i].x)
                    + ","
                    + str(hand_landmarks.landmark[i].y)
                    + ","
                    + str(hand_landmarks.landmark[i].z)
                    + ","
                )
            f.write(
                str(hand_landmarks.landmark[20].x)
                + ","
                + str(hand_landmarks.landmark[20].y)
                + ","
                + str(hand_landmarks.landmark[20].z)
                + " "
            )
            f.write("0 0,0 0,0,0,0 ")
            for i in range(20):
                f.write("0,0,0,")
            f.write("0,0,0 ")
        else:
            f.write("0 0,0 0,0,0,0 ")
            for i in range(20):
                f.write("0,0,0,")
            f.write("0,0,0 ")
            f.write(
                str(results.multi_handedness[j].classification[0].score)
                + " "
                + str(image_width)
                + ","
                + str(image_hight)
                + " "
                + str(x_center - min_x)
                + ","
                + str(y_center - min_y)
                + ","
                + str(w)
                + ","
                + str(h)
                + " "
            )
            for i in range(20):
                f.write(
                    str(hand_landmarks.landmark[i].x)
                    + ","
                    + str(hand_landmarks.landmark[i].y)
                    + ","
                    + str(hand_landmarks.landmark[i].z)
                    + ","
                )
            f.write(
                str(hand_landmarks.landmark[20].x)
                + ","
                + str(hand_landmarks.landmark[20].y)
                + ","
                + str(hand_landmarks.landmark[20].z)
                + " "
            )
    elif num == 2:
        if results.multi_handedness[0].classification[0].label == "Right":
            j = 0
            for hand_landmarks in results.multi_hand_landmarks:
                xmin = ymin = 1
                xmax = ymax = 0
                for i in nonFingerId:
                    xmin = min(hand_landmarks.landmark[i].x, xmin)
                    xmax = max(hand_landmarks.landmark[i].x, xmax)
                    ymin = min(hand_landmarks.landmark[i].y, ymin)
                    ymax = max(hand_landmarks.landmark[i].y, ymax)

                xscale = xmax - xmin
                yscale = ymax - ymin
                x_center = xmin + xscale * 0.5
                y_center = ymin + yscale * 0.5
                yrescale = yscale / 2 * 2.1
                xrescale = xscale / 2 * 2.1
                min_x = min(xrescale, x_center)
                min_y = min(yrescale, y_center)
                w = min_x + min(min_x, 1 - x_center)
                h = min_y + min(min_y, 1 - y_center)

                f.write(
                    str(results.multi_handedness[j].classification[0].score)
                    + " "
                    + str(image_width)
                    + ","
                    + str(image_hight)
                    + " "
                    + str(x_center - min_x)
                    + ","
                    + str(y_center - min_y)
                    + ","
                    + str(w)
                    + ","
                    + str(h)
                    + " "
                )
                for i in range(20):
                    f.write(
                        str(hand_landmarks.landmark[i].x)
                        + ","
                        + str(hand_landmarks.landmark[i].y)
                        + ","
                        + str(hand_landmarks.landmark[i].z)
                        + ","
                    )
                f.write(
                    str(hand_landmarks.landmark[20].x)
                    + ","
                    + str(hand_landmarks.landmark[20].y)
                    + ","
                    + str(hand_landmarks.landmark[20].z)
                    + " "
                )
                j = j + 1
        else:
            for j in [1, 0]:
                hand_landmarks = results.multi_hand_landmarks[j]
                xmin = ymin = 1
                xmax = ymax = 0
                for i in nonFingerId:
                    xmin = min(hand_landmarks.landmark[i].x, xmin)
                    xmax = max(hand_landmarks.landmark[i].x, xmax)
                    ymin = min(hand_landmarks.landmark[i].y, ymin)
                    ymax = max(hand_landmarks.landmark[i].y, ymax)

                xscale = xmax - xmin
                yscale = ymax - ymin
                x_center = xmin + xscale * 0.5
                y_center = ymin + yscale * 0.5
                yrescale = yscale / 2 * 2.1
                xrescale = xscale / 2 * 2.1
                min_x = min(xrescale, x_center)
                min_y = min(yrescale, y_center)
                w = min_x + min(min_x, 1 - x_center)
                h = min_y + min(min_y, 1 - y_center)

                f.write(
                    str(results.multi_handedness[j].classification[0].score)
                    + " "
                    + str(image_width)
                    + ","
                    + str(image_hight)
                    + " "
                    + str(x_center - min_x)
                    + ","
                    + str(y_center - min_y)
                    + ","
                    + str(w)
                    + ","
                    + str(h)
                    + " "
                )
                for i in range(20):
                    f.write(
                        str(hand_landmarks.landmark[i].x)
                        + ","
                        + str(hand_landmarks.landmark[i].y)
                        + ","
                        + str(hand_landmarks.landmark[i].z)
                        + ","
                    )
                f.write(
                    str(hand_landmarks.landmark[20].x)
                    + ","
                    + str(hand_landmarks.landmark[20].y)
                    + ","
                    + str(hand_landmarks.landmark[20].z)
                    + " "
                )
            # cv2.rectangle(
            #     annotated_image,
            #     (
            #         int((x_center - min_x) * image_width),
            #         int((y_center - min_y) * image_hight),
            #     ),
            #     (
            #         int((x_center + min_x) * image_width),
            #         int((y_center + min_y) * image_hight),
            #     ),
            #     (255, 0, 0),
            #     2,
            # )
    # cv2.imwrite("annotated_image" + str(idx) + ".png", cv2.flip(annotated_image, 1))
    f.write("\n")
hands.close()
f.close()

# # For webcam input:
# hands = mp_hands.Hands(
#     min_detection_confidence=0.5, min_tracking_confidence=0.5)
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#   success, image = cap.read()
#   if not success:
#     print("Ignoring empty camera frame.")
#     # If loading a video, use 'break' instead of 'continue'.
#     continue

#   # Flip the image horizontally for a later selfie-view display, and convert
#   # the BGR image to RGB.
#   image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
#   # To improve performance, optionally mark the image as not writeable to
#   # pass by reference.
#   image.flags.writeable = False
#   results = hands.process(image)

#   # Draw the hand annotations on the image.
#   image.flags.writeable = True
#   image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#   if results.multi_hand_landmarks:
#     for hand_landmarks in results.multi_hand_landmarks:
#       mp_drawing.draw_landmarks(
#           image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#   cv2.imshow('MediaPipe Hands', image)
#   if cv2.waitKey(5) & 0xFF == 27:
#     break
# hands.close()
# cap.release()