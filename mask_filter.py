import cv2 as cv
import numpy as np
import mediapipe as mp

img = cv.imread(r"D:\Virtual_glass_try_on\51x4s4iSfQL._AC_UY1100_-removebg-preview.png")
cap = cv.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                            min_detection_confidence=0.5, min_tracking_confidence=0.3)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Define landmarks to use
            landmark_indices = [10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                                172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109]

            landmark_coords = [(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                int(face_landmarks.landmark[i].y * frame.shape[0])) for i in landmark_indices]
            mask = np.zeros(frame.shape, dtype=np.uint8)
            cv.fillPoly(mask, [np.array(landmark_coords, dtype=np.int32)], (255, 255, 255))
            gray_mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
            contours, _ = cv.findContours(gray_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv.boundingRect(contours[0])
                # resizing the filter according to the face detected
                resized_img = cv.resize(img, (w, h))
                # selecting that region from face where we want to apply the filter
                roi = frame[y:y + h, x:x + w]
                # converting the image to gray
                img_gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
                # creating the mask of the image
                _, img_mask = cv.threshold(img_gray, 1, 255, cv.THRESH_BINARY)
                img_mask_inv = cv.bitwise_not(img_mask)
                # selecting that region from frame where filter will be applied
                frame_bg = cv.bitwise_and(roi, roi, mask=img_mask_inv)
                # so img_mask has created a mask and now we will take those region from resized image which is there in mask
                img_fg = cv.bitwise_and(resized_img, resized_img, mask=img_mask)
                # it will add the filter in that region
                dst = cv.add(frame_bg, img_fg)
                frame[y:y+h, x:x+w] = dst
        # Show augmented reality frame
        cv.imshow('Ar', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()
