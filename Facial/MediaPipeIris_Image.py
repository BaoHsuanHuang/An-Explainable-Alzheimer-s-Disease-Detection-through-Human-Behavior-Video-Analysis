import cv2
import math
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  



COLOR_GREEN = (0, 255, 0)

def landmarksDetection(img, results, draw=False):
    imgHeight, imgWidth= img.shape[:2]

    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * imgWidth), int(point.y * imgHeight)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(img, p, 2, COLOR_GREEN, -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

def fillPolyTrans(img, points, color, opacity):
    """
    @param img: (mat) input image, where shape is drawn.
    @param points: list [tuples(int, int) these are the points custom shape,FillPoly
    @param color: (tuples (int, int, int)
    @param opacity:  it is transparency of image.
    @return: img(mat) image with rectangle draw.
    """
    list_to_np_array = np.array(points, dtype=np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay,[list_to_np_array], color )
    new_img = cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0)
    img = new_img
    cv2.polylines(img, [list_to_np_array], True, color,1, cv2.LINE_AA)
    return img

def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

def calculateBlinkRatio(img, landmarks, right_indices, left_indices):
    ### [Right] horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    ### [Right] vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    ### draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    ### [Left] horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    ### [Left] vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance
    ratio = (reRatio+leRatio)/2

    return ratio 




def main():
    with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
        image = cv2.imread('test-image/image1.jpg')

        ### Convert the BGR image to RGB before processing.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            print(f'[Notice] Something wrong.')
        else:
            FRAME_COUNTER = 0
            CLOSED_EYES_FRAME = 3
            TOTAL_BLINKS = 0

            mesh_coords = landmarksDetection(image, results, False)
            blinkRatio = calculateBlinkRatio(image, mesh_coords, RIGHT_EYE, LEFT_EYE)
            print(f'blinkRatio: {blinkRatio}')

            if blinkRatio > 5.5:
                FRAME_COUNTER += 1
            else:
                if FRAME_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    FRAME_COUNTER = 0
                    print(f'Total Blinks: {TOTAL_BLINKS}')

            image = fillPolyTrans(image, [mesh_coords[p] for p in LEFT_EYE], COLOR_GREEN, opacity=0.4)
            image = fillPolyTrans(image, [mesh_coords[p] for p in RIGHT_EYE], COLOR_GREEN, opacity=0.4)
            
            cv2.imwrite('test-image/iris-image1.jpg', image)

if __name__ == "__main__":
    main()