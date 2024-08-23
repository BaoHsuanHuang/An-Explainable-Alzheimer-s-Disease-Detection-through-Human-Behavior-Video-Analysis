import os
import cv2
import math
import numpy as np
import mediapipe as mp
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_IRIS = [474,475, 476, 477]
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_IRIS = [469, 470, 471, 472]

FONTS = cv2.FONT_HERSHEY_COMPLEX

COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_PINK = (147,20,255)
COLOR_YELLOW = (0,255,255)


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
    tmp = 1e9
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

    reRatio = rhDistance/(rvDistance+tmp)
    leRatio = lhDistance/(lvDistance+tmp)
    ratio = (reRatio+leRatio)/2

    return ratio 

def colorBackgroundText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
    """
    Draws text with background, with  control transparency
    @param img:(mat) which you want to draw text
    @param text: (string) text you want draw
    @param font: fonts face, like FONT_HERSHEY_COMPLEX, FONT_HERSHEY_PLAIN etc.
    @param fontScale: (double) the size of text, how big it should be.
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be
    @param textPos: tuple(x,y) position where you want to draw text
    @param textThickness:(int) fonts weight, how bold it should be.
    @param textColor: tuple(BGR), values -->0 to 255 each
    @param bgColor: tuple(BGR), values -->0 to 255 each
    @param pad_x: int(pixels)  padding of in x direction
    @param pad_y: int(pixels) 1 to 1.0 (), controls transparency of  text background 
    @return: img(mat) with draw with background
    """
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv2.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    cv2.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img


def main():
    isForThesis = True ### 2024/07/25
    isHistory = False ### 2024/07/25

    if isForThesis==True:
        # Load drawing_utils and drawing_styles
        mp_drawing = mp.solutions.drawing_utils 
        mp_drawing_styles = mp.solutions.drawing_styles

        # with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
        #     pid = 27
        #     thesis_folder = "data/" + str(pid) + "/Thesis_used/"
        #     if not os.path.exists(thesis_folder):
        #         os.makedirs(thesis_folder)

        #     video_input = "data/" + str(pid) + "/eyeVideo.mp4"
        #     video_output = thesis_folder + "mediapipe_facemesh_result.mp4"

        #     cap = cv2.VideoCapture(video_input)
        #     n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #     frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #     FPS = cap.get(cv2.CAP_PROP_FPS)

        #     ### Define Video Writer
        #     VIDEO_CODEC = "mp4v"
        #     video_writer = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (frameWidth, frameHeight))

        #     for frame in tqdm(range(n_frames), total=n_frames):
        #         ret, image = cap.read()
        #         if ret == False:
        #             break

        #         ### Convert the BGR image to RGB before processing.
        #         rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #         results = face_mesh.process(rgb_image)

        #         if not results.multi_face_landmarks:
        #             continue
        #         annotated_image = image.copy()
        #         for face_landmarks in results.multi_face_landmarks:
        #             mp_drawing.draw_landmarks(
        #                 image=annotated_image,
        #                 landmark_list=face_landmarks,
        #                 connections=mp_face_mesh.FACEMESH_TESSELATION,
        #                 landmark_drawing_spec=None,
        #                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        #             mp_drawing.draw_landmarks(
        #                 image=annotated_image,
        #                 landmark_list=face_landmarks,
        #                 connections=mp_face_mesh.FACEMESH_CONTOURS,
        #                 landmark_drawing_spec=None,
        #                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        #             mp_drawing.draw_landmarks(
        #                 image=annotated_image,
        #                 landmark_list=face_landmarks,
        #                 connections=mp_face_mesh.FACEMESH_IRISES,
        #                 landmark_drawing_spec=None,
        #                 connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())


        #         video_writer.write(annotated_image)
        #     video_writer.release()
        #     cap.release()

    if isHistory==True:
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
            cap = cv2.VideoCapture("test-video/video4.mp4")
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            FPS = cap.get(cv2.CAP_PROP_FPS)

            ### Define Video Writer
            video_output_path = "test-video/iris-video4.mp4"
            VIDEO_CODEC = "mp4v"
            video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (frameWidth, frameHeight))

            FRAME_COUNTER = 0
            TOTAL_BLINKS = 0
            CLOSED_EYES_FRAME = 3

            for frame in tqdm(range(n_frames), total=n_frames):
                ret, image = cap.read()
                if ret == False:
                    break

                ### Convert the BGR image to RGB before processing.
                rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                results = face_mesh.process(rgb_image)

                if not results.multi_face_landmarks:
                    print(f'[Notice] Something wrong.')
                else:
                    mesh_coords = landmarksDetection(image, results, False)
                    blinkRatio = calculateBlinkRatio(image, mesh_coords, RIGHT_EYE, LEFT_EYE)
                    image = colorBackgroundText(image,  f'Ratio : {round(blinkRatio, 2)}', FONTS, 0.7, (30,100),2, COLOR_PINK, COLOR_YELLOW)
                    # print(f'blinkRatio: {blinkRatio}')

                    if blinkRatio > 5.5:
                        FRAME_COUNTER += 1
                        image = colorBackgroundText(image,  f'Blink', FONTS, 1.7, (int(frameHeight/2), 100), 2, COLOR_BLUE, pad_x=6, pad_y=6)
                    else:
                        if FRAME_COUNTER > CLOSED_EYES_FRAME:
                            TOTAL_BLINKS += 1
                            FRAME_COUNTER = 0
                            # print(f'Total Blinks: {TOTAL_BLINKS}')

                    image = colorBackgroundText(image,  f'Total Blinks: {TOTAL_BLINKS}', FONTS, 0.7, (30,150), 2)
                    # image = fillPolyTrans(image, [mesh_coords[p] for p in LEFT_EYE], COLOR_GREEN, opacity=0.4)
                    # image = fillPolyTrans(image, [mesh_coords[p] for p in RIGHT_EYE], COLOR_GREEN, opacity=0.4)
                    image = cv2.polylines(image,  [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, COLOR_GREEN, 1, cv2.LINE_AA)
                    image = cv2.polylines(image,  [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, COLOR_GREEN, 1, cv2.LINE_AA)

                    ### Iris segmentation
                    mesh_points = np.array([np.multiply([p.x, p.y], [frameWidth, frameHeight]).astype(int) for p in results.multi_face_landmarks[0].landmark])
                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)
                    image = cv2.circle(image, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
                    image = cv2.circle(image, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

                    video_writer.write(image)


if __name__ == "__main__":
    main()