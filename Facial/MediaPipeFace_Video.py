import os
import cv2
import mediapipe as mp
from tqdm import tqdm


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def main():
    isForThesis = True ### 2024/07/25
    isHistory = False ### 2024/07/25

    if isForThesis:
        ### Reference: https://colab.research.google.com/drive/1FCxIsJS9i58uAsgsLFqDwFmiPO14Z2Hd#scrollTo=BAivyQ_xOtFp
        with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
            pid = 27
            thesis_folder = "data/" + str(pid) + "/Thesis_used/"
            if not os.path.exists(thesis_folder):
                os.makedirs(thesis_folder)

            video_input = "data/" + str(pid) + "/eyeVideo.mp4"
            video_output = thesis_folder + "mediapipe_facemesh_result.mp4"
            print(f'[Input video] {video_input}')
            print(f'[Output video] {video_output}')

            cap = cv2.VideoCapture(video_input)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            FPS = cap.get(cv2.CAP_PROP_FPS)

            ### Define Video Writer
            VIDEO_CODEC = "mp4v"
            video_writer = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (frameWidth, frameHeight))

            for frame in tqdm(range(n_frames), total=n_frames):
                ret, image = cap.read()
                if ret == False:
                    break

                ### Convert the BGR image to RGB before processing.
                rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                results = face_mesh.process(rgb_image)

                if not results.multi_face_landmarks:
                    continue
                annotated_image = image.copy()
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                    mp_drawing.draw_landmarks(
                        image=annotated_image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                video_writer.write(annotated_image)
            video_writer.release()
            cap.release()



    if isHistory:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            cap = cv2.VideoCapture("test-video/video2.mp4")
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            FPS = cap.get(cv2.CAP_PROP_FPS)

            ### Define Video Writer
            video_output_path = "test-video/output-video2.mp4"
            VIDEO_CODEC = "mp4v"
            video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (frameWidth, frameHeight))

            for frame in tqdm(range(n_frames), total=n_frames):
                ret, image = cap.read()
                if ret == False:
                    break
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if not results.multi_face_landmarks:
                    print(f'[Notice] Something wrong.')
                else:
                    annotated_image = image.copy()
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        video_writer.write(annotated_image)

if __name__ == "__main__":
    main()