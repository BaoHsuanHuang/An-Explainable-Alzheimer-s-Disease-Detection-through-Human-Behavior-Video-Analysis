import os
import cv2
import csv
import math
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts, plot_skeleton_kpts_RLeg
from utils.general import non_max_suppression_kpt, strip_optimizer
from tqdm import tqdm



def extractKeypoints(device, model, frame):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = letterbox(frame, (frameWidth), stride=64, auto=True)[0]
    frame_ = frame.copy()
    frame = transforms.ToTensor()(frame)
    frame = torch.tensor(np.array([frame.numpy()]))
    frame = frame.to(device)
    frame = frame.float()

    with torch.no_grad():
        output, _ = model(frame)
    
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
    output = output_to_keypoint(output)
    result = frame[0].permute(1, 2, 0) * 255
    result = result.cpu().numpy().astype(np.uint8)
    
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    coordinates = np.zeros((51,), dtype=float)
    if output.shape[0] == 0:
        # print(f'No people detected.')
        noPeople = True
    else:
        noPeople = False
        for idx in range(output.shape[0]):
            ### extract keypoints coordinates
            coordinates = output[idx, 7:]
            # print(f'type(coordinates): {coordinates.shape}')

            plot_skeleton_kpts(result, output[idx, 7:].T, 3)
            xmin, ymin = (output[idx, 2]-output[idx, 4]/2), (output[idx, 3]-output[idx, 5]/2)
            xmax, ymax = (output[idx, 2]+output[idx, 4]/2), (output[idx, 3]+output[idx, 5]/2)
            cv2.rectangle(result,(int(xmin), int(ymin)),(int(xmax), int(ymax)), color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    
    return output, result, coordinates, noPeople

def preProcessing(image):
    ### Gray scale
    # result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # result = np.stack((result,)*3, axis=-1)

    ### Enhance contrast: https://www.wongwonggoods.com/all-posts/python/python_opencv/opencv-modify-contrast/s
    # brightness = 0
    # contrast = 100 # - 減少對比度/+ 增加對比度
    # B = brightness / 255.0
    # c = contrast / 255.0 
    # k = math.tan((45 + 44 * c) / 180 * math.pi)
    # image = (image - 127.5 * (1 - B)) * k + 127.5 * (1 + B)
    # result = np.clip(image, 0, 255).astype(np.uint8)

    ### Enhance lightness, saturation: https://www.wongwonggoods.com/all-posts/python/python_opencv/opencv-lightness-saturation/
    fImg = image.astype(np.float32) ### 圖像歸一化，且轉換為浮點型
    fImg = fImg / 255.0

    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS) ### 顏色空間轉換 BGR -> HLS
    hlsCopy = np.copy(hlsImg)

    lightness = 30 # lightness 調整為  "1 +/- 幾 %"
    saturation = 30 # saturation 調整為 "1 +/- 幾 %"

    # 亮度調整
    hlsCopy[:, :, 1] = (1 + lightness / 100.0) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 飽和度調整
    hlsCopy[:, :, 2] = (1 + saturation / 100.0) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 顏色空間反轉換 HLS -> BGR 
    result = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result = ((result * 255).astype(np.uint8))

    return result

def calculateAngle(output, p1, p2, p3):
    kpts = output[0, 7:].T
    
    coord = []
    n_kpts = len(kpts) // 3
    for i in range(n_kpts):
        x, y = kpts[3*i], kpts[3*i+1]
        conf = kpts[3*i+2]
        coord.append([i, x, y, conf])

    x1, y1 = coord[p1][1:3]
    x2, y2 = coord[p2][1:3]
    x3, y3 = coord[p3][1:3]

    angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))
    if angle<0: angle += 360
    if angle>180.0: angle = 360.0 - angle
    angle = round(angle, 2)
    return angle

def create_original_video(path, out_path, isWriteVideo):
    ### Select device
    device = select_device("0")

    ### Load model
    modelWeight = 'yolov7-w6-pose.pt'
    model = attempt_load(modelWeight, map_location=device)  # load FP32 model
    _ = model.eval()

    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    ### Define Video Writer
    if isWriteVideo:
        VIDEO_CODEC = "mp4v"
        OUT_WIDTH, OUT_HEIGHT = 1920, 1080
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (OUT_WIDTH, OUT_HEIGHT))

    frameIdx = 0
    for frame in tqdm(range(n_frames), total=n_frames):
        ret, image = cap.read()
        if ret == False:
            break
        
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        # image = preProcessing(image)

        ### Whole body keypoints detection
        output, result, coordinates, noPeople = extractKeypoints(device, model, image)
        
        if noPeople==False:
            frameIdx += 1 ### Count number of frame
            if isWriteVideo:
                video_writer.write(image)

    ### Close all windows
    if isWriteVideo:
        video_writer.release()
    cap.release()

def save_coord_and_angle_csv_file(video_path, coord_path, angle_path):
    ### Select device
    device = select_device("0")

    ### Load model
    modelWeight = 'yolov7-w6-pose.pt'
    model = attempt_load(modelWeight, map_location=device)  # load FP32 model
    _ = model.eval()

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    frameIdx = 0
    all_keypoints = []
    all_kneeAngle = []
    for frame in tqdm(range(n_frames), total=n_frames):
        ret, image = cap.read()
        if ret == False:
            break
        
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        image = preProcessing(image)

        ### Whole body keypoints detection
        output, result, coordinates, noPeople = extractKeypoints(device, model, image)
        
        if noPeople==False:
            frameIdx += 1 ### Count number of frame
            
            ### Keypoint (x, y) Coordinate data
            all_keypoints.append(coordinates) ### Save keypoints coordinate

            ### Knee Angle data
            l_angle = calculateAngle(output, 11, 13, 15)
            r_angle = calculateAngle(output, 12, 14, 16)
            all_kneeAngle.append([l_angle, r_angle])

    ### Save keypoints coordinate into csv file
    all_keypoints_array = np.array(all_keypoints)
    np.savetxt(coord_path, all_keypoints_array, delimiter=",")

    ### Save knee angle into csv file
    all_kneeAngle_array = np.array(all_kneeAngle)
    np.savetxt(angle_path, all_kneeAngle_array, delimiter=",")

    cap.release()


def cut_video(path, out_path, isWriteVideo, startIdx, endIdx):
    ### Select device
    device = select_device("0")

    ### Load model
    modelWeight = 'yolov7-w6-pose.pt'
    model = attempt_load(modelWeight, map_location=device)  # load FP32 model
    _ = model.eval()

    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    ### Define Video Writer
    if isWriteVideo:
        VIDEO_CODEC = "mp4v"
        OUT_WIDTH, OUT_HEIGHT = 1920, 1080
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (OUT_WIDTH, OUT_HEIGHT))

    frameIdx = 0
    for frame in tqdm(range(n_frames), total=n_frames):
        ret, image = cap.read()
        if ret == False:
            break
        
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        image = preProcessing(image)

        ### Whole body keypoints detection
        output, result, coordinates, noPeople = extractKeypoints(device, model, image)
        if noPeople==False: ### People detected
            if frameIdx>=startIdx and frameIdx<=endIdx and isWriteVideo==True:
                video_writer.write(image)
            frameIdx += 1 ### Count number of frame
    if isWriteVideo:
        video_writer.release()
    cap.release()

def cut_video_directly(path, out_path, isWriteVideo, startIdx, endIdx): ### update 2023/08/13
    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    ### Define Video Writer
    if isWriteVideo:
        VIDEO_CODEC = "mp4v"
        OUT_WIDTH, OUT_HEIGHT = 1920, 1080
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (OUT_WIDTH, OUT_HEIGHT))

    frameIdx = 0
    for frame in tqdm(range(n_frames), total=n_frames):
        ret, image = cap.read()
        if ret == False:
            break
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        if frameIdx>=startIdx and frameIdx<=endIdx and isWriteVideo==True:
            video_writer.write(image)
        frameIdx += 1 ### Count number of frame

    if isWriteVideo:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()

def flip_video(path, out_path):
    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    ### Define Video Writer
    VIDEO_CODEC = "mp4v"
    video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (frameWidth, frameHeight))

    for frame in tqdm(range(n_frames), total=n_frames):
        ret, image = cap.read()
        if ret == False:
            break
        image = cv2.flip(image, 1)
        video_writer.write(image)
    video_writer.release()
    cap.release()
    cv2.destroyAllWindows()


def execute_pose_estimation(path, out_path, csv_path, isWriteVideo):
    ### Select device
    device = select_device("0")

    ### Load model
    modelWeight = 'yolov7-w6-pose.pt'
    model = attempt_load(modelWeight, map_location=device)  # load FP32 model
    _ = model.eval()

    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth, frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)

    ### Define Video Writer
    if isWriteVideo:
        VIDEO_CODEC = "mp4v"
        OUT_WIDTH, OUT_HEIGHT = 1920, 1080
        video_writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, (OUT_WIDTH, OUT_HEIGHT))

    frameIdx = 0
    all_keypoints = []
    for frame in tqdm(range(n_frames), total=n_frames):
        ret, image = cap.read()
        if ret == False:
            break
        
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_NEAREST)
        image = preProcessing(image)

        ### Whole body keypoints detection
        output, result, coordinates, noPeople = extractKeypoints(device, model, image)
        
        if noPeople==False:
            frameIdx += 1 ### Count number of frame
            all_keypoints.append(coordinates) ### Save keypoints coordinate
            result = cv2.resize(result, (1920, 1080), interpolation=cv2.INTER_NEAREST)
            if isWriteVideo:
                video_writer.write(result)

    ### Save keypoints coordinate into csv file
    all_keypoints_array = np.array(all_keypoints)
    np.savetxt(csv_path, all_keypoints_array, delimiter=",")

    ### Close all windows
    if isWriteVideo:
        video_writer.release()
    cap.release()
    cv2.destroyAllWindows()




def main():
    DATAROOT = "../1_GaitAnalysis_ver1/data/"
    DATA_OUTPUT_ROOT= "../4_Cut_Video/data/"

    isCreateVideo = False
    isSaveCoordAngle = False

    isGetFrameIdx_Walking = False ### load the [Walking Subtask] frame index from CSV file
    isGetFrameIdx_SitStand = False ### load the [Sit-and-Stand Subtask] frame index from CSV file
    isGetFrameIdx_Turning = False ### load the [Turning Subtask] frame index from CSV file

    isCreateNewFolder = False ### Create new folder to save new data

    isCutVideo = False ### [Walking Subtask] create cut1 and cut2 videos
    isFlipVideo = False ### [Walking Subtask] create mirrored cut2 video
    isPoseEstimation = True ### [Walking Subtask] do pose estimation to visualize keypoints on video

    isWriteVideo = True

    ### ===== Define global variables ================================================
    allID_list = []
    pid_mapping_dict = {}

    walking_frameIdx_dict = {}
    walking_problem_list = []
    walking_pidList = []
    sitstand_frameIdx_dict = {}
    sitstand_problem_list = []
    sitstand_pidList = []
    turning_frameIdx_dict = {}
    turning_problem_list = []
    turning_pidList = []

    ### ===== Get all pid (string) in data folder =====================================
    for folderName in list(os.listdir(DATA_OUTPUT_ROOT)):
        if folderName!="50":
            allID_list.append(folderName)  

            pid_int = int(folderName)
            pid_str = folderName
            pid_mapping_dict[pid_int] = pid_str

    print(f'[Step1-1] allID_list: {len(allID_list)} subjects. \n{allID_list}\n')
    print(f'[Step1-2] pid_mapping_dict: {len(pid_mapping_dict)} subjects. \n{pid_mapping_dict}\n')

    ### ===== Create Valid video ======================================================
    if isCreateVideo==True: ### create the original video: 01_video.mp4
        for pid in allID_list:
            # if pid == "01":
            video_path = DATAROOT + pid + '/_yolo/' + pid + '_video.mp4'
            video_path_output = DATA_OUTPUT_ROOT + pid + '/' + pid + '_video.mp4'

            if (os.path.exists(video_path)==False):
                print(f'[pid {pid}] video file does NOT exist.')
            else:
                print(f'[Step1 Create Video] pid {pid}, {video_path}')
                create_original_video(video_path, video_path_output, isWriteVideo)

    ### ===== Create Coordinate CSV file and Angle CSV file ===========================
    if isSaveCoordAngle==True:
        for pid in allID_list:
            # if pid == "10":
            video_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_video.mp4'
            coord_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_video_keypoints1.csv'
            angle_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_video_kneeAngles.csv'

            if (os.path.exists(video_path)==False):
                print(f'[pid {pid}] video file does NOT exist.')
            else:
                print(f'[Step2 Coord & Angle] pid {pid}, {video_path}')
                save_coord_and_angle_csv_file(video_path, coord_path, angle_path)



    ### ===== load the [Walking Subtask] frame index from CSV file ====================
    if isGetFrameIdx_Walking==True: 
        for pid in allID_list:
            walking_pidList.append(pid)

        # walking_frameIdx_path = '../4_Cut_Video/walking_frame_index.csv' 
        # walking_frameIdx_path = '../4_Cut_Video/walking_frame_index_refine_w2_start.csv'  ### update 2023/08/13
        # walking_frameIdx_path = '../4_Cut_Video/walking_frame_index_refine_w2_end.csv'  ### update 2023/08/13
        walking_frameIdx_path = '../4_Cut_Video/frameIdx_w1_and_w2.csv'  ### update 2023/08/20

        if os.path.exists(walking_frameIdx_path):
            walking_frameIdx_df = pd.read_csv(walking_frameIdx_path)
            n_row, n_col = walking_frameIdx_df.shape[0], walking_frameIdx_df.shape[1]

            for i in range(n_row):
                pid = walking_frameIdx_df.loc[i, 'pid']
                pid = pid_mapping_dict[pid]
                start1 = walking_frameIdx_df.loc[i, 'start1_refine']   ### update 2023/08/13 ### walking_frameIdx_df.loc[i, 'start1']
                end1 = walking_frameIdx_df.loc[i, 'end1']
                period1 = walking_frameIdx_df.loc[i, 'period1_refine'] ### update 2023/08/13 ### walking_frameIdx_df.loc[i, 'period1']
                start2 = walking_frameIdx_df.loc[i, 'start2_refine']   ### update 2023/08/13 ### walking_frameIdx_df.loc[i, 'start2']
                end2 = walking_frameIdx_df.loc[i, 'end2_refine'] ### update 2023/08/13 ### walking_frameIdx_df.loc[i, 'end2'] 
                period2 = walking_frameIdx_df.loc[i, 'period2_refine'] ### update 2023/08/13 ### walking_frameIdx_df.loc[i, 'period2']
                
                if start1 != 'x': ### update 2023/08/13
                    walking_frameIdx_dict[pid] = {}
                    walking_frameIdx_dict[pid]['start1'] = start1
                    walking_frameIdx_dict[pid]['end1'] = end1
                    walking_frameIdx_dict[pid]['period1'] = period1
                    walking_frameIdx_dict[pid]['start2'] = start2
                    walking_frameIdx_dict[pid]['end2'] = end2
                    walking_frameIdx_dict[pid]['period2'] = period2
                else:
                    walking_problem_list.append(pid)
            
            print(f'[Step4-1 Walking] walking_frameIdx_dict: {len(walking_frameIdx_dict)} subjects.\n{walking_frameIdx_dict}\n')
            print(f'[Step4-1 Walking] Problem_list: {len(walking_problem_list)} subjects. {walking_problem_list}')
            for pid in walking_problem_list:
                walking_pidList.remove(pid)
                print(f' - Remove [pid {pid}] from walking_pidList.')
            print(f'\n[Step4-2 Walking] Update walking_pidList: {len(walking_pidList)} subjects.')
            print(f'[Step4-2 Total] allID_list: {len(allID_list)} subjects.\n')
        else:
            print(f'[Notice] No Walking Frame Index CSV file.')

    ### ===== load the [Sit-and-Stand Subtask] frame index from CSV file ==============
    if isGetFrameIdx_SitStand==True: 
        for pid in allID_list:
            sitstand_pidList.append(pid)

        # sitstand_frameIdx_path = '../4_Cut_Video/sit2stand_frame_index.csv'
        sitstand_frameIdx_path = '../4_Cut_Video/frameIdx_sit_and_stand_refine.csv'  ### update 2023/08/20

        if os.path.exists(sitstand_frameIdx_path):
            df = pd.read_csv(sitstand_frameIdx_path)
            n_row, n_col = df.shape[0], df.shape[1]

            for i in range(n_row):
                pid = df.loc[i, 'pid']
                pid = pid_mapping_dict[pid]
                start1 = df.loc[i, 'start1']   
                end1 = df.loc[i, 'end1']
                period1 = df.loc[i, 'period1'] 
                start2 = df.loc[i, 'start2']  
                end2 = df.loc[i, 'end2']  
                period2 = df.loc[i, 'period2'] 
                
                if start1 != 'x': 
                    sitstand_frameIdx_dict[pid] = {}
                    sitstand_frameIdx_dict[pid]['start1'] = start1
                    sitstand_frameIdx_dict[pid]['end1'] = end1
                    sitstand_frameIdx_dict[pid]['period1'] = period1
                    sitstand_frameIdx_dict[pid]['start2'] = start2
                    sitstand_frameIdx_dict[pid]['end2'] = end2
                    sitstand_frameIdx_dict[pid]['period2'] = period2
                else:
                    sitstand_problem_list.append(pid)

            print(f'[Step5-1 Sit-and-Stand] sitstand_frameIdx_dict: {len(sitstand_frameIdx_dict)} subjects.\n{sitstand_frameIdx_dict}\n')
            print(f'[Step5-1 Sit-and-Stand] Problem_list: {len(sitstand_problem_list)} subjects. {sitstand_problem_list}')
            for pid in sitstand_problem_list:
                sitstand_pidList.remove(pid)
                print(f' - Remove [pid {pid}] from sitstand_pidList.')
            print(f'[Step5-2 Sit-and-Stand] Update sitstand_pidList: {len(sitstand_pidList)} subjects.')
            print(f'[Step5-2 Total] allID_list: {len(allID_list)} subjects.\n')
        else:
            print(f'[Notice] No Sit-and-Stand Frame Index CSV file.')

    ### ===== load the [Turning Subtask] frame index from CSV file ====================
    if isGetFrameIdx_Turning==True: 
        for pid in allID_list:
            turning_pidList.append(pid)

        # turning_frameIdx_path = '../4_Cut_Video/turning1_frame_index.csv'
        turning_frameIdx_path = '../4_Cut_Video/frameIdx_t1_and_t2_refine.csv'  ### update 2023/08/20

        if os.path.exists(turning_frameIdx_path):
            df = pd.read_csv(turning_frameIdx_path)
            n_row, n_col = df.shape[0], df.shape[1]

            for i in range(n_row):
                pid = df.loc[i, 'pid']
                pid = pid_mapping_dict[pid]
                start1 = df.loc[i, 'start1']   
                end1 = df.loc[i, 'end1']
                period1 = df.loc[i, 'period1'] 
                start2 = df.loc[i, 'start2']  
                end2 = df.loc[i, 'end2']  
                period2 = df.loc[i, 'period2'] 
                
                if start1 != 'x': 
                    turning_frameIdx_dict[pid] = {}
                    turning_frameIdx_dict[pid]['start1'] = start1
                    turning_frameIdx_dict[pid]['end1'] = end1
                    turning_frameIdx_dict[pid]['period1'] = period1
                    turning_frameIdx_dict[pid]['start2'] = start2
                    turning_frameIdx_dict[pid]['end2'] = end2
                    turning_frameIdx_dict[pid]['period2'] = period2
                else:
                    turning_problem_list.append(pid)

            print(f'[Step6-1 Turning] turning_frameIdx_dict: {len(turning_frameIdx_dict)} subjects.\n{turning_frameIdx_dict}\n')
            print(f'[Step6-1 Turning] Problem_list: {len(turning_problem_list)} subjects. {turning_problem_list}')
            for pid in turning_problem_list:
                turning_pidList.remove(pid)
                print(f' - Remove [pid {pid}] from turning_pidList.')
            print(f'\n[Step6-2 Turning] Update turning_pidList: {len(turning_pidList)} subjects.')
            print(f'[Step6-2 Total] allID_list: {len(allID_list)} subjects.\n')
        else:
            print(f'[Notice] No Sit-and-Stand Frame Index CSV file.')
    
    ### ===== create new folder ======================================================
    if isCreateNewFolder==True:
        for pid in allID_list:
            # if pid == "01":
            folder_path = DATA_OUTPUT_ROOT + pid + '/Walking_0817/'
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            else:
                print(f'[pid {pid}] Folder [Walking_0817] has existed.')



    ### ===== create cut1 and cut2 videos ==============================================
    if isCutVideo==True:
        for pid in walking_pidList:
            # if pid == "01":
            # video_path = DATAROOT + pid + '/_yolo/' + pid + '_video.mp4' ### update 2023/08/13
            video_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_video.mp4'
            video_path_cut1 = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut1.mp4'
            video_path_cut2 = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut2.mp4'
            
            if (os.path.exists(video_path)==False):
                print(f'[pid {pid}] video file does NOT exist.')
            else:
                print(f'[Step7] pid {pid}, {video_path}')
                start1 = walking_frameIdx_dict[pid]['start1']
                end1 = walking_frameIdx_dict[pid]['end1']
                start2 = walking_frameIdx_dict[pid]['start2']
                end2 = walking_frameIdx_dict[pid]['end2']
                # cut_video(video_path, video_path_cut1, isWriteVideo, start1, end1) ### update 2023/08/13
                # cut_video(video_path, video_path_cut2, isWriteVideo, start2, end2) ### update 2023/08/13
                cut_video_directly(video_path, video_path_cut1, isWriteVideo, int(start1), int(end1))
                cut_video_directly(video_path, video_path_cut2, isWriteVideo, int(start2), int(end2))

    if isFlipVideo==True: ### create mirrored cut2 video ====================================================================
        for pid in allID_list:
            # if pid=="05" or pid=="06" or pid=="66":
            if pid > "01":
                # video_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_cut2.mp4' ### update 2023/08/13
                # video_path_mirrored = DATA_OUTPUT_ROOT + pid + '/' + pid + '_cut2_mirrored.mp4' ### update 2023/08/13
                video_path = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut2.mp4'
                video_path_mirrored = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut2_mirrored.mp4'

                if (os.path.exists(video_path)==False):
                    print(f'[pid {pid}] video file does NOT exist.')
                else:
                    print(f'[Step7-2] pid {pid}, {video_path}')
                    flip_video(video_path, video_path_mirrored)

    if isPoseEstimation==True: ### visualize keypoints on video & get coordinate CSV file
        for pid in allID_list:
            # if pid=="05" or pid=="06" or pid=="66":
            if pid > "01":
                # cut1_video_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_cut1.mp4' ### update 2023/08/13
                # cut1_video_path_output = DATA_OUTPUT_ROOT + pid + '/' + pid + '_cut1_output.mp4'
                # cut1_csv_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_cut1_keypoints1.csv'
                # cut2_mirrored_video_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_cut2_mirrored.mp4'
                # cut2_mirrored_video_path_output = DATA_OUTPUT_ROOT + pid + '/' + pid + '_cut2_mirrored_output.mp4'
                # cut2_csv_path = DATA_OUTPUT_ROOT + pid + '/' + pid + '_cut2_keypoints1.csv'

                cut1_video_path = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut1.mp4'
                cut1_video_path_output = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut1_output.mp4'
                cut1_csv_path = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut1_keypoints1.csv'
                cut2_mirrored_video_path = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut2_mirrored.mp4'
                cut2_mirrored_video_path_output = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut2_mirrored_output.mp4'
                cut2_csv_path = DATA_OUTPUT_ROOT + pid + '/Walking_0817/' + pid + '_cut2_keypoints1.csv'
                
                if (os.path.exists(cut1_video_path)==False) or (os.path.exists(cut2_mirrored_video_path)==False):
                    print(f'[pid {pid}] video file does NOT exist.')
                else:
                    print(f'[Step4: Pose Estimation] pid {pid}, {cut1_video_path}')
                    execute_pose_estimation(cut1_video_path, cut1_video_path_output, cut1_csv_path, isWriteVideo)
                    execute_pose_estimation(cut2_mirrored_video_path, cut2_mirrored_video_path_output, cut2_csv_path, isWriteVideo)


if __name__ == "__main__":
    main()