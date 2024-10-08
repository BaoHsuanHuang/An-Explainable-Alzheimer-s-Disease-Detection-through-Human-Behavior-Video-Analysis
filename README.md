# An Explainable Alzheimer's Disease Detection through Human Behavior Video Analysis

## Abstract
Alzheimer's disease (AD) is a progressive neurodegenerative disease characterized by cognitive and functional symptoms such as memory loss and balance problems. Current diagnostic methods, such as brain imaging and cognitive impairment questionnaires, are costly and time-consuming, making early detection challenging. In this work, we develop a computer vision-based method to detect AD using behavioral data collected from the Timed Up and Go (TUG) test and the Cookie Theft (CT) picture description task. y analyzing body joints and facial landmarks through 2D Convolutional Neural Networks (2D CNN) and Support Vector Machines (SVM), we classified subjects into AD and Non-AD categories across four subtasks: Walking, Sit-Stand, Turning, and Describing. Our approach achieved an F1-score of 0.90, demonstrating the potential of video-based analysis for AD detection. To enhance the reliability of our predictions, we apply model explanation methods, Gradient-weighted Class Attention Map (Grad-CAM) and SHapley Additive Explanation (SHAP) to identify key features and symptoms in the model's decision-making process.

## Dataset
### Subjects
<img src="src/3_Dataset_subjects.png" alt="3_Dataset_subjects" style="width:70%;"/>

### Experimental Setup
1. Timed Up and Go (TUG) trial.
<img src="src/3_Dataset_setting1.png" alt="3_Dataset_setting1" style="width:30%;"/>
2. Cookie Theft picture description trial.
<img src="src/3_Dataset_setting2.png" alt="3_Dataset_setting2" style="width:30%;"/>

## Methodology
### Data Pre-processing
1. Body Landmarks Detection: YOLOv7 Pose Estimation
<img src="src/4_Method_body_landmarks.jpg" alt="4_Method_body_landmarks" style="width:40%;"/>

2. Facial Landmarks Detection: MediaPipe Face Mesh Detection
<img src="src/4_Method_facial_landmarks.jpg" alt="4_Method_facial_landmarks" style="width:35%;"/>

### Modeling Overview
<img src="src/4_Method_model_overview.png" alt="4_Method_model_overview" style="width:90%;"/>


## Results
### Performance Evaluation
Evaluation results of ACC (Accuracy), SEN (Sensitivity), SPE (Specificity), PRE (Precision), and F1-score. The values in bold represent the best performance for each evaluation metric, while the underlined values indicate the second-best performance.

<img src="src/5_Results_table_complete.png" alt="5_Results_table_complete" style="width:80%;"/>

### Model Interpretation and Visualization

#### Walking Posture and Track
Model interpretation results on a figure of distance changes in the x-direction between the left and right ankles during walking with Gradient-weighted Class Activation Mapping (Grad-CAM) method. Non-AD subjects walk faster and have longer stride lengths, while AD patients have slower walking paces and shorter stride lengths. In addition, the
distance difference of each step taken by AD patients is relatively unstable and marked with darker red dots, thus showing the abnormal characteristics of AD patients’ gait.
<img src="src/5_Results_walking_tracks.png" alt="5_Results_walking_tracks" style="width:90%;"/>

#### Head Movement
We visualized the model interpretation results in the subjects' head movement trajectories while performing the picture description task.
The findings indicate that individuals with AD tend to exhibit head movement during the experiment, and the model interpretation results also consider the trajectory of significant head movements as an important feature of the model decision-making process (marked by dark red lines). The movement of the head also includes the action that subjects turn their head to seek help or recognition from medical staff.
<img src="src/5_Results_head_movement.png" alt="5_Results_head_movement" style="width:90%;"/>

#### Turuning Period Importance
In the Turning Subtask, we apply SHAP on the test data for each fold to understand how the model behaves on the data separately in a five-fold crossvalidation. 
We can figure out that the range of the SHAP value of feature "period 1' along the x-axis is larger than that of the feature 'period 2', which means that feature 'period 1' is more important for the final predictions of the model. Therefore, we can conclude that the longer it takes a subject to perform the first turning action, the more likely that subject is to exhibit signs of AD.
<img src="src/5_Results_turning_SHAP.png" alt="5_Results_turning_SHAP" style="width:80%;"/>
