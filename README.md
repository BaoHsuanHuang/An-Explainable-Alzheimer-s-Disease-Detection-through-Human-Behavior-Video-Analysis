# An Explainable Alzheimer's Disease Detection through Human Behavior Video Analysis

## Abstract
Alzheimer's disease (AD) is a progressive neurodegenerative disease characterized by cognitive and functional symptoms such as memory loss and balance problems. Current diagnostic methods, such as brain imaging and cognitive impairment questionnaires, are costly and time-consuming, making early detection challenging. In this work, we develop a computer vision-based method to detect AD using behavioral data collected from the Timed Up and Go (TUG) test and the Cookie Theft (CT) picture description task. y analyzing body joints and facial landmarks through 2D Convolutional Neural Networks (2D CNN) and Support Vector Machines (SVM), we classified subjects into AD and Non-AD categories across four subtasks: Walking, Sit-Stand, Turning, and Describing. Our approach achieved an F1-score of 0.90, demonstrating the potential of video-based analysis for AD detection. To enhance the reliability of our predictions, we apply model explanation methods, Gradient-weighted Class Attention Map (Grad-CAM) and SHapley Additive Explanation (SHAP) to identify key features and symptoms in the model's decision-making process.

## Dataset
### Subjects
![image](src/3_Dataset_subjects.png =90%x)

### Experimental Setup
1. Timed Up and Go (TUG) trial.
![3_Dataset_setting1](src/3_Dataset_setting1.png =40%x)

2. Cookie Theft picture description trial.
![3_Dataset_setting2](src/3_Dataset_setting2.png =40%x)

## Methodology

## Results

### Performance Evaluation


### Model Interpretation and Visualization

#### Walking Posture and Track


#### Head Movement


#### Turuning Period Importance

### Score Prediction Results