import sys

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import  KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

def rotate_3d_points(points, axis, degrees=180):
    radians = np.radians(degrees)
    rotation_matrices = {
        'x': np.array([[1, 0, 0], [0, np.cos(radians), -np.sin(radians)], [0, np.sin(radians), np.cos(radians)]]),
        'y': np.array([[np.cos(radians), 0, np.sin(radians)], [0, 1, 0], [-np.sin(radians), 0, np.cos(radians)]]),
        'z': np.array([[np.cos(radians), -np.sin(radians), 0], [np.sin(radians), np.cos(radians), 0], [0, 0, 1]])
    }
    rotated_points = np.dot(points, rotation_matrices[axis].T)
    return rotated_points

def rotateX(df):
    
    # Extract the 'feature' columns as a NumPy array
    original_landmarks = np.array(df.iloc[:, :249])  # Assuming the landmark features are the first 249 columns
    # Reshape the landmarks to have three dimensions (60402 samples, 83 landmarks, 3 coordinates)
    original_landmarks_reshaped = original_landmarks.reshape(-1, 83, 3)
    # Apply rotation to the x-axis
    x_rotated_landmarks = rotate_3d_points(original_landmarks_reshaped, axis='x')
    x_rotated_landmarks_reshaped = x_rotated_landmarks.reshape(original_landmarks.shape)
    # Create a new DataFrame with the x-rotated values
    x_rotated_df = pd.DataFrame()
    x_rotated_df = pd.DataFrame(x_rotated_landmarks_reshaped, columns=[f'feature_{i}' for i in range(249)])
    # Join 'label' column from the original DataFrame df
    x_rotated_df = pd.concat([x_rotated_df, df[['label']]], axis=1)
    return x_rotated_df

def rotateZ(df):
    
    # Extract the 'feature' columns as a NumPy array
    original_landmarks = np.array(df.iloc[:, :249])  # Assuming the landmark features are the first 249 columns
    # Reshape the landmarks to have three dimensions (60402 samples, 83 landmarks, 3 coordinates)
    original_landmarks_reshaped = original_landmarks.reshape(-1, 83, 3)
    z_rotated_landmarks = rotate_3d_points(original_landmarks_reshaped, axis='z')
    z_rotated_landmarks_reshaped = z_rotated_landmarks.reshape(original_landmarks.shape)
    # Create a new DataFrame with the z-rotated values
    z_rotated_df = pd.DataFrame()
    z_rotated_df = pd.DataFrame(z_rotated_landmarks_reshaped, columns=[f'feature_{i}' for i in range(249)])
    # Join 'label' column from the original DataFrame df
    z_rotated_df = pd.concat([z_rotated_df, df[['label']]], axis=1)
    return z_rotated_df


def rotateY(df):
    
    # Extract the 'feature' columns as a NumPy array
    original_landmarks = np.array(df.iloc[:, :249])  # Assuming the landmark features are the first 249 columns
    # Reshape the landmarks to have three dimensions (60402 samples, 83 landmarks, 3 coordinates)
    original_landmarks_reshaped = original_landmarks.reshape(-1, 83, 3)
    # Apply rotation to the y-axis
    y_rotated_landmarks = rotate_3d_points(original_landmarks_reshaped, axis='y')
    y_rotated_landmarks_reshaped = y_rotated_landmarks.reshape(original_landmarks.shape)
    # Create a new DataFrame with the y-rotated values
    y_rotated_df = pd.DataFrame()
    y_rotated_df = pd.DataFrame(y_rotated_landmarks_reshaped, columns=[f'feature_{i}' for i in range(249)])
    # Join 'label' columns from the original DataFrame df
    y_rotated_df = pd.concat([y_rotated_df, df[['label']]], axis=1)
    return y_rotated_df
          
def translate_to_origin(landmarks):
    # Calculate the average landmark for each coordinate (x, y, z)
    avg_landmark = np.mean(landmarks, axis=1)  # Use axis=1 to calculate mean across landmarks
    # Translate each landmark to the origin by subtracting the average
    translated_landmarks = landmarks - avg_landmark[:, np.newaxis, :]
    return translated_landmarks

def translateData(df):
    # Extract the 'feature' columns as a NumPy array
    original_landmarks = np.array(df.iloc[:, :249])  
    # Reshape the landmarks to have three dimensions (60402 samples, 83 landmarks, 3 coordinates)
    original_landmarks_reshaped = original_landmarks.reshape(-1, 83, 3)
    # Apply translation to the origin
    translated_landmarks = translate_to_origin(original_landmarks_reshaped)
    # Reshape the translated landmarks back to two dimensions (60402 samples, 249 features)
    translated_landmarks_reshaped = translated_landmarks.reshape(original_landmarks.shape)
    translated_landmarks_reshaped.shape

    # Create a new DataFrame with the translated values
    translated_columns = [f'feature_{i}' for i in range(249)]  # Create column names dynamically
    translated_df = pd.DataFrame(
        {col: translated_landmarks_reshaped[:, i] for i, col in enumerate(translated_columns)}
    )
    # Add 'label' column
    translated_df['label'] = df['label']
    return translated_df

def read_bnd_file(file_path):
    # Read the content of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    # Parse and extract the x, y, z coordinates
    landmarks = []
    for line in lines:  # Skip the header line
        if line.strip():  # Ignore empty lines
            values = line.split()
            x, y, z = float(values[1]), float(values[2]), float(values[3])
            landmarks.append([x, y, z])
    # Return a numpy array of 3D landmarks
    return np.array(landmarks)

def getprocessedata(datatype, df):
    if datatype == 'x':
        x_rotated_df = rotateX(df)
        x = x_rotated_df.iloc[:, :249]
        y = x_rotated_df.iloc[:, 249:250]
    elif datatype == 'y':
        y_rotated_df = rotateY(df)
        x = y_rotated_df.iloc[:, :249]
        y = y_rotated_df.iloc[:, 249:250]
    elif datatype == 'z':
        z_rotated_df = rotateZ(df)
        x = z_rotated_df.iloc[:, :249]
        y = z_rotated_df.iloc[:, 249:250]
    elif datatype == 'o':
        x = df.iloc[:, :249]
        y = df.iloc[:, 249:250]
    elif datatype == 't':
        translated_df = translateData(df)
        x = translated_df.iloc[:, :249]
        y = translated_df.iloc[:, 249:250]
    else:
        x = None
        y = None
    return x, y


def process_and_create_dataframe(data_directory):
    # Initialize lists to store data, labels, and genders
    data = []
    labels = []
    genders = []

    # Mapping for expression labels to integers
    expression_to_int = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3, 'Sad': 4, 'Surprise': 5}

    # Iterate through gender folders (F and M)
    for gender in ['F', 'M']:
        # Iterate through subject directories (1 to 58)
        for subject_id in range(1, 59):
            subject_directory = os.path.join(data_directory, f"{gender}{subject_id:03d}")

            # Check if the directory exists
            if os.path.exists(subject_directory):
                # Access the files within this subject directory
                # print(f"Processing {subject_directory}")

                # Iterate through expression folders
                for expression in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise']:
                    expression_directory = os.path.join(subject_directory, expression)

                    if os.path.exists(expression_directory):
                        #print(f"Processing {expression_directory}")

                        # Read BND files
                        for bnd_file_name in os.listdir(expression_directory):
                            if bnd_file_name.endswith(".bnd"):
                                bnd_file_path = os.path.join(expression_directory, bnd_file_name)
                                # Read and parse BND file
                                landmarks_data = read_bnd_file(bnd_file_path)

                                # Append data, label, and gender information
                                data.append(landmarks_data.flatten())
                                labels.append(expression_to_int[expression])
                                genders.append(gender)

    # Convert lists to NumPy arrays
    data = np.array(data)
    labels = np.array(labels)
    genders = np.array(genders)

    # Create a DataFrame
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(len(data[0]))])
    df['label'] = labels
    df['gender'] = genders

    return df

def getResult(conf_matrices, accuracies, precisions, recalls):
    
    # Calculate the sum of confusion matrices element-wise
    sum_cm = np.sum(conf_matrices, axis=0)

    # Calculate the total number of confusion matrices
    num_matrices = len(conf_matrices)

    # Calculating the average confusion matrix element-wise
    avg_cm = (sum_cm / num_matrices).astype(int)

    # Print average confusion matrix
    print("Average Confusion Matrix (Element-wise):")
    print(avg_cm)

    # Print average evaluation metrics over all folds
    print("Average Metrics:")
    print("Mean Accuracy:", np.mean(accuracies))
    print("Mean Precision:", np.mean(precisions))
    print("Mean Recall:", np.mean(recalls))

def trainwithSVM(x,y):
    # Initialize SVM Classifier
    svm_classifier = SVC()

    # Initialize 10-fold cross-validation
    kf = KFold(n_splits=10)
    # Initialize lists to store evaluation metrics for each fold
    conf_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    predicted_labels = []

    # Perform 10-fold cross-validation
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Fit classifier on training data
        svm_classifier.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = svm_classifier.predict(X_test)
        predicted_labels.append(y_pred)
        
        # Calculate evaluation metrics
        conf_matrices.append(confusion_matrix(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
    getResult(conf_matrices,accuracies,precisions,recalls)
    
def trainwithRF(x,y):
    # Initialize Random Forest Classifier
    rf_classifier = RandomForestClassifier()

    # Initialize 10-fold cross-validation
    kf = KFold(n_splits=10)

    # Initialize lists to store evaluation metrics for each fold
    conf_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    predicted_labels = []

    # Perform 10-fold cross-validation
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #print(X_train.shape)
        # Fit classifier on training data
        rf_classifier.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = rf_classifier.predict(X_test)
        predicted_labels.append(y_pred)
        
        # Calculate evaluation metrics
        conf_matrices.append(confusion_matrix(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
    getResult(conf_matrices,accuracies,precisions,recalls)
def trainwithDT(x,y):
    
    # Initialize Classifier
    classifier = DecisionTreeClassifier()


    # Initialize 10-fold cross-validation
    kf = KFold(n_splits=10)
    # Initialize lists to store evaluation metrics for each fold
    conf_matrices = []
    accuracies = []
    precisions = []
    recalls = []
    predicted_labels = []

    # Perform 10-fold cross-validation
    for train_index, test_index in kf.split(x):
        X_train, X_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Fit classifier on training data
        classifier.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = classifier.predict(X_test)
        predicted_labels.append(y_pred)
        
        # Calculate evaluation metrics
        conf_matrices.append(confusion_matrix(y_test, y_pred))
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average='weighted'))
        recalls.append(recall_score(y_test, y_pred, average='weighted'))
    getResult(conf_matrices,accuracies,precisions,recalls)


def scatter_plot(df):
    # Select random sample
    random_samples = np.random.choice(len(df), 1, replace=False)

    # Plot original translated and rotated landmarks in 3D for the selected sample
    for i, sample_index in enumerate(random_samples):
        # Extract original and rotated landmarks for the selected sample
        translated_df=translateData(df)
        x_rotated_df=rotateX(df)
        y_rotated_df=rotateY(df)
        z_rotated_df=rotateZ(df)
        original_landmarks = np.array(df.iloc[sample_index, :249]).reshape(83, 3)
        translated_landmarks = np.array(translated_df.iloc[sample_index, :249]).reshape(83, 3)
        x_rotated_landmarks = np.array(x_rotated_df.iloc[sample_index, :249]).reshape(83, 3)
        y_rotated_landmarks = np.array(y_rotated_df.iloc[sample_index, :249]).reshape(83, 3)
        z_rotated_landmarks = np.array(z_rotated_df.iloc[sample_index, :249]).reshape(83, 3)

        # Create 3D subplots for sample
        fig = plt.figure(figsize=(20, 5))

        # Plot original landmarks
        ax1 = fig.add_subplot(151, projection='3d')
        ax1.scatter(original_landmarks[:, 0], original_landmarks[:, 1], original_landmarks[:, 2], c='blue', marker='o')
        ax1.set_title(f'Sample {sample_index + 1}: Original')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.set_zlabel('Z-axis')
        
        # Plot translated landmarks
        ax1 = fig.add_subplot(152, projection='3d')
        ax1.scatter(translated_landmarks[:, 0], translated_landmarks[:, 1], translated_landmarks[:, 2], c='brown', marker='o')
        ax1.set_title(f'Sample {sample_index + 1}: Translated')
        ax1.set_xlabel('X-axis')
        ax1.set_ylabel('Y-axis')
        ax1.set_zlabel('Z-axis')
        
        # Plot x_rotated landmarks
        ax2 = fig.add_subplot(153, projection='3d')
        ax2.scatter(x_rotated_landmarks[:, 0], x_rotated_landmarks[:, 1], x_rotated_landmarks[:, 2], c='black', marker='o')
        ax2.set_title(f'Sample {sample_index + 1}: x_rotated')
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.set_zlabel('Z-axis')

        # Plot y_rotated landmarks
        ax3 = fig.add_subplot(154, projection='3d')
        ax3.scatter(y_rotated_landmarks[:, 0], y_rotated_landmarks[:, 1], y_rotated_landmarks[:, 2], c='green', marker='o')
        ax3.set_title(f'Sample {sample_index + 1}: y_rotated')
        ax3.set_xlabel('X-axis')
        ax3.set_ylabel('Y-axis')
        ax3.set_zlabel('Z-axis')

        # Plot z_rotated landmarks
        ax4 = fig.add_subplot(155, projection='3d')
        ax4.scatter(z_rotated_landmarks[:, 0], z_rotated_landmarks[:, 1], z_rotated_landmarks[:, 2], c='red', marker='o')
        ax4.set_title(f'Sample {sample_index + 1}: z_rotated')
        ax4.set_xlabel('X-axis')
        ax4.set_ylabel('Y-axis')
        ax4.set_zlabel('Z-axis')

        # Adjust layout for better visualization
        #plt.tight_layout()
        plt.show()
    
if __name__=='__main__':
    if len(sys.argv) != 4:
        print("Usage: python filename.py <arg1> <arg2> <arg3>")
        sys.exit(1)

    # Extract the input from the command line arguments
    algo =  sys.argv[1]
    # 'DT'
   
    datatype = sys.argv[2]
    # 'z'
    # 
    dataset = sys.argv[3]
    # 'BU4DFE_BND_V1.1'
    # 
    # <file name [RF,svm,DT] [x,y,z,o,t] dataset. 
    #    0         1           2           3
    # df=readtheData(data_directory=dataset)
    df=process_and_create_dataframe(dataset)
    # #print(df)
    x_data,y_data=getprocessedata(datatype,df)
    if algo=='RF':
        trainwithRF(x_data,y_data.values.ravel())
    if algo=='SVM':
        trainwithSVM(x_data,y_data.values.ravel())
    if algo=='DT':
        trainwithDT(x_data,y_data.values.ravel())    
    scatter_plot(df)
    
    
    
    