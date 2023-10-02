# Face Recognition Attendance System

A sophisticated face recognition system developed using OpenCV that identifies faces from a live video feed and logs their attendance into an Excel sheet.

## Key Features

- **Real-time Face Recognition**: The system can detect and recognize faces in real-time using a webcam feed.
- **Attendance Logging**: Recognized faces, along with their timestamps, are automatically logged into an Excel file.
- **Error Handling**: Unrecognized faces are conveniently labeled as "Unknown", ensuring no face goes unnoticed.
- **Optimized for Live Feeds**: The system is designed to perform efficiently with live video feeds.

## Prerequisites & Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Soham-Chaudhuri/Face-Recognition-Attendance.git
    cd Face-Recognition-Attendance
    ```

2. **Install Dependencies**:
    Before running the system, ensure Python is installed. Then, install the necessary libraries:
    ```bash
    pip install opencv-python openpyxl pandas
    ```

3. **Dataset Structure**:
    - Organize your dataset within a `dataset` directory at the root of the project.
    - Create individual subdirectories for each person you intend to recognize. The subdirectory's name will be the label for the person's face.
    - Populate each person's directory with clear images.

    Example Dataset Structure:

    ```
    /dataset
        /Alice
            alice1.jpg
            alice2.jpg
        /Bob
            bob1.jpg
    ```

## Running the System

1. **Train & Recognize**:
    Execute the script:
    ```bash
    python main.py
    ```

2. **Face Detection**:
    Present faces to the webcam. Known faces will be recognized, labeled, and their attendance marked in `attendance_record.xlsx`. Faces not in the dataset will be marked as "Unknown".

3. To exit the live feed, press the 'q' key.

## Potential Enhancements

- Integration with deep learning models for enhanced recognition accuracy.
- Real-time attendance sync with cloud-based databases or systems.
- Functionality to handle multiple simultaneous face detections efficiently.
