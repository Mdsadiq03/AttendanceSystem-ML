# AttendanceSystem-ML

Face Detection based attendance system
- Using HaarCascade Classifier
- tkinter for GUI
- openCV (Opensource Computer Vision)
## How it works :

### Launching the Application:

- Run train.py: This script initiates the application.
- Enter ID and Name: A window prompts you to input the ID & Name of the person.

### Capturing Images:

- Click 'Take Images':
The computer's camera opens.
60 image samples of the person are captured.
- Storage:
The ID and Name are stored in a file named StudentDetails.csv in the StudentDetails folder.
The captured images are saved in the TrainingImage folder.
Notification: The system notifies you once the images are saved.

### Training the Model:

- Click 'Train Image':
The system processes the captured images to train the model.
A Trainner.yml file is created and stored in the TrainingImageLabel folder.
- Duration: This process takes a few seconds.

### Tracking and Recognizing Faces:

- Click 'Track Image':
The camera opens again.
If a face is recognized, the system displays the ID and Name of the person on the screen.
- Exiting the Window:
Press Q or q to close the window.
Upon exit, the system logs the attendance of the recognized person.

### Attendance Logging:

- Storage: Attendance details, including Name, ID, Date, and Time, are saved in a CSV file within the Attendance folder.
- Display: The attendance record is also displayed on the window.
