import csv
import datetime
import time
import tkinter as tk

import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image


def clear():
    """
    Clears the entry field for ID and resets the notification message.
    """
    # Clear the text entry field for ID
    txt.delete(0, 'end')
    # Reset the notification message
    res = ""
    message.configure(text=res)


def clear2():
    """
    Clears the entry field for Name and resets the notification message.
    """
    # Clear the text entry field for Name
    txt2.delete(0, 'end')
    # Reset the notification message
    res = ""
    message.configure(text=res)


def is_number(s):
    """
    Checks if the provided string can be converted to a number.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string can be converted to a number, otherwise False.
    """
    try:
        # Attempt to convert the string to a float
        float(s)
        return True
    except ValueError:
        pass

    try:
        # Attempt to get the numeric value of the string (handles unicode numbers)
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def takeimages():
    """
    Captures face images from the webcam, saves them to a specified directory, and updates student details in a CSV file.
        """
    # Get the ID and name entered by the user
    global result_message
    student_id = txt.get()
    student_name = txt2.get()

    # Check if the ID is numeric and the name is alphabetic
    if is_number(student_id) and student_name.isalpha():
        # Initialize the webcam
        cam = cv2.VideoCapture(0)
        # Path to the Haar Cascade classifier for face detection
        haarcascade_path = "haarcascade_frontalface_default.xml"
        # Create a face detector using the Haar Cascade classifier
        detector = cv2.CascadeClassifier(haarcascade_path)

        sample_num = 0

        while True:
            # Capture a frame from the webcam
            ret, img = cam.read()
            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect faces in the image
            faces = detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Draw a rectangle around the detected face
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Increment the sample number
                sample_num += 1
                # Save the captured face image
                filename = f"TrainingImage/{student_name}.{student_id}.{sample_num}.jpg"
                cv2.imwrite(filename, img[y:y + h, x:x + w])
                # Display the frame with detected faces
                cv2.imshow('frame', img)

            # Wait for 100 milliseconds and check if 'q' is pressed
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # Break the loop if the sample number exceeds 60
            elif sample_num > 60:
                break

        # Release the webcam and close all OpenCV windows
        cam.release()
        cv2.destroyAllWindows()

        # Update the message label with the result
        result_message = f"Images Saved for ID: {student_id} Name: {student_name}"
        message.configure(text=result_message)

        # Append the student details to the CSV file
        student_row = [student_id, student_name]
        with open('StudentDetails/StudentDetails.csv', 'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(student_row)
    else:
        # Display error messages if the input is invalid
        if is_number(student_id):
            result_message = "Enter Alphabetical Name"
        if student_name.isalpha():
            result_message = "Enter Numeric ID"
        message.configure(text=result_message)


def train_images():
    """
        Trains the face recognizer using the images stored in the 'TrainingImage' directory and saves the trained model.
        """
    # Create an instance of the LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    # Path to the Haar Cascade classifier for face detection
    haarcascade_path = "haarcascade_frontalface_default.xml"
    # Create a face detector using the Haar Cascade classifier
    detector = cv2.CascadeClassifier(haarcascade_path)

    # Retrieve images and their corresponding labels
    faces, ids = getImagesAndLabels("TrainingImage")

    # Train the recognizer with the face images and their labels
    recognizer.train(faces, np.array(ids))

    # Save the trained model to a file
    recognizer.save("TrainingImageLabel/Trainner.yml")

    # Update the message label to indicate that the training is complete
    result_message = "Image Trained"
    message.configure(text=result_message)


def getImagesAndLabels(path):
    """
        Retrieves images and their corresponding labels from the specified path for training.

        Args:
            path (str): The directory path where the images are stored.

        Returns:
            tuple: A tuple containing two lists:
                - faces: A list of numpy arrays representing the grayscale face images.
                - ids: A list of integers representing the IDs associated with each face image.
        """
    # Get the list of all image file paths in the directory
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    # Create an empty list to store the face images
    faces = []
    # Create an empty list to store the corresponding IDs
    ids = []

    # Loop through each image path
    for image_path in image_paths:
        # Load the image and convert it to grayscale
        pil_image = Image.open(image_path).convert('L')

        # Convert the grayscale PIL image to a numpy array
        image_np = np.array(pil_image, dtype=np.uint8)

        # Extract the ID from the image filename
        image_id = int(os.path.split(image_path)[-1].split(".")[1])

        # Append the numpy array of the face image to the faces list
        faces.append(image_np)
        # Append the ID to the ids list
        ids.append(image_id)

    return faces, ids


def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(r"TrainingImageLabel\Trainner.yml")

    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    try:
        df = pd.read_csv(r"StudentDetails\StudentDetails.csv", header=None, names=['Id', 'Name'])
    except FileNotFoundError:
        print("Error: The StudentDetails.csv file was not found.")
        return
    except pd.errors.EmptyDataError:
        print("Error: The StudentDetails.csv file is empty.")
        return

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    try:
        while True:
            ret, im = cam.read()
            if not ret:
                print("Error: Failed to capture image from camera.")
                break

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
                Id, conf = recognizer.predict(gray[y:y + h, x:x + w])

                if conf < 50:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

                    aa = df.loc[df['Id'] == Id]['Name'].values
                    if len(aa) > 0:
                        name = str(aa[0])
                    else:
                        name = 'Unknown'

                    tt = f"{Id}-{name}"
                    attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                else:
                    Id = 'Unknown'
                    tt = str(Id)

                if conf > 75:
                    if not os.path.exists("ImagesUnknown"):
                        os.makedirs("ImagesUnknown")
                    noOfFile = len(os.listdir("ImagesUnknown")) + 1
                    cv2.imwrite(f"ImagesUnknown/Image{noOfFile}.jpg", im[y:y + h, x:x + w])

                cv2.putText(im, tt, (x, y + h), font, 1, (255, 255, 255), 2)

            attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
            cv2.imshow('im', im)

            if cv2.waitKey(1) == ord('q'):
                break

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        Hour, Minute, Second = timeStamp.split(":")
        fileName = f"Attendance/Attendance_{date}_{Hour}-{Minute}-{Second}.csv"
        attendance.to_csv(fileName, index=False)

    finally:
        cam.release()
        cv2.destroyAllWindows()

    res = attendance
    message2.configure(text=res)


# Create the main window
window = tk.Tk()
window.title("Face Recogniser")

# Configure window appearance
window.configure(background='#F0F0F0')  # Light neutral background color
window.geometry('1200x800')  # Set a specific size for the window

# Create and place widgets
message = tk.Label(window,
                   text="Face Recognition Attendance System",
                   bg="#4B0082",  # Indigo background for a royal look
                   fg="White",
                   width=50,
                   height=3,
                   font=('Times New Roman', 24, 'bold'),
                   relief="raised",  # Raised border
                   bd=2)  # Border width
message.place(x=150, y=20)

lbl = tk.Label(window,
               text="Enter ID",
               width=20,
               height=2,
               fg="Black",
               bg="#DCDCDC",  # Light gray for contrast
               font=('Times New Roman', 14, 'bold'),
               relief="solid",
               bd=1)  # Border width
lbl.place(x=400, y=150)

txt = tk.Entry(window,
               width=20,
               bg="White",
               fg="Black",
               font=('Times New Roman', 14),
               relief="solid",
               bd=1)  # Border width
txt.place(x=650, y=165)

lbl2 = tk.Label(window,
                text="Enter Name",
                width=20,
                fg="Black",
                bg="#DCDCDC",  # Light gray for contrast
                height=2,
                font=('Times New Roman', 14, 'bold'),
                relief="solid",
                bd=1)  # Border width
lbl2.place(x=400, y=250)

txt2 = tk.Entry(window,
                width=20,
                bg="White",
                fg="Black",
                font=('Times New Roman', 14),
                relief="solid",
                bd=1)  # Border width
txt2.place(x=650, y=265)

lbl3 = tk.Label(window,
                text="Notification:",
                width=20,
                fg="Black",
                bg="#DCDCDC",  # Light gray for contrast
                height=2,
                font=('Times New Roman', 14, 'bold'),
                relief="solid",
                bd=1)  # Border width
lbl3.place(x=400, y=350)

message = tk.Label(window,
                   text="",
                   bg="White",
                   fg="#FF4500",  # Orange red for emphasis
                   width=30,
                   height=2,
                   font=('Times New Roman', 14),
                   relief="solid",
                   bd=1)  # Border width
message.place(x=650, y=350)

lbl4 = tk.Label(window,
                text="Attendance:",
                width=20,
                fg="Black",
                bg="#DCDCDC",  # Light gray for contrast
                height=2,
                font=('Times New Roman', 14, 'bold'),
                relief="solid",
                bd=1)  # Border width
lbl4.place(x=400, y=450)

message2 = tk.Label(window,
                    text="",
                    fg="#FF4500",  # Orange red for emphasis
                    bg="White",
                    width=30,
                    height=2,
                    font=('Times New Roman', 14),
                    relief="solid",
                    bd=1)  # Border width
message2.place(x=650, y=450)

# Create and place buttons
clearButton = tk.Button(window,
                        text="Clear ID",
                        command=clear,
                        fg="White",
                        bg="#4B0082",  # Indigo for a royal look
                        width=20,
                        height=2,
                        font=('Times New Roman', 14, 'bold'),
                        relief="raised",
                        bd=2)  # Border width
clearButton.place(x=950, y=150)

clearButton2 = tk.Button(window,
                         text="Clear Name",
                         command=clear2,
                         fg="White",
                         bg="#4B0082",  # Indigo for a royal look
                         width=20,
                         height=2,
                         font=('Times New Roman', 14, 'bold'),
                         relief="raised",
                         bd=2)  # Border width
clearButton2.place(x=950, y=250)

takeImg = tk.Button(window,
                    text="Take Images",
                    command=takeimages,
                    fg="White",
                    bg="#6A5ACD",  # Slate Blue for a regal touch
                    width=20,
                    height=3,
                    font=('Times New Roman', 14, 'bold'),
                    relief="raised",
                    bd=2)  # Border width
takeImg.place(x=200, y=550)

trainImg = tk.Button(window,
                     text="Train Images",
                     command=train_images,
                     fg="White",
                     bg="#6A5ACD",  # Slate Blue for a regal touch
                     width=20,
                     height=3,
                     font=('Times New Roman', 14, 'bold'),
                     relief="raised",
                     bd=2)  # Border width
trainImg.place(x=500, y=550)

trackImg = tk.Button(window,
                     text="Track Images",
                     command=TrackImages,
                     fg="White",
                     bg="#6A5ACD",  # Slate Blue for a regal touch
                     width=20,
                     height=3,
                     font=('Times New Roman', 14, 'bold'),
                     relief="raised",
                     bd=2)  # Border width
trackImg.place(x=800, y=550)

quitWindow = tk.Button(window,
                       text="Quit",
                       command=window.destroy,
                       fg="White",
                       bg="#B22222",  # Firebrick for a strong finish
                       width=20,
                       height=3,
                       font=('Times New Roman', 14, 'bold'),
                       relief="raised",
                       bd=2)  # Border width
quitWindow.place(x=1100, y=550)

# Developer credit text
copyWrite = tk.Text(window,
                    background=window.cget("background"),
                    borderwidth=0,
                    font=('Times New Roman', 16, 'italic'),
                    fg="#808080")  # Indigo for a royal look
copyWrite.insert("insert", "Developed by Sadiq")
copyWrite.configure(state="disabled")
copyWrite.pack(side="left")
copyWrite.place(x=1180, y=680)

# Start the main loop
window.mainloop()



# Configure window appearance
# window.configure(background='Lavender Blush')
# window.grid_rowconfigure(0, weight=1)
# window.grid_columnconfigure(0, weight=1)

# Optional: Fullscreen and geometry configurations
# window.geometry('1280x720')
# window.attributes('-fullscreen', True)

# Optional: Display an image
# path = "profile.jpg"
# img = ImageTk.PhotoImage(Image.open(path))
# panel = tk.Label(window, image=img)
# panel.pack(side="left", fill="y", expand="no")

# Optional: Display an image using Canvas
# cv_img = cv2.imread("img541.jpg")
# x, y, no_channels = cv_img.shape
# canvas = tk.Canvas(window, width=x, height=y)
# canvas.pack(side="left")
# photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))
# canvas.create_image(0, 0, image=photo, anchor=tk.NW)

# Optional: Display a message
# msg = Message(window, text='Hello, world!')
