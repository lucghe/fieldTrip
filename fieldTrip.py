# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 00:33:13 2021

@author: lucian @lucghe

Based on:
+ Faizan Shaikh's article https://www.analyticsvidhya.com/blog/2018/12/introduction-face-detection-video-deep-learning-python/
+ and face_recognition example by @ageitgey + @DEDZTBH on https://github.com/ageitgey/face_recognition/blob/master/examples/facerec_from_webcam_faster.py

Photo by cottonbro from Pexels https://www.pexels.com/photo/woman-in-red-sweater-beside-woman-in-green-coat-5711959/
Video by cottonbro from Pexels https://www.pexels.com/video/affectionate-young-couple-travelling-by-tram-5711968/
"""

# Measure time
import time
import datetime
start = time.time()

# Import OpenCV, face_recognition API
import cv2
import face_recognition

# Read the video and get the length
input_movie = cv2.VideoCapture("video_in.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Load sample image(s) of the subject to identify them in the video
image = face_recognition.load_image_file("photo_crop.jpg")
face_encoding = face_recognition.face_encodings(image)[0]
known_faces = [
face_encoding,
]

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

# Initialize video
codec = int(input_movie.get(cv2.CAP_PROP_FOURCC))
fps = int(input_movie.get(cv2.CAP_PROP_FPS))
frame_width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_movie = cv2.VideoWriter('pexels-c-technical-5711968_out.mp4', codec, fps, (frame_width, frame_height))

"""
Looping:
+ Extract a frame from the video
+ Find face(s) and identify them
+ Add frame to output video combining the original frame with the annotated location of the face(s)
"""
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1    

    # Quit when the input video file ends
    if not ret:
        break

    """
    Performance step: Only look for faces in frames where it is most likely to be detected
    Use countFrames.py and the rule of three (math) to figure out which frames to skip
    For clip pexels-c-technical-5711968_out.mp4 the first 9 seconds are skipped, meaning 226 frames out of 352
    """
    if frame_number < 226:
        print("Writing to output and skipping detection for frame {} / {}.".format(frame_number, length))
        output_movie.write(frame)
        continue

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    print("\nLooking for faces...")
    face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
    # Performance step: skip the rest if no face detected
    if not face_locations:
        print("No face(s) detected.")
        print("Writing to output and skipping other processing for frame {} / {}.\n".format(frame_number, length))
        output_movie.write(frame)
        continue
    
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:            
            name = "X"
            print("Face match for {} in frame {}.".format(name, frame_number))

        face_names.append(name)
    
    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            print("No face match in frame {}, location {}.".format(frame_number, (top, right, bottom, left)))
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    
    # Write the resulting image to the output video file
    print("Writing frame {} / {}.".format(frame_number, length))
    output_movie.write(frame)
    sofar = time.time()
    print("Time passed so far: {}.\n".format(str(datetime.timedelta(seconds=sofar-start))))
    
# Release!
input_movie.release()
output_movie.release()
cv2.destroyAllWindows()
end = time.time()
print("\nTotal time passed: {}.".format(str(datetime.timedelta(seconds=end-start))))