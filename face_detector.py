import cv2 as cv

# TODO Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# TODO Choose an image to detect face in
# img = cv.imread('photos/khanh.JPG')

# TODO Capture video from webcam
webcam = cv.VideoCapture(0)
while True:
    # Read current frame
    successful_frame_read, frame = webcam.read()
    # Must convert to Grayscale
    grayscale_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Coordinates of face detected
    face_coordinates = trained_face_data.detectMultiScale(grayscale_frame)
    for (x, y, w, h) in face_coordinates:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    cv.imshow('My webcam', frame)
    key = cv.waitKey(1)
    # TODO Recognition
    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break


# TODO Must convert to grayscale
grayscaled_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# TODO Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)

# TODO Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)


# TODO Display image with the face
cv.imshow('My face', img)
#
cv.waitKey()


print('Code completed')


