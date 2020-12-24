from face_detector import agr_recognition
import os
import cv2 as cv
from time import sleep


def turn_webcam(*agrs):
    webcam = cv.VideoCapture(0)
    while True:
        ret, frame = webcam.read()

        if not ret:
            print("Please check your webcam")
            break
        cv.imshow("AGR Register FaceID", cv.flip(frame, 1))

        cv.waitKey(1)


def webcam_capturing(employee_name, folder_dir):

    img_counter = 0
    on_off = True
    while on_off:
        turn_webcam(on_off)
        turn_webcam(on_off)
        # Capture and Save Frame as Image
        img_name = 'AGR_' + employee_name + '_{}.png'.format(img_counter)
        img_dir = os.path.join(folder_dir, img_name)
        cv.imwrite(img_dir, frame)
        print(img_name)
        img_counter += 1

def agr_register(name):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder_dir = os.path.join(BASE_DIR, 'faces_train', name)
    # os.mkdir(folder_dir)
    print(folder_dir)




turn_off = False

agr_register('Hung')
while not turn_off:
    option = input('''
    What do you want?
    1. Register AGR's employee.
    2. Recognition AGR's employee.
    // Press others to exit
    Choose number: ''')


    if option == '1':
        name = input("What is your name?: ")
        # agr_register()
    elif option == '2':
        agr_recognition()
    else:
        turn_off = True

