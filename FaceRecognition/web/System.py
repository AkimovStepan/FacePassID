import face_recognition
import pickle
from cv2 import cv2
import os


def detect_person():
    count = 0
    countHot = 0
    if not os.path.exists("screenshots"):
        os.mkdir("screenshots")
    data = pickle.loads(open("encodings/Admin_encodings_encodings.pickle", "rb").read())
    capture = cv2.VideoCapture(0)

    while True:
        ret, image = capture.read()
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        for face_encoding, face_location in zip(encodings, locations):
            result = face_recognition.compare_faces(data["encodings"],face_encoding)
            match = None

            if True in result:
                match = data["name"]
                print(f'[INFO] Match found: {match}')
            else:
                print('[INFO] No matches')


            left_top = (face_location[3]), face_location[0]
            right_bottom = (face_location[1], face_location[2])
            color = [0, 200, 0]
            cv2.rectangle(image, left_top, right_bottom, color,5)

            left_bottom = (face_location[3],face_location[2])
            right_bottom = (face_location[1], face_location[2]+20)
            cv2.rectangle(image, left_bottom, right_bottom, color, cv2.FILLED)
            cv2.putText(
                image,
                match,
                (face_location[3] + 10, face_location[2] + 15),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (255, 255, 255),
                4
            )
            cv2.imshow("[INFO] Started detecting...", image)

        k = cv2.waitKey(1)
        if k == ord(" "):
            cv2.imwrite(f'screenshots/Screenshot{countHot}.jpg', image)
            countHot += 1
            print(f'[INFO] Screenshot{countHot} was done')

        if k == ord('q'):
            print('[INFO] Detecting was stopped')
            break


def main():
    detect_person()


if __name__ == '__main__':
    main()