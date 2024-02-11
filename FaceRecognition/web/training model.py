import os
import sys
import face_recognition
import pickle
from cv2 import cv2


# Create more accurate model of face (analyse more photos)
def training_by_img(name):

    if not os.path.exists("data/Lyubov"):
        print("[ERROR] There is not directory 'data'")
        sys.exit()

    data_encodings = []
    images = os.listdir("data/Lyubov")

#    print(images)

    for(i, image) in enumerate(images):
        print(f'[+] processing img {i + 1}/{len(images)}')
        print(image)

        face_img = face_recognition.load_image_file(f"data/Lyubov/{image}")
        face_encoding = face_recognition.face_encodings(face_img)[0]

        # print(face_encoding)

        if len(data_encodings) == 0:
            data_encodings.append(face_encoding)
        else:
            for item in range(0, len(data_encodings)):
                result = face_recognition.compare_faces([face_encoding], data_encodings[item])
                #print(result)

                if result[0]:
                    data_encodings.append(face_encoding)
                    #print("[INFO] Same person")
                    break
                else:
                    #print("[INFO] Another person")
                    break
    # print(data_encodings)
    # print(len(data_encodings))

    AcceptData = {
        "name": name,
        "encodings": data_encodings
    }

    with open (f"{name}_encodings.pickle","wb") as file:
        file.write(pickle.dumps(AcceptData))

    return f"[INFO] File {name}_encodings.pickle created"


# take screenshot in real time/video (every 3 sec + u can use hot key)
def take_screenshot():
    cap = cv2.VideoCapture("test_video.mp4")
    count = 0
    countHot = 0

    if not os.path.exists("dataset"):
        os.mkdir("dataset")

    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        multiplier = int(round(fps * 3))
        #print(multiplier)

        if ret:
            frame_id = int(round(cap.get(1)))
         #   print(frame_id)
            cv2.imshow("frame", frame)
            k = cv2.waitKey(20)

            if frame_id % multiplier == 0:
                cv2.imwrite(f'dataset/Screen{count}.jpg', frame)
                print(f'Take a screenshot {count}')
                count += 1

            if k == ord(" "):
                cv2.imwrite(f'dataset/ExtraScreen{countHot}.jpg', frame)
                countHot += 1
                print(f'Take an extra screen {countHot}')
            elif k == ord('q'):
                print('[INFO] Code was stopped')
                break


        else:
            print("[ERROR] Can't get screenshot")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print(training_by_img("Lyubov_encodings"))
        #take_screenshot()

if __name__ ==  "__main__":
    main()