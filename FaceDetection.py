import cv2
import mediapipe as mp
import numpy as np
from FacialExpression import FacialExpression

class FaceDetection():
    def main(self):
        self.face_detection()

    def face_detection(self):
        facial = FacialExpression() 
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        pre_deteciton = 0
        detectoin = 0
        count = 0 

        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        cap = cv2.VideoCapture(0)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
 
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                height, width, c = image.shape
                if results.multi_face_landmarks:
                    for index, landmarks in enumerate(results.multi_face_landmarks):
                        detectoin = facial.detect_expression(landmarks, width, height)
                        # cv2.circle(image, (right_corner[0],right_corner[1]), 5, (255,255,0))
                
                # cv2.putText(image, 
                #                 text='3', 
                #                 org=(100,300), 
                #                 fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #                 fontScale=10.0,
                #                 color=(0, 255, 0),
                #                 thickness=5,
                #                 lineType=cv2.LINE_4)
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
                
                if detectoin == pre_deteciton:
                    count += 1
                else:
                    count = 0
                
                pre_deteciton = detectoin
                if count < fps:
                    
                    continue

                if detectoin == 1:
                    print("smile")
                    exp = cv2.imread("./Resource/smile.png")
                elif detectoin == 2:
                    print("angry")
                    exp = cv2.imread("./Resource/angry.png")
                elif detectoin == 3:
                    print("sad")
                    exp = cv2.imread("./Resource/sad.png")
                elif detectoin == 4:
                    print("suprised")
                    exp = cv2.imread("./Resource/surprised.png")
                elif detectoin == 5:
                    print("normal")
                    exp = cv2.imread("./Resource/normal.png")
                cv2.imshow('Expression', exp)
                
                
        cap.release()

if __name__ == "__main__":
    fc = FaceDetection()
    fc.main()