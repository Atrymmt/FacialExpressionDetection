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
        detectoin = 0
 
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        cap = cv2.VideoCapture(0)
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
                
                
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                if detectoin == 1:
                    print("smile")
                elif detectoin == 4:
                    print("suprised")
                elif detectoin == 5:
                    print("normal")
                    
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()

if __name__ == "__main__":
    fc = FaceDetection()
    fc.main()