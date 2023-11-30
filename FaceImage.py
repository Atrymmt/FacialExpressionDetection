import mediapipe as mp
import cv2 as cv

mp_drawing = mp.solutions.drawing_utils 
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

image = cv.imread('Resource/myface.jpg') 
rgb_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
results = face_mesh.process(rgb_image)

annotated_image = image.copy()

for face_landmarks in results.multi_face_landmarks:
  print(face_landmarks)
  ###################################
  # landmark {
  #  x: 0.4557078182697296
  #  y: 0.6814221143722534
  #  z: -0.022573839873075485
  # }
  # ...
  # landmark {
  #  x: 0.5611855983734131
  #  y: 0.48328155279159546
  #  z: 0.0023858787026256323
  # }
  ###################################
  mp_drawing.draw_landmarks(
    image=annotated_image,
    landmark_list=face_landmarks,
    connections=mp_face_mesh.FACEMESH_TESSELATION,
    landmark_drawing_spec=drawing_spec,
    connection_drawing_spec=drawing_spec)
cv.imwrite('Resource/face_mesh_result.png', annotated_image) 
face_mesh.close()