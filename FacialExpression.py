import numpy as np

class FacialExpression():
	def detect_expression(self, landmarks, width, height):
		right_corner	= np.array([landmarks.landmark[61].x * width, landmarks.landmark[61].y * height])
		left_corner		= np.array([landmarks.landmark[291].x * width, landmarks.landmark[291].y * height])
		upper_mouth		= np.array([landmarks.landmark[13].x * width, landmarks.landmark[13].y * height])
		lower_mouth		= np.array([landmarks.landmark[14].x * width, landmarks.landmark[14].y * height])
		right_eye		= np.array([landmarks.landmark[133].x * width, landmarks.landmark[133].y * height])
		left_eye		= np.array([landmarks.landmark[362].x * width, landmarks.landmark[362].y * height])
		
		mid_x = (landmarks.landmark[107].x * width + landmarks.landmark[55].x * width) / 2
		mid_y = (landmarks.landmark[107].y * height + landmarks.landmark[55].y * height) / 2
		right_brow_tip = np.array([mid_x, mid_y])
		mid_x = (landmarks.landmark[336].x * width + landmarks.landmark[285].x * width) / 2
		mid_y = (landmarks.landmark[336].y * height + landmarks.landmark[285].y * height) / 2
		left_brow_tip  = np.array([mid_x, mid_y])
		mid_x = (landmarks.landmark[70].x * width + landmarks.landmark[46].x * width) / 2
		mid_y = (landmarks.landmark[70].y * height + landmarks.landmark[46].y * height) / 2
		right_brow_end =np.array([mid_x, mid_y])
		mid_x = (landmarks.landmark[300].x * width + landmarks.landmark[276].x * width) / 2
		mid_y = (landmarks.landmark[300].y * height + landmarks.landmark[276].y * height) / 2
		left_brow_end  = np.array([mid_x, mid_y])

		#�ڂƔ��̋���
		dist_eye_brow = (np.linalg.norm(right_eye - right_brow_tip) + np.linalg.norm(left_eye - left_brow_tip)) / 2
		
		#���p
		mid_corner = (right_corner + left_corner) / 2
		dist_upper = np.linalg.norm(upper_mouth - mid_corner)
		dist_lower = np.linalg.norm(lower_mouth - mid_corner)
		
		#���̏c��
		height_mouth  = np.linalg.norm(upper_mouth - lower_mouth)
		
		#�ڊԊu
		dist_eye  = np.linalg.norm(right_eye - left_eye)
		
		#���Ԋu
		dist_brow_tip  = np.linalg.norm(right_brow_tip - left_brow_tip)
		
		#���p�x
		length_right_brow = np.linalg.norm(right_brow_tip - right_brow_end)
		length_left_brow  = np.linalg.norm(left_brow_tip - left_brow_end)
		cos = np.inner(length_right_brow, length_left_brow) / (length_right_brow * length_left_brow)
		

		#�\���
		#smile:1, angry:2, sad:3, surprised:4, other:5
		if dist_lower > 20:
			return 1
		if height_mouth > 40 and dist_eye_brow > 45:
			return 4
		else:
			return 5
