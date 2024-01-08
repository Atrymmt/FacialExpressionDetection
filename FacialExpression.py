import numpy as np

class FacialExpression():
	def detect_expression(self, landmarks, width, height):
		right_corner	= np.array([landmarks.landmark[61].x * width, landmarks.landmark[61].y * height])
		left_corner		= np.array([landmarks.landmark[291].x * width, landmarks.landmark[291].y * height])
		upper_mouth		= np.array([landmarks.landmark[13].x * width, landmarks.landmark[13].y * height])
		lower_mouth		= np.array([landmarks.landmark[14].x * width, landmarks.landmark[14].y * height])
		right_eye_tip	= np.array([landmarks.landmark[133].x * width, landmarks.landmark[133].y * height])
		left_eye_tip	= np.array([landmarks.landmark[362].x * width, landmarks.landmark[362].y * height])
		right_eye_end	= np.array([landmarks.landmark[33].x * width, landmarks.landmark[33].y * height])
		left_eye_end	= np.array([landmarks.landmark[263].x * width, landmarks.landmark[263].y * height])		

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

		#–Ú‚Æ”û‚Ì‹——£
		dist_eye_brow_tip = (np.linalg.norm(right_eye_tip - right_brow_tip) + np.linalg.norm(left_eye_tip - left_brow_tip)) / 2
		dist_eye_brow_end = (np.linalg.norm(right_eye_end - right_brow_end) + np.linalg.norm(left_eye_end - left_brow_end)) / 2

		#ŒûŠp
		mid_corner = (right_corner + left_corner) / 2
		dist_upper = np.linalg.norm(upper_mouth - mid_corner)
		dist_lower = np.linalg.norm(lower_mouth - mid_corner)
		
		#Œû‚Ìc•
		height_mouth  = np.linalg.norm(upper_mouth - lower_mouth)
		
		#–ÚŠÔŠu
		dist_eye  = np.linalg.norm(right_eye_tip - left_eye_tip)
		
		#”ûŠÔŠu
		dist_brow_tip  = np.linalg.norm(right_brow_tip - left_brow_tip)
		
		#”ûŠp“x
		length_right_brow = np.linalg.norm(right_brow_tip - right_brow_end)
		length_left_brow  = np.linalg.norm(left_brow_tip - left_brow_end)
		cos = np.inner(length_right_brow, length_left_brow) / (length_right_brow * length_left_brow)
		

		#•\î”»’è
		#smile:1, angry:2, sad:3, surprised:4, other:5

		if dist_lower > 25:			
			return 1
		elif dist_eye_brow_end > 40:
			if dist_eye_brow_tip > 40:
				return 4
		elif dist_eye_brow_tip < 25:
			return 2
		elif dist_eye_brow_tip > 35 or dist_brow_tip < 50:
			return 3
		else:
			return 5