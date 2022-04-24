import cv2
import time
import threading
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials

subscription_key = "XXXXXXXXXXXXXXXXXXX"
endpoint = "https://facedetectiontask.cognitiveservices.azure.com/"
face_client = FaceClient(endpoint, CognitiveServicesCredentials(subscription_key))


class NearRealtimeFaceDetector(threading.Thread):
	"""
	NearRealtimeFaceDetector is a derived class from base class Thread.
	It detects Face in near real-time (after 1 second) using a video feed from webcam
	"""

	def __init__(self):
		threading.Thread.__init__(self)
		self.video_feed = cv2.VideoCapture(0)
		# left frame to show the webcam feed
		self.frame_1 = None
		# right frame to show the face detector output
		self.frame_2 = None

	def run(self):
		_, self.frame_1 = self.video_feed.read()
		self.frame_2 = cv2.flip(self.frame_1.copy(), 1)

		while True:
			_, frame = self.video_feed.read()
			frame = cv2.flip(frame, 1)
			# create a copy of the flipped frame
			self.frame_1 = frame.copy()
			# Left frame header to indicate that this is the Webcam Feed
			frame = cv2.putText(frame, "Webcam Feed", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
			# Create a side-by-side display
			frame_full = cv2.hconcat([frame, self.frame_2])
			# Display the feeds in a window
			cv2.imshow("Near Real-time Face detection", frame_full)
			# 1 millisecond delay to allow the window to load
			cv2.waitKey(1)
			# if the window is closed then break out of the infinite loop
			if cv2.getWindowProperty("Near Real-time Face detection", cv2.WND_PROP_VISIBLE) < 1:
				break
		cv2.destroyAllWindows()

	# Coordinates for the bounding box
	def get_rectangle_coordinates(self, face_dict):
		rect = face_dict.face_rectangle
		left = rect.left
		top = rect.top
		right = left + rect.width
		bottom = top + rect.height
		return left, top, right, bottom



	# Draws the Bounding Box and Creates Annotation labels under the Bounding Box to display the Face Attributes
	def draw_bounding_box_with_annotation_labels(self, frame, left, top, right, bottom, age, gender, emotion):
		# Draw the bounding box
		border_color = (198, 255, 0)
		frame = cv2.rectangle(frame, (left, top), (right, bottom + 100), border_color, 3)

		# Draw a filled rectangle for showing the output as Annotation labels
		frame = cv2.rectangle(frame, (left, bottom), (right, bottom + 100), border_color, cv2.FILLED)

		label_top = 20
		# Show the results on screen as labels under the bounding box
		frame = cv2.putText(frame, age, (left, bottom + label_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
							(0, 0, 0), 2, cv2.LINE_AA)
		label_top += 20
		frame = cv2.putText(frame, gender, (left, bottom + label_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
							(0, 0, 0), 2, cv2.LINE_AA)
		label_top += 20
		frame = cv2.putText(frame, "emotion: ", (left, bottom + label_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
							cv2.LINE_AA)
		label_top += 20
		# emotion value is moved to the next line to allow enough space
		frame = cv2.putText(frame, emotion, (left, bottom + label_top), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
							(0, 0, 0), 2, cv2.LINE_AA)
		return frame

	# Returns the age label text
	def age_label_text(self, age):
		return "age: " + str(int(age))

	# Returns the gender label text
	def gender_label_text(self, gender):
		gender = (gender.split('.'))[0]
		return "gender: " + str(gender)

	# Recognize the most prevalent emotion and return the name of the emotion
	# that the face detector has returned with most confidence
	def get_prevalent_emotion(self, emotion_details):
		emotions = emotion_details.__dict__
		emotion_keys = list(emotions.keys())[1:8]
		emotion_values = list(emotions.values())[1:8]
		highest_emotion_value = max(emotion_values)
		highest_emotion_index = emotion_values.index(highest_emotion_value)
		highest_emotion = emotion_keys[highest_emotion_index]
		return highest_emotion

	def detector(self):
		face_attributes = ['emotion', 'age', 'gender']
		while True:
			# Introduce a delay of 1 second
			time.sleep(1)
			# Create a copy of the frame
			frame = self.frame_1.copy()
			# Save the frame as an image
			cv2.imwrite('temp.jpg', frame)
			# Load the saved image to be streamed to face detection API
			local_image = open('temp.jpg', "rb")
			# Pass the image to the detector webservice
			faces = face_client.face.detect_with_stream(local_image, return_face_attributes=face_attributes, detection_model='detection_01')
			face_found = len(faces) > 0
			# check if any face has been detected
			if face_found:
				frame = cv2.putText(frame, "Face Detector", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
				single_face = faces[0]

				# Extract face attributes of the first face found
				age_label_value = self.age_label_text(single_face.face_attributes.age)
				gender_label_value = self.gender_label_text(single_face.face_attributes.gender)

				# Retrieve the most prevalent emotion from the face attributes
				prevalent_emotion = self.get_prevalent_emotion(faces[0].face_attributes.emotion)

				# Get the coordinates of the rectangular bounding box
				left, top, right, bottom = self.get_rectangle_coordinates(faces[0])
				frame = self.draw_bounding_box_with_annotation_labels(frame, left, top, right, bottom, age_label_value, gender_label_value, prevalent_emotion)

				# Refresh the frame_2 with the output
				self.frame_2 = frame


near_realtime_face_detector = NearRealtimeFaceDetector()
near_realtime_face_detector.start()
near_realtime_face_detector.detector()
