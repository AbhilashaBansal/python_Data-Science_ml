#Write a Python script to capture images from videostream, extract all faces (using haarcascades)
#Store the face information into a numpy arrays

# 1. Read the video stream, and capture images
# 2. Detect faces, and make a bounding box around them
# 3. Flatten and store information of the largest face by area, inside a numpy array
# 4. Repeat for multiple people to generate training data, for face recognition 

import numpy as np 
import cv2

#Init camera
cap = cv2.VideoCapture(0)

#Face Detection Object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = "./face-data/"
face_data = []
#we'll skip some frames
skip = 0 

file_name = input("Enter name of person getting scanned/ clicked: ")

while True:
	ret, frame = cap.read()

	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


	#Face Detection 
	faces = face_cascade.detectMultiScale(frame, 1.3, 5) #scaling factor & number of neighbours
	if len(faces)==0:
		continue

	faces = sorted(faces, key = lambda f: f[2]*f[3])

	#Pick the face with the largest area, to store, ie the last face
	face = faces[-1]
	x, y, w, h = face
	cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 255, 100), 2)

	#Extract/ crop out the required portion of face to store: REGION OF INTEREST
	offset = 15
	#in a frame, by default, first axis is the y axis, second is x axis
	face_section = frame[y-offset:y+h+offset, x-offset: x+w+offset]
	#must resize for standardization of data being stored & recognized
	face_section = cv2.resize(face_section, (100,100))

	skip += 1
	if skip%10==0:
		face_data.append(face_section)
		print(len(face_data))

	cv2.imshow("Complete Frame", frame)
	cv2.imshow("Face Section", face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('l'):
		break

	if len(face_data)>20:
		break

#end

#Convert list of face sections into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1)) #flattening the images, basically 


#Save this data into file system, as a npy file
np.save(dataset_path + file_name + '.npy', face_data)
print("Data saved succcessfully at " + dataset_path+file_name+'.npy')

#release the device
cap.release()
cv2.destroyAllWindows()








