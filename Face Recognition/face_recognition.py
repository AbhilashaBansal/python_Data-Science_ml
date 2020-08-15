#STEPS
# 1. Load the training data (numpy arrays for each person)
# 2. Assign y-labels to each person 
# 3. Read a videostream, extract all faces
# 4. Use KNN to predict the face - label
# 5. Map label with name of the person 
# 6. Display prediction + bounding box on faces

import cv2
import os
import numpy as np 

####################################################

def distance(v1, v2): 
	#Eucledian
	return np.sqrt(sum((v1-v2)**2))

def KNN(train, test, k=5):
	dist = []

	for i in range(train.shape[0]):
		#get the vector & the label 
		ix = train[i, :-1] #vector
		iy = train[i, -1] #label
		#compute distance from test point
		d = distance(test, ix)
		dist.append((d, iy)) #append distance as well as label

	#sort based on distance & retain the top k-nearest pts
	dk = sorted(dist, key = lambda x: x[0])[:k]

	#retrieve the labels
	labels = np.array(dk)[:, 1]

	#get frequencies of label
	labels = np.unique(labels, return_counts = True)
	max_freq_label_index = np.argmax(labels[1])

	return labels[0][max_freq_label_index]

####################################################

#init camera
cap = cv2.VideoCapture(0)

#face detection object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

counter = 0
dataset_path = "./face-data/"
face_data = []

class_id = 0 #labels for the given file
labels = []
names = {} #mapping id with name


#Data Preparation 
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#create mapping b/w class_id and name
		names[class_id] = fx[:-4] #removing .npy
		print("Loaded " + fx)
		data_item = np.load(dataset_path + fx)
		face_data.append(data_item)


		#create labels for the class 
		target = class_id*np.ones((data_item.shape[0],))
		labels.append(target)
		class_id += 1

print(len(face_data))
print(len(labels))

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

#attaching the labels along with flattened images, to prep data for KNN
trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

#Testing
while True:
	ret, frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame, 1.3, 5)
	if(len(faces) == 0):
		continue

	for face in faces:
		x,y,w,h = face

		#get the REGION OF INTEREST
		offset = 15
		face_section = frame[y-offset: y+offset+h, x-offset: x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		#prediction label (output)
		out = KNN(trainset, face_section.flatten())

		#Display on screen
		pred_name = names[int(out)]

		cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 2, cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w, y+h), (100, 100, 100), 2)


	cv2.imshow("Faces", frame)

	key = cv2.waitKey(1) & 0xff
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()








