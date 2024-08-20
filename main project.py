import face_recognition
import cv2
import numpy as np


imgparth = face_recognition.load_image_file('photos/parth.jpg')
imgparth = cv2.cvtColor(imgparth, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('photos/parth_test.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgparth)[0]
encodeparth = face_recognition.face_encodings(imgparth)[0]
cv2.rectangle(imgparth,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2) # top, right, bottom, left

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeparth], encodeTest)
faceDis = face_recognition.face_distance([encodeparth], encodeTest)
print(results, faceDis)

cv2.imshow('parth patel', imgparth)
cv2.imshow('parth test', imgTest)
cv2.waitKey(0)
