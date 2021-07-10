import os
from tensorflow.keras.models import load_model
#from keras.models import load_model
import numpy as np
from flask import Flask, request, redirect, url_for, send_from_directory,render_template
from keras.preprocessing import image
from flask_ngrok import run_with_ngrok
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import cv2

from werkzeug.utils import secure_filename
app = Flask(__name__)
run_with_ngrok(app)
#model=tk.load_model("mask_detector.h5")
#model.encode().decode()
model_path="models/mymodel.h5"
model=load_model(model_path,compile=False)
def model_predict(img_path, model):
    #test_img = cv2.imread(img_path,0)
    #print(test_img.shape)
    test_image=image.load_img(img_path,target_size=(150,150,3))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    res=model.predict(test_image)[0][0]
    return res
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
@app.route('/help')
def form():
    return render_template('help.html')
@app.route('/upload', methods=["GET","POST"])
def upload():
    if request.method == 'POST':
        file = request.files['image_file']
        basepath=os.path.dirname(__file__)
        filename=secure_filename(file.filename)
        filepath=os.path.join(basepath,'uploads/',file.filename)
        file.save(filepath)
        print(filepath)
        livepreds = model_predict(filepath,model)
        if livepreds==1:
            return render_template('withoutmask.html',filename=filename)
        else:
            return render_template('withmask.html',filename=filename)
    return None

@app.route('/upload/<filename>')
def send_image(filename):
    #print('display_image filename: ' + filename)
    return send_from_directory("uploads", filename)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
@app.route('/face')
def face_detect():  
    # load our serialized face detector model from disk
    prototxtPath = "models/deploy.prototxt"
    weightsPath = "models/res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    # load the face mask detector model from disk
    maskNet = load_model("models/mask_detector.model")
    # initialize the video stream
    vs = VideoStream(src=0).start()
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
	    # to have a maximum width of 400 pixels
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        # detect faces in the frame and determine if they are wearing a
	    # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        # loop over the detected face locations and their corresponding locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            # determine the class label and color we'll use to draw
		    # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            # display the label and bounding box rectangle on the output frame
            cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imshow("Face Mask Detector", frame)
        # show the output frame
        key = cv2.waitKey(1) & 0xFF
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
if __name__ == '__main__':
          app.run()
          
        

