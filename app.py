import os
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, send_from_directory,render_template,Response
from keras.preprocessing import image
import cv2
import os
from werkzeug.utils import secure_filename
app = Flask(__name__)
face_classifier = cv2.CascadeClassifier('model/stream_faceclassifier.xml')
model_path="model/model.h5"
classifier=load_model('model/model.h5')
ds_factor=0.6
model=load_model(model_path,compile=False)
def model_predict(img_path, model):
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
@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method == 'POST':
        file = request.files['image_file']
        basepath=os.path.dirname(__file__)
        filename=secure_filename(file.filename)
        filepath=os.path.join(basepath,'upload/',file.filename)
        file.save(filepath)
        livepreds = model_predict(filepath,model)
        if livepreds==1:
            return render_template('withoutmask.html',filename=filename)
        else:
            return render_template('withmask.html',filename=filename)
    return None

@app.route('/predict/<filename>')
def send_image(filename):
    return send_from_directory("upload", filename)

@app.route('/live')
def live():
    return render_template('live.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'

               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    def __del__(self):
        self.video.release()
    def get_frame(self):
        ret,img=self.video.read()
        face=face_classifier.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        for(x,y,w,h) in face:
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite('upload/temp.jpg',face_img)
            test_image=image.load_img('upload/temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=model.predict(test_image)[0][0]
            if pred==1:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        ret,jpeg = cv2.imencode('.jpg',img)
        return jpeg.tobytes()
if __name__ == '__main__':
          app.run()
          
        

