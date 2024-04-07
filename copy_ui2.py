from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import pandas   # changed 10/4/22
import webbrowser   #changed today
import form_ex
import sqlite3
import datetime 

#for flask 
import os

from sys import argv
from flask import Flask, flash, request, redirect, url_for,send_from_directory,render_template,Response, jsonify

from werkzeug.utils import secure_filename
from flask_cors import CORS




detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_classifier = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
points=[]
#argss = "C:\\Users\\user\\Desktop\\data\\video\\test.mp4"
argss = './data/video/test.mp4'
comp_stress = 0
score = 0
final_stress_value = 0
form_status = 0
submittion_status= -1     
user_details = ""
final_stress_value_final = 0

def eye_brow_distance(leye,reye):
    distq = dist.euclidean(leye,reye)
    points.append(int(distq))
    print("POINTS",points)
    return distq

def emotion_finder(faces,frame):
    global emotion_classifier
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['scared','sad']:
        label = 'stressed'
    else:
        label = 'not stressed'
    return label
    
def normalize_values(points,disp):
    normalized_value = abs(disp - np.min(points))/abs(np.max(points) - np.min(points))
    print("NORMALIZED VALUE",normalized_value)
    stress_value = np.exp(-(normalized_value))
    #print(stress_value)
    if stress_value>=75:
        return stress_value,"High Stress"
    else:
        return stress_value,"low_stress"

def value():
    val = input("Enter file name or press enter to start webcam : \n")
    if val == "":
        val = 0
    return val
 
#for flask
UPLOAD_FOLDER = 'data/video'
#ALLOWED_EXTENSIONS = {'mp4','m4p','m4v','mov','wmv','avi','webm'}
ALLOWED_EXTENSIONS = {'mp4'}
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#req = app.test_request_context()

@app.route('/logincredentials', methods=['GET', 'POST'])
def logincredentials():
    global submittion_status
    global user_details
    error = None
    if request.method == 'POST':
        user_details = request.form['username']
        if request.form['username'] not in ['admin','user'] or request.form['password'] not in ['admin','user']:
            #error = 'Invalid Credentials. Please try again.'
            return redirect('/logincredentials')
        else:
            submittion_status = 0
            return redirect('/')
            
            
@app.route("/login", methods=["POST","GET"])
def login():
    global score
    global final_stress_value
    global submittion_status 
    global comp_stress
    global user_details

    if request.method == "POST":
        val = request.form
        if val["n1"] == "YES":
            score += 5
        else:
            score += 3

        if val["n3"] == "1 - 3 hours":
            score += 3

        else:
             score += 5
        
        if val["n5"] == "Less than a week":
            score += 3

        else:
            score += 5

        if val["n7"] == "Less than a week":
            score += 3

        else:
            score += 5

        if val["n9"] == "Below 30%":
            score += 1

        elif val["n9"] == "Below 60%":
            score += 3

        else:
            score += 5

        if val["n12"] == "YES":
            score += 5

        else:
            score += 3

        if val["n14"] == "YES":
            score += 5

        else:
            score += 3

        if val["n16"] == "YES":
            score += 5

        else:
            score += 3
            
        if val['n18'] == "YES":
            score += 5
        else:
            score += 2
        
        if val['n20'] == "More often":
            score += 5
        else:
            score += 3
        
        
        print("form score:",score)
        
        final_stress_value = score + comp_stress
        final_stress_value_final = int(final_stress_value)
        conn = sqlite3.connect('stressed_employee.db')
        print("connection created")
        print(user_details)
        print(final_stress_value_final)
        cursor = conn.cursor()
        if user_details == 'user':
            curr_time = datetime.datetime.now()
            test = """INSERT INTO EMPLOYEESTRESS values('{}','{}','{}')""".format('user',curr_time,final_stress_value_final)
            cursor.execute(test)
        elif user_details == 'admin':
            cur = conn.cursor()
            cur.execute("SELECT * FROM EMPLOYEESTRESS")
            print(cur.fetchall())
        conn.commit()
        conn.close()
        
        
        
        if(final_stress_value > 40):
           submittion_status = 3        #r1
        if(final_stress_value > 50):
            submittion_status = 4       #r2
        if(final_stress_value > 60):
            submittion_status = 5       #r3
        if(final_stress_value > 70):
            submittion_status = 6       #r4
        if(final_stress_value > 80):
            submittion_status = 7       #r5
            
        if(final_stress_value > 80):    
            submittion_status = 8 
            
        return redirect('/')
        
        #print("yes ",score)
      

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global submittion_status
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'test.mp4'))
            submittion_status=1
            
            #return redirect(url_for('uploaded_file',filename=filename))
            
    if(submittion_status == -1):
        return render_template('logincredentials.html')
    if(submittion_status==0):
    
        return render_template('index.html')       
    if(submittion_status==1):
    
        return render_template('index2.html')
   
    if(submittion_status==2):
        return render_template('login.html')
    #start
    if(submittion_status == 3):
        return render_template('remedy4.html')
        
    if(submittion_status == 4):
        return render_template('remedy4.html')
    
    if(submittion_status == 5):
        return render_template('remedy4.html')
        
    if(submittion_status == 6):
        return render_template('remedy4.html')
    
    if(submittion_status == 7):
        return render_template('remedy4.html')    
    
    if(submittion_status == 8):
        return render_template('remedy4.html')  
   

    

@app.route('/download')
def download_file():
    return send_from_directory(app.config['UPLOAD_FOLDER'],'result.mp4')
  
@app.route("/terminate/", methods=['POST'])
def terminate():
    global submittion_status
    submittion_status = -1
    return redirect('/')  

@app.route("/detectlive/", methods=['POST'])
def detectlive():
    global argss
    global submittion_status
    submittion_status = 1
    argss = 0
    return redirect('/')      #changed 9/4/22
    #return render_template('index2.html')

def gen_frames():  
    while True:
        
        success, frame = main_frame  # read main_frame saved by fn main
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
               


    
def calc_stress(cap,frame):
    global comp_stress
    capture_duration = time.time() + 10
    

    while time.time() < capture_duration:
        (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

        #preprocessing the image
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        detections = detector(gray,0)
        for detection in detections:
            emotion = emotion_finder(detection,gray)
            #cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            shape = predictor(frame,detection)
            shape = face_utils.shape_to_np(shape)
               
            leyebrow = shape[lBegin:lEnd]
            reyebrow = shape[rBegin:rEnd]
                
            reyebrowhull = cv2.convexHull(reyebrow)
            leyebrowhull = cv2.convexHull(leyebrow)

            cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

            distq = eye_brow_distance(leyebrow[-1],reyebrow[0])

            stress_value,stress_label = normalize_values(points,distq)
            print(stress_value)
            if stress_value > comp_stress:
                comp_stress = int(stress_value * 100)
             
    if comp_stress > 50:
        comp_stress = comp_stress - 50
    print("computed stress:",comp_stress)
    #form_status = 1
    return comp_stress

@app.route('/video_feed')
def video_feed():
    global comp_stress
    print("feeder called")
    return Response(main(argss), mimetype='multipart/x-mixed-replace; boundary=frame')

    
    
@app.route('/form/', methods=['POST'])
def form():
    global submittion_status 
    submittion_status = 2
    return redirect('/')


def main(arg):
    global submittion_status
    cap = cv2.VideoCapture(arg)
    flag = 0
    while(flag <= 0):
        _,frame = cap.read()
        if arg == 0:
            frame = cv2.flip(frame,1)
        
        frame = imutils.resize(frame,width=800,height=800)    #changed 9/4/22
        
        
        (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
        (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]

        #preprocessing the image
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        detections = detector(gray,0)
        for detection in detections:
            emotion = emotion_finder(detection,gray)
            cv2.putText(frame, emotion, (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            """shape = predictor(frame,detection)
            shape = face_utils.shape_to_np(shape)
               
            leyebrow = shape[lBegin:lEnd]
            reyebrow = shape[rBegin:rEnd]
                
            reyebrowhull = cv2.convexHull(reyebrow)
            leyebrowhull = cv2.convexHull(leyebrow)

            cv2.drawContours(frame, [reyebrowhull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [leyebrowhull], -1, (0, 255, 0), 1)

            distq = eye_brow_distance(leyebrow[-1],reyebrow[0])"""
            
            if emotion == 'stressed':
                print(emotion)
                comp_stress = calc_stress(cap,frame)
                print("This is the final stress level", comp_stress)
                flag = 1
                #submittion_status = 2
                #return redirect('/')
                #cap.release()
                #webbrowser.open("http://127.0.0.1:5000/login")
                #form_ex.main()
                
        #cv2.imshow("Frame", frame)
        
        # for straeming
        
        framestream = cv2.imencode('.jpg',frame)[1].tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + framestream + b'\r\n')  # concat frame one by one and show result
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()

    #plt.plot(range(len(points)),points,'ro')
    #plt.title("Stress Levels")
    #plt.show()
    
    

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",use_reloader=False)

