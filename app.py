from flask import Flask, render_template, Response, session, request, redirect, url_for,jsonify
import cv2
from get_dbase import get_dbase
import numpy as np
import base64
from mtcnn import MTCNN



from datacollection import *



app=Flask(__name__, template_folder = 'templates')

dbase = get_dbase()
db_collection = dbase['user']

app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route('/')
def index():
    username = session.get("username")
    if username:
        current_user = username
        status = "Log out"
    else:
        current_user = "User"
        status = "Log in"
    return render_template('index.html', current_user = current_user, status = status)

@app.route('/face_first')
def face_first():
    username = session.get("username")
    if username:
        return redirect(url_for("home_recog"))
    else:
        return redirect(url_for('sign_in_up'))
    
@app.route('/already-scanned')
def already_scanned():
    return render_template('already_scanned.html')


@app.route("/scan")
def scan():
    username = session.get("username")   
    if username:
        query = {'Email' : username}
        user = db_collection.find_one(query)
        model = user["Face_model"]
        if model:
            return redirect(url_for('already_scanned'))
        return render_template("scan.html") 
    else:
        return redirect(url_for('sign_in_up'))

@app.route("/scanning_camera")
def scanning_cam():
    return render_template('scanning_camera.html')

@app.route("/searching_camera")
def scanning_cam():
    return render_template('searching_camera.html')

@app.route("/search")
def search():
    username = session.get("username")   
    if username:
        query = {'Email' : username}
        user = db_collection.find_one(query)
        model = user["Face_model"]
        if model:
            return render_template("search.html") 
        return render_template('scan_first.html')
    else:
        return redirect(url_for('sign_in_up'))

@app.route('/model_testing')
def model_testing():
    try:
        username = session.get("username")
        user = db_collection.find_one({'Email': username})
        
        stored_embedding = np.array(user["Embedding"])    
        
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect face
        detector = MTCNN()
        result = detector.detect_faces(img)
        if len(result) == 0:
            return jsonify({"success": False, "message": "No face detected"})
        
        face = result[0]['box']
        x, y, w, h = face
        face_img = img[y:y+h, x:x+w]


        embed_im = get_embeddings(face_img)
        cos = cosine(stored_embedding,embed_im)
        if cos >= 0.5:
            return jsonify({"success": True, "message": "Match found!"})
        else:
            return jsonify({"success": True, "message": "Match not found!"})
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "message": "Something went wrong."})

@app.route('/home_recog')
def home_recog():
    username = session.get("username")   
    if username:
        current_user = username
        status = "Log out"
    else:
        current_user = "User"
        status = "Log in"
    
    return render_template('home_recog.html',current_user = current_user, status = status)

@app.route('/model_training', methods=["POST"])
def model_training():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Detect face
        detector = MTCNN()
        result = detector.detect_faces(img)
        if len(result) == 0:
            return jsonify({"success": False, "message": "No face detected"})

        # Only one face detected, get embedding
        face = result[0]['box']
        x, y, w, h = face
        face_img = img[y:y+h, x:x+w]

        embed_im = get_embeddings(face_img)

        username = session.get("username")
        query = {'Email': username}
        newvalues = {
            "$set": {
                "Face_model": "Scanned",
                "Embedding": embed_im.tolist()
            }
        }
        db_collection.update_one(query, newvalues)

        return jsonify({"success": True, "message": "Face scanned!"})
    except Exception as e:
        print("Error:", e)
        return jsonify({"success": False, "message": "Something went wrong."})
    
@app.route('/success')
def success():
    return render_template('succ_scan.html')

app.route('/succ_search')
def succ_search():
    return render_template('succ_search.html')

app.route('/no_match')
def no_match():
    return render_template('no_match.html')



@app.route('/login_signup')
def sign_in_up():
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))


@app.route('/redirect_signup', methods = ['POST'])
def redirect_signup():
    name = request.form['logname']
    password = request.form['logpass']
    email = request.form['logemail']
    user = {
        "Name" : name,
        "Email" : email,
        "Password" : password,
        "Face_model" : None
    }
    db_collection.insert_one(user)
    username = session.get("username")   
    if username:
        current_user = username
        status = "Log out"
    else:
        current_user = "User"
        status = "Log in"
    
    return render_template('redirect_signup.html',current_user = current_user, status = status)

@app.route('/redirect_signin',methods = ['POST'])
def redirect_signin():
    query = {'Email' : request.form['logemail']}
    user = db_collection.find(query)
    if user:
        session["username"] = user[0]['Email']
        return redirect(url_for('index'))
    return redirect(url_for('sign_in_up'))

if __name__=='__main__':
    app.run(debug=True,use_reloader=False, port=8000)