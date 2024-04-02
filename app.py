from flask import Flask, render_template, Response, session, request, redirect, url_for
import cv2
from get_dbase import get_dbase
import pandas as pd
import numpy as np


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
    username = session.get("username")   
   
    df= pd.read_csv("user_embedding.csv")
    _, n = df.shape
    embed = df.loc[0,:n-2].to_numpy()
    try:

        face = collect_data()
        embed_im = get_embeddings(face[0])
        cos = cosine(embed,embed_im)
        if cos >= 0.5:
            return render_template('succ_search.html', username = username)
        else:
            return render_template('no_match.html', username = username)
    except:
        return render_template("error_page.html")

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

@app.route('/model_training')
def model_training():
    try:

        face = collect_data()
    
        embed_im = get_embeddings(face[0])
        
        username = session.get("username")   
        query = {'Email' : username}
        user = db_collection.find_one(query)
        em = list(embed_im)
        em.append(user["Name"])
        df = pd.DataFrame(columns = range(len(em)))
        df.loc[len(df.index)] = em
        newvalues = { "$set": { "Face_model": "Scanned" } }
        db_collection.update(query,newvalues)
        
        df.to_csv("images_emb.csv")
    
        return render_template('succ_scan.html')
    except:
        return render_template("error_page.html")


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