from flask import Flask,render_template,redirect,jsonify
from training import TestPreprocess,Audio,Model,PlayAudio
from threading import Thread
import numpy as np

app = Flask(__name__)
model = Model()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/camera")
def cam():
    return render_template("camera.html")

@app.route("/mic")
def mic():

    try:
        t1.join()
    except:
        pass

    audio = Audio()
    audio.recorder()

    X = TestPreprocess().preprocess(r"recording1.wav")
    y = model.predict(X)

    if(y=="Not crying"):
        return render_template("cry_res.html",context="Baby is Not Crying",description="Chill! No crying sound detection from baby")
    else:
        t1 = Thread(target=PlayAudio)
        t1.start()
        return render_template("cry_res.html",context="Baby is Crying",description = "Oops! Baby is Crying.. Playing some pleasant music")


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/temperature")
def temperature():
    return render_template("temp.html")





if __name__ == "__main__":
    app.run(debug=True)

