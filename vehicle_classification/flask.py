import os
from flask import Flask, request, render_template, send_from_directory
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
# SARTHAK MODI
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
@app.route("/") 
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'uploads/')

    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        new_modell = load_model('vehicle_resnet.h5')
        test_image = image.load_img('uploads\\' + filename, target_size=(224, 224))
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)
        result = new_modell.predict(test_image)
        ans = np.argmax(result, axis=1)
        if ans==0:
            prediction="Non-Vehicle"
        elif ans==1:
            prediction = "Vehicle"

    return render_template("template.html",image_name=filename, text=prediction)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("uploads", filename)

if __name__ == "__main__":
    app.run(debug=False)
