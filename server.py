from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
import caer

app = Flask(__name__)

model = load_model('model1.h5')

characters=['Betel Leaves', 'Mint Leaves', 'Balloon vine', 'Amaranthus Green', 'Coriander Leaves', 'Curry Leaf', 'Black Night Shade', 'Malabar Spinach (Green)', 'Giant Pigweed', 'False Amarnath']
IMG_SIZE=[80, 80]

def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, tuple(IMG_SIZE))
    img = caer.reshape(img, IMG_SIZE,1)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['image']
        file_path = "static/" + file.filename
        file.save(file_path)
        img = cv.imread(file_path)
        predictions = model.predict(prepare(img))
        detected_leaf= characters[np.argmax(predictions[0])]

        return render_template('index.html', prediction=detected_leaf)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
