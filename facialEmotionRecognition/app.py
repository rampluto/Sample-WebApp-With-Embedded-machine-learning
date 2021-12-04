from flask import Flask, render_template, request               
from models import Utility
import os

app = Flask(__name__)                                    
utility = Utility()
utility.get_model()

@app.route("/", methods=['GET', 'POST'])                 
def home():                                              
    return render_template('home.html')                  

@app.route("/predict", methods = ['GET','POST'])
def predict():
    file_path = ""
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join("D:/study materials/pythonDevs/mlwebapp/facialEmotionRecognition/", filename)                       #slashes should be handeled properly
        file.save(file_path)
        print(filename)
        product = utility.prediction(file_path)
        print(product)
        
    return render_template('predict.html', product = product, user_image = file_path)

if __name__ == "__main__":
    app.run()           