import csv
from flask import Flask, render_template, request, render_template_string
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.metrics import AUC
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from compare_ecg import is_ekg_image
import os
from datetime import datetime

app = Flask(__name__)

# dependencies = {
#    'auc_roc': AUC
# }

verbose_name = {
0: 'Myocardial_Infarction',
1: 'History_of_Myocardial_Infarction', 
2: 'Abnormal_Heartbeat',
3: 'Normal_person',
           }



model = load_model('ecg.h5')

def predict_label(img_path):
	test_image = image.load_img(img_path, target_size=(224,224))
	test_image = image.img_to_array(test_image)/255.0
	test_image = test_image.reshape(1, 224,224,3)

	predict_x=model.predict(test_image) 
	classes_x=np.argmax(predict_x,axis=1)
	
	return verbose_name[classes_x[0]]


def save_prediction_to_csv(img_path, prediction):
	# file_exists = os.path.isfile('predictions.csv')
	file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'predictions.csv')
	file_exists = os.path.isfile('predictions.csv')
	with open('predictions.csv', mode='a', newline='') as file:
		writer = csv.writer(file)
		if not file_exists:
			writer.writerow(['id', 'image_path', 'prediction', 'date'])	

		last_id = 0
		if file_exists:
			with open(file_path, mode='r') as read_file:
				reader = csv.reader(read_file)
				for row in reader:
					pass  
				if row:
					last_id = int(row[0])  
        
		new_id = last_id + 1
		writer.writerow([new_id, img_path, prediction, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

 
@app.route("/")
@app.route("/first")
def first():
	return render_template('first.html')
    
    
@app.route("/index", methods=['GET', 'POST'])
def index():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		
		img_path = "static/tests/" + img.filename	
		img.save(img_path)
		image_ref_path = "static/tests/reference_ecg.jpg"
		
		isImageEcgPercentage = is_ekg_image(image_ref_path, img_path)
		if isImageEcgPercentage > 65:
			predict_result = predict_label(img_path)
			save_prediction_to_csv(img_path, predict_result)
			return render_template("prediction.html", prediction = predict_result, img_path = img_path)
		else:
			return render_template_string("""
			<p>Надане зображення не є результатом ЕКГ. Будь ласка, повторіть спробу з іншим зображенням.</p> 
			<a href="/index">Повернутись</a> """), 400

@app.route("/performance")
def performance():
	return render_template('performance.html')
    
	
if __name__ =='__main__':
	app.run(debug = True)


	

	


