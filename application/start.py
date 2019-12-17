
import subprocess
from flask import Flask, render_template, request
import os
from werkzeug import secure_filename


# instance of the Flask object
app = Flask(__name__)

# Upload folder
app.config['UPLOAD_FOLDER'] = './application/uploaded'


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/Birdentifier', methods=['POST'])
def Birdentifier():
	if request.method == 'POST':
		# get the file
		f = request.files['filename']
		if f:
			filename = secure_filename(f.filename)
			# save the file in the "uploaded" directory
			f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			subprocess.run(['python', './5-Predictions.py', filename])
			return 'The answer to your request will be opened in a new browser tab. Thank you for using our service.'
		else:
			return 'No file chosen. Please go back and select a file before pressing the submit button.'


if __name__ == '__main__':
    app.run()


# python3 application/start.py