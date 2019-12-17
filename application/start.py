
import subprocess
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
		return render_template('index.html')

@app.route('/Birdentifier', methods = ['GET', 'POST'])
def Birdentifier():
	filename = request.form['filename']
	if filename:
		subprocess.run(['python', './5-Predictions.py', filename])
		return 'The answer to your request will be opened in a new browser tab. Thank you for using our service.'
	else:
		return 'No file chosen. Please go back and select a file before pressing the submit button.'

if __name__ == '__main__':
    app.run()

# python3 application/start.py