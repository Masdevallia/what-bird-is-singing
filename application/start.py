
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
		subprocess.run(['python', './8-Predictions.py', filename])
		return 'Success!'
	else:
		return 'No file chosen'

if __name__ == '__main__':
    app.run()

# python3 application/start.py