# from crypt import methods
import os

from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap
from inference import get_prediction

DIR_IMAGE_PATH = "./static/images"

app = Flask(__name__, template_folder='templates')
Bootstrap(app)

"""
This is a simple Flask application that returns a JSON response with a message.
"""
@app.route('/hello', methods=['GET'])
def hello():
    """
    This function handles the root route (/) and returns a JSON response.
    """
    # return {'message': 'Hello World'}
    return 'Hello World'


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    This function renders the index.html template.
    """
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join(DIR_IMAGE_PATH, uploaded_file.filename)
            uploaded_file.save(image_path)
            class_name = get_prediction(image_path)
            print("Class name:", class_name)  # in terminal: Class name: cat
            result = {
                "class_name": class_name,
                "image_path": image_path
            }
            return render_template('prediction_result.html', result=result)
    return render_template('index.html')


"""
This is the entry point of the Flask application.
"""
if __name__=="__main__":
    # app.run(host='0.0.0.0', port=5000, debug=True)
    # app.run(host='127.0.0.1', port=5000, debug=True)
    app.run(debug=True)
