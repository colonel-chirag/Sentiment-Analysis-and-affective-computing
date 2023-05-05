# import necessary libraries
from flask import Flask, request, jsonify, render_template
import pickle

# load the machine learning model
with open('lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# initialize the Flask application
app = Flask(__name__)

# create a route to handle requests to the root URL
@app.route('/', methods=['GET', 'POST'])
def home():
    # check if the request method is POST (i.e., form submission)
    if request.method == 'POST':
        # extract the form inputs
        input1 = float(request.form['input1'])
        input2 = float(request.form['input2'])
        input3 = float(request.form['input3'])
        input4 = float(request.form['input4'])

        # make predictions using the loaded model
        prediction = model.predict([[input1, input2, input3, input4]])

        # return the result to the user
        return render_template('result.html', prediction=prediction[0])

    # if the request method is GET (i.e., page load), display the form
    else:
        return render_template('form.html')

# run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
