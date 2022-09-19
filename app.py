# 1.0 Train our model
# 2.0 Create Web App using Flask
# 3.0 Commit the code in Github
# 4.0 Create an account in Heroku (PAAS)
# 5.0 Link Github to Heroku
# 6.0 Deploy the mode1
# 7.0 Web App is ready




import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# What happens is that the route page will render the template (index.html)
@app.route('/')
def home():
    return render_template('index.html')


# 
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

# So what this does is that it takes all the fields from the request form and transform it into an array (see above) and than use the model to predict
# 
#      <!-- Main Input For Receiving Query to our ML -->
#     <form action="{{ url_for('predict')}}"method="post">
#     	<input type="text" name="experience" placeholder="Experience" required="required" />
#         <input type="text" name="test_score" placeholder="Test Score" required="required" />
# 		<input type="text" name="interview_score" placeholder="Interview Score" required="required" />

#         <button type="submit" class="btn btn-primary btn-block btn-large">Predict</button>

# the prediction text above will be replace 
# {{ prediction_text }}



if __name__ == "__main__":
    app.run(debug=True)
    
#This is the main function that will run the entire flask 
