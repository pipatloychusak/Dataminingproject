import pickle
import numpy as np

from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def main():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    Sex = request.form['Sex']
    Age = request.form['Age'] 
    Career =  request.form['Career']
    Income  = request.form['Income']
    Budget =  request.form['Budget']
    UsagecharacteristicsNo1 =  request.form['Usage characteristics No1']
    UsagecharacteristicsNo2 =  request.form['Usage characteristics No2']
    UsagecharacteristicsNo3 =  request.form['Usage characteristics No3']
    DesigncharacteristicsNo1 =  request.form['Design characteristics No1']
    DesigncharacteristicsNo2 =  request.form['Design characteristics No2']
    DesigncharacteristicsNo3 =  request.form['Design characteristics No3']

    tmp_str = str(Sex) + str(Age) + str(Career) + str(Income) + str(Budget) + str(UsagecharacteristicsNo1) + str(UsagecharacteristicsNo2)+ str(UsagecharacteristicsNo3)+ str(DesigncharacteristicsNo1)+ str(DesigncharacteristicsNo2)+ str(DesigncharacteristicsNo3)
    # print("input"+ tmp_str)
    #Check if input is empty or not
    # 75 is 001001xxxxxxxx
    #Check if enough features or not
    if not tmp_str or len(tmp_str) != 75:
        return render_template('main.html', prediction_text='{}'.format('input error กรุณากรอกข้อมูลอีกครั้ง'))
    else:
         tmp_str_split = list(tmp_str)
    
    

    tmp_int = []
    for i in tmp_str_split:
        tmp_int.append(int(i))

        final_features = np.array(tmp_int)
        final_input = final_features.reshape(1,-1)
        prediction = model.predict(final_input)
        output = prediction[1]

        return render_template('main.html', prediction_text='{}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
