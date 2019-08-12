from flask import Flask
from flask import render_template
from flask import Flask, request, send_from_directory
from flask import jsonify
from sklearn.externals import joblib

import datetime
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/historical')
def historical(**args):
    start_date = request.args.getlist('startDate')[0]
    end_date = request.args.getlist('endDate')[0]
    acc_type = request.args.getlist('accidentType')[0]
    print(start_date, end_date, acc_type)
    historical = get_historical_values(start_date, end_date, acc_type)
    hist = build_data(historical)
    return jsonify([{'label': acc_type.lstrip('clase_'), 'borderColor': '#FF0000',
                    #  'data': [{'x': datetime.datetime.fromtimestamp(1562798731), 'y': 30},
                    #           {'x': datetime.datetime.fromtimestamp(1565477131), 'y': 330}],
                    'data': hist,
                    'type': 'line', 'fill': False
                    }])


def get_historical_values(start_timestmap, end_timestamp, acc_type):
    df = pd.read_csv('../datasets/historical.csv')
    start_date = datetime.datetime.fromtimestamp(int(start_timestmap)/1000)
    end_date = datetime.datetime.fromtimestamp(int(end_timestamp)/1000)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    raw_data = df[(df['date'] >= start_str) & (df['date'] <= end_str)]
    return raw_data[['date', acc_type]]

def build_data(data):
    output = []
    for index in range(0, data.shape[0]):
        output.append({'x': datetime.datetime.strptime(data.iloc[index][0], '%Y-%m-%d'),
                       'y': int(data.iloc[index][1])})
        index += 1
    return output

if __name__ == "__main__":
    app.run(debug=True)
