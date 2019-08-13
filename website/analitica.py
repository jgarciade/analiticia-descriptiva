from flask import Flask
from flask import render_template
from flask import Flask, request, send_from_directory
from flask import jsonify
from flask_cors import CORS
from sklearn.externals import joblib

import datetime
import pandas as pd
import holidays_co
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/historical')
def historical(**args):
    start_date = request.args.getlist('startDate')[0]
    end_date = request.args.getlist('endDate')[0]
    acc_type = request.args.getlist('accidentType')[0]
    monthly = request.args.getlist('monthly')[0] == 'true'
    historical = get_historical_values(start_date, end_date, acc_type, monthly)
    hist = build_data(historical)
    return jsonify([{'label': acc_type.lstrip('clase_'), 'borderColor': '#FF0000',
                    'data': hist,
                    'type': 'line', 'fill': False
                    }])

@app.route('/predict')
def predict(**args):
    start_date = request.args.getlist('startDate')[0]
    end_date = request.args.getlist('endDate')[0]
    acc_type = request.args.getlist('accidentType')[0]
    print(start_date, end_date, acc_type)
    df = build_predict_dataset(start_date, end_date)
    y_prd = model_predict(df, acc_type)
    output = []
    for index in range(0, df.shape[0]):
        output.append({'x': df.iloc[index][0],
                        'y': y_prd[index]})
        index += 1
    return jsonify([{'label': acc_type.lstrip('clase_'), 'borderColor': '#FF0000',
                    'data': output,
                    'type': 'line', 'fill': False
                    }])


def get_historical_values(start_timestmap, end_timestamp, acc_type, monthly):
    df = pd.read_csv('../datasets/historical.csv')
    start_date = datetime.datetime.fromtimestamp(int(start_timestmap)/1000)
    end_date = datetime.datetime.fromtimestamp(int(end_timestamp)/1000)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    raw_data = df[(df['date'] >= start_str) & (df['date'] <= end_str)]
    if not monthly:
        return raw_data[['fecha', acc_type]]
    obj_data = raw_data[['mes', acc_type]]
    obj_data = obj_data.groupby(['mes']).sum()
    obj_data = obj_data.reset_index()
    obj_data['date'] = obj_data['mes'].apply(lambda x: datetime.date(2014, x, 1).strftime("%Y-%m-%d"))
    return obj_data[['date', acc_type]]

def build_data(data):
    output = []
    for index in range(0, data.shape[0]):
        output.append({'x': datetime.datetime.strptime(data.iloc[index][0], '%Y-%m-%d'),
                       'y': int(data.iloc[index][1])})
        index += 1
    return output

def build_predict_dataset(start_timestamp, end_timestamp):
    df = pd.DataFrame()
    init_obj = datetime.datetime.fromtimestamp(int(start_timestamp)/1000)
    date = []
    while init_obj.timestamp() < int(end_timestamp)/1000:
        date.append(init_obj)
        init_obj = init_obj + datetime.timedelta(days=1)

    df['date'] = date
    df['festivo'] = df['date'].apply(lambda x: int(holidays_co.is_holiday_date(x)))
    df['mes'] = df['date'].apply(lambda x: x.month if x.month >= 10 else f'0{x.month}')
    df['dia_semana'] = df['date'].apply(lambda x: x.isoweekday())
    df['plan_retorno'] = df['date'].apply(lambda x: is_holiday_return(x))
    df['plan_exodo'] = df['date'].apply(lambda x: is_holiday_leave(x))
    df = pd.get_dummies(df, columns=['dia_semana'])
    df = pd.get_dummies(df, columns=['mes'])

    train_labels = ['festivo','plan_exodo', 'plan_retorno', 'mes_01', 'mes_02', 'mes_03', 'mes_04', 'mes_05',
                    'mes_06', 'mes_07', 'mes_08', 'mes_09', 'mes_10', 'mes_11', 'mes_12', 'dia_semana_1',
                    'dia_semana_2', 'dia_semana_3', 'dia_semana_4', 'dia_semana_5', 'dia_semana_6', 'dia_semana_7']

    for label in train_labels:
        if label in df.columns:
            continue
        df[label] = 0

    return df

def is_holiday_leave(date_obj):
    for i in range(1, 4):
        if int(holidays_co.is_holiday_date(date_obj + datetime.timedelta(days=i))):
            return int(True)
    return int(False)

def is_holiday_return(date_obj):
    if holidays_co.is_holiday_date(date_obj) and date_obj.isoweekday() == 1:
        return int(True)
    return int(False)

def model_predict(X_test, acc_type):
    train_labels = ['festivo','plan_exodo', 'plan_retorno', 'mes_01', 'mes_02', 'mes_03', 'mes_04', 'mes_05',
                'mes_06', 'mes_07', 'mes_08', 'mes_09', 'mes_10', 'mes_11', 'mes_12', 'dia_semana_1',
                'dia_semana_2', 'dia_semana_3', 'dia_semana_4', 'dia_semana_5', 'dia_semana_6', 'dia_semana_7']
    model = joblib.load(f'../predictor_{acc_type}.sk')
    y_prd = model.predict(X_test[train_labels])
    return y_prd


if __name__ == "__main__":
    app.run(debug=True)
