from flask import Flask, render_template, request, send_file
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import send_file
import io


app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # Load dataset
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    # Clean data
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    df['Children in HHS Care'] = (
        df['Children in HHS Care']
        .astype(str).str.replace(',', '').str.strip()
    )
    df['Children in HHS Care'] = pd.to_numeric(df['Children in HHS Care'], errors='coerce')
    df = df.dropna()

    # 🔽 FILTERS
    start = request.args.get('start')
    end = request.args.get('end')

    if start:
        df = df[df['Date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['Date'] <= pd.to_datetime(end)]

    # 🛑 HANDLE EMPTY DATA
    if len(df) == 0:
        return render_template("index.html",
                               table="<p>No data available</p>",
                               pred_table="<p>No prediction</p>",
                               latest=0, avg_val=0, max_val=0)

    # 🔥 KPIs
    latest = int(df['Children in HHS Care'].iloc[-1])
    avg_val = int(df['Children in HHS Care'].mean())
    max_val = int(df['Children in HHS Care'].max())

    # 🔥 MODEL (only if enough data)
    if len(df) >= 2:
        df['day_num'] = np.arange(len(df))
        X = df[['day_num']]
        y = df['Children in HHS Care']

        model = LinearRegression()
        model.fit(X, y)

        future_days = np.arange(len(df), len(df) + 7).reshape(-1, 1)
        predictions = model.predict(future_days)

        last_date = df['Date'].iloc[-1]
        future_dates = pd.date_range(last_date, periods=7)

        pred_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Care Load': predictions.astype(int)
        })
    else:
        pred_df = pd.DataFrame({
            'Date': [],
            'Predicted Care Load': []
        })

    # 🔥 GRAPH
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Children in HHS Care'], label='Actual')

    if not pred_df.empty:
        plt.plot(pred_df['Date'], pred_df['Predicted Care Load'],
                 linestyle='--', label='Predicted')

    plt.legend()
    plt.title("Care Load Forecast")
    plt.xlabel("Date")
    plt.ylabel("Children Count")

    if not os.path.exists("static"):
        os.makedirs("static")

    plt.savefig("static/graph.png")
    plt.close()

    # 🔥 TABLES
    table = df.tail(10).to_html(classes='table', index=False)
    pred_table = pred_df.to_html(classes='table', index=False) if not pred_df.empty else "<p>No prediction data</p>"

    return render_template(
        "index.html",
        table=table,
        pred_table=pred_table,
        latest=latest,
        avg_val=avg_val,
        max_val=max_val,
        start=start,
        end=end
    )

@app.route('/download')
def download():
    df = pd.read_csv("HHS_Unaccompanied_Alien_Children_Program.csv")

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])

    df['Children in HHS Care'] = (
        df['Children in HHS Care']
        .astype(str).str.replace(',', '').str.strip()
    )
    df['Children in HHS Care'] = pd.to_numeric(df['Children in HHS Care'], errors='coerce')
    df = df.dropna()

    start = request.args.get('start')
    end = request.args.get('end')

    if start:
        df = df[df['Date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['Date'] <= pd.to_datetime(end)]

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return send_file(
        io.BytesIO(buffer.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='filtered_data.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)
    
