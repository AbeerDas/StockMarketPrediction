import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from keras.models import load_model

import matplotlib.pyplot as plt
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')

import streamlit as st

st.markdown(f"<h1>Stock Prediction using Long Short Term Model</h3>", unsafe_allow_html=True)
st.markdown(f"""<p><span style='color:#f7f5f5'><i>Dymanic program where an AI model predicts stock prices based on user-entered stock ticker, start date, and end date.
            In examining three different models, I showcase the use of diverse metrics to demonstrate strategies for improving overall model performance..</i></p>""", unsafe_allow_html=True)


profile = "https://media.licdn.com/dms/image/D5603AQHq-wKhXwCIOQ/profile-displayphoto-shrink_400_400/0/1689913637650?e=2147483647&v=beta&t=7Y2IMpEPoOXO6_Js8DnoetBAs2m64G4Sm3NpTyXlwy8"
profile_markdown = f'<img src="{profile}" alt="Image" width="40" style="border-radius: 50%; margin-right: 10px;"/>'
st.markdown(f"<p>{profile_markdown}<span style='color:#BCBDBC'>By: Abeer Das<p>", unsafe_allow_html=True)


profile = "https://media.licdn.com/dms/image/D5603AQHq-wKhXwCIOQ/profile-displayphoto-shrink_400_400/0/1689913637650?e=2147483647&v=beta&t=7Y2IMpEPoOXO6_Js8DnoetBAs2m64G4Sm3NpTyXlwy8"
profile_markdown = f'<div style="display: flex; justify-content: center; border-radius: 20px;"><img src="{profile}" alt="Image" width="40"/></div>'

userinput = st.text_input('Enter Stock Ticker', 'AAPL', help="You can enter your own stock ticker")

startinput = st.date_input('Start Date', pd.to_datetime('2021-01-14'), help="Enter your own start date")

endinput = st.date_input('End Date', pd.to_datetime('2021-12-31'), help="Enter an end date")



data = yf.download(userinput, start = startinput, end = endinput)

if data.empty:
    st.error("No data available for the specified date range.")
else:
    company_info = yf.Ticker(userinput)

    current_price = company_info.history(period='1d')['Close'].iloc[-1]
    rounded_current = round(current_price, 2)
    st.write(f"Current Price: {rounded_current}")

    data.dropna(inplace=True)

    data.reset_index(inplace = True)
    data.drop(['Volume', 'Adj Close', 'Date', 'High', 'Low', 'Open'], axis=1, inplace=True)

    data_set = data.iloc[:, 0:11]
    if data_set.empty:
        st.error("""Some of the technical indicators require more than 150 days worth of data. It will otherwise not compute sorry lol""")
    else:
        
        pd.set_option('display.max_columns', None)

        userinput = userinput.upper()
        company_name = company_info.info['longName']

        if company_name.endswith(', Inc.') or company_name.endswith(' Inc.'):
            company_name = company_name.rstrip(', Inc.')

        start_date_str = str(startinput)
        end_date_str = str(endinput)

        startYear = start_date_str[:4]
        endYear = end_date_str[:4]

        sc = MinMaxScaler(feature_range=(0,1))
        data_set_scaled = sc.fit_transform(data_set)

        X = []

        backcandles = 50
        if data_set_scaled.shape[0] >= backcandles:
            for j in range(1):
                X.append([])
                for i in range(backcandles, data_set_scaled.shape[0]):#backcandles+2
                    X[j].append(data_set_scaled[i-backcandles:i, j])

            #move axis from 0 to position 2
            X=np.moveaxis(X, [0], [2])

            X, yi =np.array(X), np.array(data_set_scaled[backcandles:,-1])
            y=np.reshape(yi,(len(yi),1))
        else:
            st.markdown("Not enought data")


        splitlimit = int(len(X)*0)
        X_train, X_test = X[:splitlimit], X[splitlimit:]
        y_train, y_test = y[:splitlimit], y[splitlimit:]

        loaded_model = load_model('StockAI_NTech.h5')
        y_pred = loaded_model.predict(X_test)

        st.markdown(f"<h3><span style='color:#64DCFF'>{company_name} Stock Price</span> compared to <span style='color:#FD00BD'>AI Prediction</span> ↴</h3>", unsafe_allow_html=True)

        y_test_inv = sc.inverse_transform(y_test)
        y_pred_inv = sc.inverse_transform(y_pred)

        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(y_test_inv, label='Test')
        ax.plot(y_pred_inv, label='AI Prediction')

        ax.legend()
        title_font = {'fontweight': 'bold', 'fontsize': 18, 'verticalalignment': 'bottom'}
        axis_label_font = {'fontsize': 14}

        ax.set_title('Actual vs Predicted Values', fontdict=title_font)
        ax.set_xlabel('Days Passed', fontdict=axis_label_font)
        ax.set_ylabel('Closing Value ($)', fontdict=axis_label_font)

        # Display the plot in Streamlit
        st.pyplot(fig)
        st.markdown("""<div style="text-align: center"><i>70/30 split between training and testing, 150 layered lstm model, 1 dense layer, 30 epochs, 
                    50 backcandles</i><br></div>""", unsafe_allow_html=True)
        
        data2 = yf.download(userinput, start = startinput, end = endinput)


        data2['Target'] = data2['Adj Close']-data2.Open
        data2['Target'] = data2['Target'].shift(-1)

        data2['TargetClass'] = [1 if data2.Target[i]>0 else 0 for i in range(len(data2))]

        data2['TargetNextClose'] = data2['Adj Close'].shift(-1)
        data_table = data2
        data2.dropna(inplace=True)
        data2.reset_index(inplace = True)
        data2.drop(['Close', 'Date'], axis=1, inplace=True)
        data_set2 = data2.iloc[:, 0:11]

        sc = MinMaxScaler(feature_range=(0,1))
        data_set_scaled2 = sc.fit_transform(data_set2)

        X2 = []
        
        if data_set_scaled2.shape[0] >= backcandles:
            for j in range(8):
                X2.append([])
                for i in range(backcandles, data_set_scaled2.shape[0]):#backcandles+2
                    X2[j].append(data_set_scaled2[i-backcandles:i, j])

            #move axis from 0 to position 2
            X2=np.moveaxis(X2, [0], [2])

            X2, yi2 =np.array(X2), np.array(data_set_scaled2[backcandles:,-1])
            y2 = np.reshape(yi2, (len(yi2), 1))

       
        X_test2 = X2
        y_test2 = y2

        # Ensure X_test2 and y_test2 have the same length
        X_test2 = X_test2[:len(y_test2)]

        loaded_modelv2 = load_model('StockAI.h5')
        y_pred2 = loaded_modelv2.predict(X_test2)
   
        st.markdown(f"<h3><span style='color:#64DCFF'></span>All<span style='color:#FD00BD'> technical </span>indicators measured ↴</h3>", unsafe_allow_html=True)
 
        fig2, ax2 = plt.subplots(figsize=(16, 6))
        ax2.plot(y_test2, label='Test')
        ax2.plot(y_pred2, label='AI Prediction')

        ax2.legend()

        ax2.set_title('Actual vs Predicted Values', fontdict=title_font)
        ax2.set_xlabel('Days Passed', fontdict=axis_label_font)
        ax2.set_ylabel('Value', fontdict=axis_label_font)

        # Display the plot in Streamlit
        st.pyplot(fig2)
        st.markdown("""<div style="text-align: center"><i>70/30 split between training and testing, 150 layered LSTM model, 1 dense layer, 30 epochs, 50 backcandles. 
        This time it uses a model that takes all tehnical indicators into account and just gives us a min/maxed scale value</i><br><br></div>""", unsafe_allow_html=True)


        #Actual Worse Model
        loaded_modelv2 = load_model('StockAIv2.h5')
        y_pred3 = loaded_modelv2.predict(X_test2)
   
        fig3, ax3 = plt.subplots(figsize=(16, 6))
        ax3.plot(y_test2, label='Test')
        ax3.plot(y_pred3, label='AI Prediction')

        ax3.legend()

        ax3.set_title('Actual vs Predicted Values', fontdict=title_font)
        ax3.set_xlabel('Days Passed', fontdict=axis_label_font)
        ax3.set_ylabel('Value', fontdict=axis_label_font)

        # Display the plot in Streamlit
        st.pyplot(fig3)
        st.markdown("""<div style="text-align: center"><i>This one does the same but 33/66 split between training and testing, 2 layered LSTM model, 
                    1 dense layer, 3 epochs, 50 backcandles</i><br><br></div>""", unsafe_allow_html=True)
        st.write(data_table.style.set_table_styles([{
        'selector': 'table',
        'props': [
        ('text-align', 'center'),
        ('margin', 'auto')
         ]
        }]))



# Financial Terms
st.markdown(f"<h3>Financial Terms Explained</h3>", unsafe_allow_html=True)
st.markdown(f"""
These were a couple of technical indicators used to train the model. 
I researched and retrieved them from the yfinance library. Here are some of the terms explained simply.
            
*<h5><span style='color:#D1D1D1'>RSI (Relative Strength Index)</span></h5>*
The [Relative Strength Index (RSI)]({"https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI#:~:text=Description,and%20oversold%20when%20below%2030."}) 
acts like a stock speedometer, measuring how fast prices change. It helps spot overpriced (overbought) 
or underpriced (oversold) stocks, indicating potential trend reversals. 
High RSI suggests overpricing, low RSI suggests underpricing, helping identify moments when a stock's trend might change.

*<h5><span style='color:#D1D1D1'>Exponential Moving Averages (EMAF, EMAM, EMAS)</span></h5>*
EMAF, EMAM, and EMAS are types of moving averages that provide a smoother view of stock prices over time.
EMAF gives more weight to recent prices, making it more responsive to short-term price changes.
Similar to EMAF, EMAM smoothens out stock prices but with a longer period, providing a smoother average.
EMAS uses an even longer period for a more extended-term trend indication.

*<h5><span style='color:#D1D1D1'>Target Calculation</span></h5>*
The target is a simple math problem: subtract the opening price of a stock from its adjusted closing price. 
This gives us a number that tells us how much the stock's value changed in a day. It's like 
figuring out if I made or lost money on a stock during the day.


*<h5><span style='color:#D1D1D1'>TargetClass</span></h5>*
TargetClass is a bit like a traffic light for stocks. It says, "<span style='color:#ABEBC6'>Green</span>" if the stock is likely to go up, and "<span style='color:#E74C3C'>Red</span>" if it's likely to go down. It's a simple way of predicting the future movement of a stock based on its recent behavior. This can be handy for making decisions about whether to buy or sell a stock. In technical terms, it is a binary classification label indicating whether 
the target price change is positive or negative.
""", unsafe_allow_html=True)

# Training Model
st.markdown(f"<h3>How I Trained the <span style='color:#FD00BD'>Model</h3>", unsafe_allow_html=True)
st.markdown(f"""
*<h5><span style='color:#D1D1D1'>Data Gathering and Preparation</span></h5>*
I began by collecting historical stock data using the yfinance library for a specific stock, 
such as Apple (AAPL). This data included various features, such as opening and closing prices, 
volume, and other technical indicators like Relative Strength Index (RSI) and 
Exponential Moving Averages (EMAF, EMAM, EMAS). This particular model looked into 50 backcandles.
This means we took data from the last 50 days to predict the next day's return. 

*<h5><span style='color:#D1D1D1'>Data Scaling</span></h5>*
To ensure consistency and improve model performance, I applied [Min-Max Scaling]({"https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/"}) 
to normalize the input features within a specific range (0 to 1).
""", unsafe_allow_html=True)

image_url = "https://media.licdn.com/dms/image/D4D12AQGP8LwyQQfojw/article-cover_image-shrink_600_2000/0/1669525006395?e=2147483647&v=beta&t=OerveEWmzNA6pi3FMfvT2EgAJxN4KzZvCz7fxbACD1I"
markdown_content = f'<div style="display: flex; justify-content: center;"><img src="{image_url}" alt="Image" width="200" style="border-radius: 5%;"/></div>'

st.markdown(markdown_content, unsafe_allow_html=True)
lstm_video = "https://www.youtube.com/watch?v=Z03f7Wu5a6A&t=40s&ab_channel=AnalyticsVidhya"
lstm_web = "https://www.analyticsvidhya.com/blog/2021/03/introduction-to-long-short-term-memory-lstm/"

st.markdown(f"""

*<h5><span style='color:#D1D1D1'>LSTM Model Architecture</span></h5>*
I chose to use a type of artificial neural network called Long Short-Term Memory (LSTM). LSTMs are particularly effective for sequence prediction tasks, making them suitable for time-series data like stock prices.
The model architecture consisted of an LSTM layer with 150 units, followed by a dense layer with one unit. 
The activation function used was linear, which is commonly employed for regression tasks.
This [video]( {lstm_video} ) explains LSTM at a high level. For more information you can also check out this [website]({lstm_web}).""",unsafe_allow_html=True)

image_url2 = "https://d2mk45aasx86xg.cloudfront.net/image17_11zon_2727417658.webp"
markdown_content2 = f'<span style="display: flex; justify-content: center; border-radius: 20px;"><img src="{image_url2}" alt="Image" width="400"/></span>'

st.markdown(markdown_content2, unsafe_allow_html=True)

web1 = "https://www.analyticsvidhya.com/blog/2023/09/what-is-adam-optimizer/#:~:text=The%20Adam%20optimizer,%20short%20for,Developed%20by%20Diederik%20P."

st.markdown(f"""<div style="text-align: center"><i>An LSTM (Long Short-Term Memory) model is like a 
            smart door with three gates: forget, input, and output. The forget gate decides what information from the past to keep or discard. 
            The input gate lets new information in. Finally, the output gate decides what information to share with the outside world. It's like having a 
            thoughtful gatekeeper that helps the model remember important things, learn new stuff, and share the right information at the right time.
            </i></div>

*<h5><span style='color:#D1D1D1'>Model Compilation and Training</span></h5>*
Before training the model, I compiled it using the [Adam optimizer]({web1}) 
and Mean Squared Error (MSE) loss function. Adam is an optimization algorithm commonly used in training neural networks. 
MSE is a measure of the average squared difference between predicted and actual values, making it suitable for regression problems.

The training process involved feeding the model with sequences of past stock data (X_train) 
and their corresponding target values (y_train). I used a batch size of 15 and trained the model for 30 [epochs](https://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning#). 
The dataset was shuffled, and a portion (30%) was reserved for validation to assess the model's performance during training.
""", unsafe_allow_html=True)

image_url3 ="https://algotrading101.com/learn/wp-content/uploads/2020/06/training-validation-test-data-set.png"
markdown_content3 = f'''<div style="display: flex; justify-content: center; border-radius: 20px;"><img src="{image_url3}" alt="Image" width="400"/></div>'''
st.markdown(markdown_content3, unsafe_allow_html=True)

st.markdown("""<div style="text-align: center"><i>Data set was split: 30% Testing, 70% Training, and 10% Validation within the Training</i></div>
        
*<h5><span style='color:#D1D1D1'>Model Evaluation</span></h5>*
After training, I loaded my pre-trained model and used it to make predictions on the test set (X_test). 
I compared the predicted values (y_pred) with the actual values (y_test) to evaluate the model's accuracy.
""",unsafe_allow_html=True)

def error_calculate(y_t, y_p):
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    return mae, r2, rmse

def make_table(model,mae, r2, rmse):
    table_data = {
        f'Model {model} Metrics': ['Mean Absolute Error (MAE)', 'R-squared (R2)', 'Root Mean Squared Error (RMSE)'],
        'Value': [mae, r2, rmse],
        'Preferred Value': ['0', '1', '0']
    }
    st.table(table_data)

mae, r2, rmse = error_calculate(y_test, y_pred)
make_table("1",mae,r2,rmse)

mae, r2, rmse = error_calculate(y_test2, y_pred2)
make_table("2", mae,r2,rmse)

mae, r2, rmse = error_calculate(y_test2, y_pred3)
make_table("3", mae,r2,rmse)


st.markdown("""
MAE: Smaller is better. It tells you the average difference between predictions and actual values.

R-squared (R2): Closer to 1 is better. It shows how well your model explains the variation in the data.

RMSE: Smaller is better. It's another way to measure prediction accuracy, giving more weight to larger errors.
""")


st.markdown(f"""
<h3>Pitfalls and <span style='color:#64DCFF'>Things to Look out for</span></h3>
The stock market is incredible volatile and a ton of factors go into predicting the prices of a singular share. If there was a perfectly mathematically sound way
of predicting the stock prices everytime, we would have a lot more billionaires. A better AI would do [sentimenet analysis]({"https://www.damcogroup.com/blogs/ai-in-stock-market-predicting-the-unpredictable-with-confidence"}) 
and  analyze news articles, companies’ financial reports, and social media conversations in real-time. 
That being said, you can still make a lot of positive returns from Stock predictors that are
purely based on technical indicators as it removes human bias.<br>""", unsafe_allow_html=True)

image_url4 ="https://miro.medium.com/v2/resize:fit:885/1*qfAQAcUQDjNnZpySY9QV8A.png"
markdown_content4 = f'''<div style="display: flex; justify-content: center; border-radius: 20px;"><img src="{image_url4}" alt="Image" width="400"/></div>'''
st.markdown(markdown_content4, unsafe_allow_html=True)

st.markdown("""<div style="text-align: center"><i>Not a great example but here we see a stock predictor that takes sentiment analysis into account</i></div>
<br>To train this model, I used Apple's stock over a decade as training data. This was picked because I think Apple's stock price fluctuation refelcts the behaviour
of most S&P 500 companies fairly well. However in doing so, we risk making worse predictions for stocks that are 
more volatile as not all stocks are as steady as Apple. <br><br>
            
One other thing to note is that we use 50 backcandles to predict the price. This is why when given two dates, it never takes into account the first 50 days
as that would not fit the data requirements needed to make a prediction. I could try to figure out a different way of modelling the learning so that this
is not a requirement. Given a better graphics card, I could also probably train a model with a larger amount of data as well as more epochs. I could also play around and try different 
amounts of layers, technical indicators, backcandles, etc. 
            
I had a lot of fun making this and learnt a lot more than I would if I just read and watched things around this topic as opposed to making something. 
I will list below a bunch of videos and websites I watched to make this possible. Also, email me know
if you see any ways this could be improved and I would love to integrate feedback! <br>
            
Email: abeerdas647@gmail.com

""", unsafe_allow_html=True)

st.markdown(f"""
<h3>Additional Resources</h3>
Another project I looked at : [Stock Price Prediction Using Machine Learning Project]({"https://www.projectpro.io/article/stock-price-prediction-using-machine-learning-project/571"})
<br>Highly reccomended playlist for beginners: [StatQuest]({"https://www.youtube.com/watch?v=zxagGtF9MeU&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&ab_channel=StatQuestwithJoshStarmer"})
""", unsafe_allow_html=True)

import os
image_file = 'ai_workspace.png'  
image_path = os.path.join(os.getcwd(), image_file)
