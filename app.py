import streamlit as s
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle

# Title
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

s.title('California Housing Price Prediction')
s.image('https://sdmntprnorthcentralus.oaiusercontent.com/files/00000000-9c44-622f-b6d8-e22873aabedc/raw?se=2025-08-23T06%3A48%3A05Z&sp=r&sv=2024-08-04&sr=b&scid=1b8221be-5972-57a2-b0b7-bc4eadd9a778&skoid=38550de5-1fab-49d1-9ebb-83af5557cc43&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-08-22T23%3A03%3A36Z&ske=2025-08-23T23%3A03%3A36Z&sks=b&skv=2024-08-04&sig=FXSsKe%2B2cp0HC7Ldw7VOx%2Bxst9y3ZqFH6k1yc%2BkNoDY%3D')

s.header('Model of housing prices to predict median house values in California.',divider=True)

# s.header('''User must enter given values to Predict Price
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

s.sidebar.title('Select House Features🏠')

s.sidebar.image('https://images.pexels.com/photos/462358/pexels-photo-462358.jpeg?cs=srgb&dl=architectural-design-architecture-blue-sky-462358.jpg&fm=jpg')

# read_data
temp_df = pd.read_csv('california.csv')


random.seed(12)

all_values=[]
for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var = s.sidebar.slider(f'select {i} value',int(min_value),int(max_value),random.randint(int(min_value),int(max_value)))


    all_values.append(var)


ss = StandardScaler()
ss.fit(temp_df[col])
final_value = ss.transform([all_values])

 
# s.write(final_value)
with open('house_price_pred_ridge_model.pkl','rb') as f:
    chatGPT= pickle.load(f)


price = chatGPT.predict(final_value)[0]
# s.write(price)


import time


s.write(pd.DataFrame(dict(zip(col,all_values)),index =[1]))


progress_bar = s.progress(0)
placeholder = s.empty()
placeholder.subheader('Predicting Price')
place = s.empty()
place.image('https://cdn-icons-gif.flaticon.com/11677/11677497.gif',width = 50)

if price>0:
    progress_bar = s.progress(9)
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
        
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
    # s.subheader(body)
    placeholder.empty()
    place.empty()
    s.success(body)
else:
    body = 'Invalid House features Values'
    s.warning(body)



s.markdown('Designed by: **Narender**')
