import base64

import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("House Price Predictor🏡")

from PIL import Image
image = Image.open('Home.jpg')
st.image(image, caption='Home Sweet Home')

#build
Build = st.selectbox('BuildQual',[1,2,3,4,5,6,7,8,9,10])

#Year
YearBuilt = st.selectbox('BuiltYear',[1950,1960,1970,1980,1990,1995,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010])

YearRemodAdd = st.selectbox('RenovationYear',[1950,1960,1970,1980,1990,1995,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010])

TotalBsmtSF = st.selectbox('BasementSurfaceArea',[400,600,800,1000,1200,1400])

Floor = st.selectbox('FirstFloorSurfaceArea',[400,600,800,1000,1200,1500,1800,2200,2800,3500,4000])

GrLivArea = st.selectbox('GroundArea',[600,900,1200,1500,1800,2100,2400,2700,3000,3300,3500])

FullBath = st.selectbox('Bath',[1,2,3])

TotRmsAbvGrd = st.selectbox('Rooms',[2,3,4,5,6,7,8,9,10,11,12,13,14])

GarageCars = st.selectbox('GarageCars',[0,1,2,3,4])

GarageArea = st.selectbox('GarageArea',[200,400,600,800,1000,1200,1400])

BedroomAbvGr = st.selectbox('Bedroom',[0,1,2,3,4,5,6,7,8])

KitchenAbvGr = st.selectbox('Kitchen', [0, 1, 2, 3])


#Categorical

MSZoning = st.selectbox('Zones',df['MSZoning'].unique())
Utilities = st.selectbox('Utilities',df['Utilities'].unique())
BldgType = st.selectbox('BuildingType',df['BldgType'].unique())
Heating = st.selectbox('Heating',df['Heating'].unique())
KitchenQual = st.selectbox('KitchenQuality',df['KitchenQual'].unique())
SaleCondition = st.selectbox('SaleCondition',df['SaleCondition'].unique())
LandSlope = st.selectbox('LandSlope',df['LandSlope'].unique())

if st.button('Predict Price'):
    # query
    query = np.array([Build, YearBuilt, YearRemodAdd, TotalBsmtSF, Floor,GrLivArea, FullBath, TotRmsAbvGrd, GarageCars, GarageArea,BedroomAbvGr,KitchenAbvGr , MSZoning, Utilities,BldgType, Heating, KitchenQual, SaleCondition, LandSlope])

    query = query.reshape(1,19)
    #st.title("The predicted price of this configuration is " + str((np.exp(pipe.predict(query)[0]))))
    st.title("The Predicted Price of House is "+str(int(pipe.predict(query)[0])))
