import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import datetime as dt
import matplotlib.pyplot as plt
import altair as alt
from sklearn.utils import shuffle


#####################################################################
## Global variables
SIM_INPUT_MAXDATE = dt.date.today() + dt.timedelta(days=365*2)
SIM_INPUT_NUMERIC_LINSPACE = 4


#####################################################################
st.write('# Melbourne Housing Market')

st.write('## Introduction')

if st.checkbox('Objective'):
	st.write('''
			* Perform exploratory data analysis using [Jupyter notebook](https://jupyter.org/)
			* Data cleaning, feature engineering, and modelling
			* Build an app using [Streamlit](https://docs.streamlit.io/en/stable/)
			* Allow user to do exploratory analysis of a pre-selected suburb
			* Using the pre-trained model, allow user to estimate/forecast Housing price
		''')

if st.checkbox('Technical overview'):
	st.write('''
			* Data source: [Melbourne Housing Market](https://www.kaggle.com/anthonypino/melbourne-housing-market)
			* Import raw data
			* Data cleaning, missing data imputations, feature engineering
			* ML modelling (xgboost)
			* Model analysis (feature importance, explain the model using [SHAP](https://shap.readthedocs.io/en/latest/))
			* Save the data & trained model (used as an input to this app)
			* Build interactive visualisations (using altair package) to allow exploratory data analysis for a particular Suburb
			* Predict the House Price using the ML model
			* Display output for analysis
		''')

if st.checkbox('Simulation overview (sidebar)'):
	st.write('''
			* The raw data set only contains data for more than two years (between 2016-01-28 and 2018-10-03).
			* This has made the ML model to _not_ capture the influence of time of sale with the sale Price.
			* Therefore, the Simulation will be fixed to only estimate the date as of the max date in the available data set (i.e. 2018-10-03).
			* The user can select the desired suburb to analyse and simulate.
			* Once a suburb is selected:
			*	* The allowed input value for other parameters also set to the min & max from the data set according to the suburb.
			*	* Those parameters (type, building area, land size, etc) are defaulted to the mode/median value within the data set of the selected suburb.
			*	* Of course, the user is allowed to modify the parameters as desired for Price estimation.
			* Exploratory Data Analysis only shows transactions that took place at the selected suburb.
			* However, the ML model are trained using the entire data set.
		''')


#####################################################################

st.write('## Input Data & Pre-trained Model')

## Load raw data
@st.cache(persist=True)
def load_data():
	df = pd.read_csv('./input/df.csv')
	df['Suburb'] = df['Suburb'].str.capitalize()
	df['Type_string'] = df['Type'].map({
			'h': 'House, Cottage, Villa, Semi, Terrace'
			, 't': 'Townhouse'
			, 'u': 'Unit, Duplex'
		})
	df['Type_string_sub'] = df['Type_string'].apply(lambda x: x[:21])
	return df
df = load_data()
mtx_target = 'Price'
mtx_object = ['Date','Type','Suburb']
mtx_extra  = ['(Predict)','Type_string','Type_string_sub']
mtx_features = [c for c in df.columns if c not in [mtx_target]+mtx_object+mtx_extra]

## Create dimensions
df_date = pd.DataFrame(pd.date_range(df['Date'].min(), SIM_INPUT_MAXDATE), columns=['Date'])
df_date['Date_int'] = df_date['Date'].apply(lambda x: x.value/10**9)
df_date['Year'] = df_date['Date'].dt.year
df_suburb = df[['Suburb','Suburb_avg']].drop_duplicates()
df_type = df[['Type','Type_string']].drop_duplicates()
df_type['Type_int'] = df_type['Type'].map({'u':0, 't':1, 'h':2})

## Toggle raw data
if st.checkbox('Raw data'):
	excludes_columns = ['Date_int','Type_int','Suburb_avg','Type_string']
	st.write(df[[c for c in df if c not in excludes_columns]].head(200))

## Load pre-trained xgboost model from pickle
model = pickle.load(open('./input/model_xgboost.pkl','rb'))
if st.checkbox('Model'):
	st.write(model)

## Obtain the list of N suburbs with most transactions
@st.cache(persist=True)
def get_suburb_with_most_transactions(N=5):
	unqV, unqN = np.unique(df['Suburb'], return_counts=True)
	sorter = np.argsort(unqN)
	unqV, unqN = unqV[sorter], unqN[sorter]
	return list(unqV[-N:][::-1])
mostTransSuburbs = get_suburb_with_most_transactions(N=2)


#####################################################################

st.sidebar.write('# Simulaton (Forecast)')

## By default, we estimate the date as of latest date in the data
pDate = df['Date'].max()
st.sidebar.write('As of Date: %s'%(df['Date'].max()))

## Suburb to analyse
pSuburb = st.sidebar.selectbox(
	'Suburb'
	, df['Suburb'].sort_values().unique().tolist()
	, index = df['Suburb'].sort_values().unique().tolist().index('Richmond')
)

## The rest of the parameter values will be defaulted to the values based on the selected Suburb above
tmp_df_suburb = df[df['Suburb']==pSuburb]
pType = st.sidebar.radio(
	'Type'
	, tmp_df_suburb['Type_string_sub'].sort_values().unique().tolist()
	, index = tmp_df_suburb['Type_string_sub'].sort_values().unique().tolist().index(
				tmp_df_suburb['Type_string_sub'].mode().values[0])
)
pBuildingArea = st.sidebar.slider(
	'Building area'
	, value = int(np.percentile(tmp_df_suburb['BuildingArea'], 55))
	, min_value = int(np.percentile(tmp_df_suburb['BuildingArea'],  1))
	, max_value = int(np.percentile(tmp_df_suburb['BuildingArea'], 99))
)
pLandsize = st.sidebar.slider(
	'Land size'
	, value = int(np.percentile(tmp_df_suburb['Landsize'], 55))
	, min_value = int(np.percentile(tmp_df_suburb['Landsize'],  1))
	, max_value = int(np.percentile(tmp_df_suburb['Landsize'], 99))
)
pRooms2 = st.sidebar.slider(
	'Number of room(s)'
	, value = int(np.percentile(tmp_df_suburb['Rooms2'], 55))
	, min_value = int(np.percentile(tmp_df_suburb['Rooms2'],  1))
	, max_value = int(np.percentile(tmp_df_suburb['Rooms2'], 99))
)
pBathroom = st.sidebar.slider(
	'Number of bathroom(s)'
	, value = int(np.percentile(tmp_df_suburb['Bathroom'], 55))
	, min_value = int(np.percentile(tmp_df_suburb['Bathroom'],  1))
	, max_value = int(np.percentile(tmp_df_suburb['Bathroom'], 99))
)
pCar = st.sidebar.slider(
	'Garage capacity (number of cars)'
	, value = int(np.percentile(tmp_df_suburb['Car'], 55))
	, min_value = int(np.percentile(tmp_df_suburb['Car'],  1))
	, max_value = int(np.percentile(tmp_df_suburb['Car'], 99))
)

## The submit button
st.sidebar.write('----------')
BT_SIM_SUBMIT = st.sidebar.button('Submit')


#####################################################################

st.write('## Exploratory Data Analysis')

st.write('Suburb: %s'%(pSuburb))

cols_print = ['Date','BuildingArea','Distance','Suburb','Type_string','Landsize','Rooms2','Bathroom','Car','Price','(Predict)']

if st.checkbox('Click to show/hide visualisations'):

	st.write('''
			Note:
			* This is an interactive visualisations
			* The user is allowed to select data points in the _scatter plots_
			* Bar plots on the bottom-right will be updated accordingly
		''')

	## Plot suburb data
	tmp = tmp_df_suburb.copy()
	tmp['Date'] = pd.to_datetime(tmp['Date'])
	tmp['diff'] = tmp['(Predict)'] - tmp['Price']
	tmp['diff_log'] = np.log(np.abs(tmp['diff']))

	## Plot
	selector = alt.selection(type='interval', encodings=['x','y'])
	## Row 1
	points = alt.Chart(tmp).mark_point(filled=True, size=20, opacity=0.7).encode(
		    x=alt.X('Date:T', scale=alt.Scale(zero=False))
		    , y=alt.Y('Price', scale=alt.Scale(zero=False))
		    , color=alt.condition(selector,'Type_string_sub:N',alt.value('lightgray'))
			, tooltip=cols_print
		).properties(width=600, height=130).add_selection(selector)
	chart_row1 = (points)
	## Row 2 - Col 1
	scat_size = alt.Chart(tmp).mark_circle(size=20).encode(
			x='BuildingArea', y='Landsize'
		    , color=alt.condition(selector,'Type_string_sub:N',alt.value('lightgray'))
			, tooltip=cols_print
		).properties(width=330, height=330).add_selection(selector)#.transform_filter(selector)
	## Row 2 - Col 3
	bar_room = alt.Chart(tmp).mark_bar(opacity=0.5).encode(
		    x=alt.X('count()', title=None)
		    , y=alt.Y('Rooms2', bin=True, title='Room(s)') ## bin=alt.Bin(step=1)
		    , color=alt.Color('Type_string_sub:N')
		).properties(width=210, height=85).transform_filter(selector)
	bar_bathr = alt.Chart(tmp).mark_bar(opacity=0.5).encode(
		    x=alt.X('count()', title=None)
		    , y=alt.Y('Bathroom', title='Bathroom(s)') ## bin=alt.Bin(step=1)
		    , color=alt.Color('Type_string_sub:N')
		).properties(width=210, height=85).transform_filter(selector)
	bar_car = alt.Chart(tmp).mark_bar(opacity=0.5).encode(
		    x=alt.X('count()', title=None)
		    , y=alt.Y('Car', title='Garage capacity') ## bin=alt.Bin(step=1)
		    , color=alt.Color('Type_string_sub:N')
		).properties(width=210, height=85).transform_filter(selector)
	chart_row2_col2 = alt.vconcat(bar_room, bar_bathr)
	chart_row2_col2 = alt.vconcat(chart_row2_col2, bar_car)
	chart_row2 = alt.hconcat(scat_size, chart_row2_col2)
	## Combine charts
	charts = chart_row1
	charts = alt.vconcat(charts, chart_row2)
	charts = charts.configure_legend(orient='top')
	st.altair_chart(charts, use_container_width=False)



#####################################################################

st.write('## Simulation')

if BT_SIM_SUBMIT:

	## Generate input data points for prediction based on the parameter give in the sidebar
	p_Date = [pDate]
	ndata = len(p_Date)
	p_Landsize = [pLandsize] * ndata
	p_BuildingArea = [pBuildingArea] * ndata
	p_Rooms = [pRooms2] * ndata
	p_Bathroom = [pBathroom] * ndata
	p_Car = [pCar] * ndata
	p_Distance = [np.round(tmp_df_suburb['Distance'].mean(), 2)] * ndata
	p_Type = [pType] * ndata
	p_Suburb = [pSuburb] * ndata
	SIM_DATA = pd.DataFrame(
		np.array([p_Landsize, p_BuildingArea, p_Rooms, p_Bathroom, p_Car, p_Distance, p_Date, p_Type, p_Suburb]).T
		, columns = ['Landsize', 'BuildingArea', 'Rooms2', 'Bathroom', 'Car', 'Distance', 'Date', 'Type_string', 'Suburb'])
	SIM_DATA['Date'] = pd.to_datetime(SIM_DATA['Date'])
	SIM_DATA = pd.merge(SIM_DATA, df_date, how='left', on='Date')
	SIM_DATA = pd.merge(SIM_DATA, df_type, how='left', on='Type_string')
	SIM_DATA = pd.merge(SIM_DATA, df_suburb, how='left', on='Suburb')

	## Perform the prediction
	SIM_DATA['(Predict)'] = model.predict(SIM_DATA[mtx_features].values).round(0)
	SIM_DATA['Date'] = SIM_DATA['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))

	## Prepare dataframe for plotting
	df_plot = SIM_DATA.copy()
	df_plot['(DataType)'] = 'Estimate'
	df_plot = df_plot.append(tmp_df_suburb)
	df_plot['(DataType)'].fillna('Raw data', inplace=True)

	cols_print = [c for c in cols_print if c!='Price']
	st.write(shuffle(SIM_DATA[cols_print]).head(50).T)
else:
	st.info("Please input simulation parameters on the _sidebar_ and click the *Submit* button.")

