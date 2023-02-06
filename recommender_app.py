import streamlit as st
import pandas as pd
import time
import backend as backend

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid import GridUpdateMode, DataReturnMode




# Basic webpage setup
st.set_page_config(
   page_title="Course Recommender System",
   layout="wide",
   initial_sidebar_state="expanded",
)

# 
# Streamlit run recommender_app.py

# ------- Functions ------
# Load datasets
@st.cache
def load_ratings():
    return backend.load_ratings()


@st.cache
def load_course_sims():
    return backend.load_course_sims()


@st.cache
def load_courses():
    return backend.load_courses()


@st.cache
def load_bow():
    return backend.load_bow()

# MM added

@st.cache
def load_genres():
    return backend.load_course_genres()

@st.cache
def load_profiles():
    return backend.load_profiles()

# Initialize the app by first loading datasets
def init__recommender_app():

    with st.spinner('Loading datasets...'):
        ratings_df = load_ratings()
        sim_df = load_course_sims()
        course_df = load_courses()
        course_bow_df = load_bow()
        #added by MM
        

    # Select courses
    st.success('Datasets loaded successfully...')

    st.markdown("""---""")
    st.subheader("Select courses that you have audited or completed: ")

    # Build an interactive table for `course_df`
    gb = GridOptionsBuilder.from_dataframe(course_df)
    gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    gb.configure_side_bar()
    grid_options = gb.build()

    # Create a grid response
    response = AgGrid(
        course_df,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
    )

    results = pd.DataFrame(response["selected_rows"], columns=['COURSE_ID', 'TITLE', 'DESCRIPTION'])
    results = results[['COURSE_ID', 'TITLE']]
    st.subheader("Your courses: ")
    st.table(results)
    return results


def train(model_name, params):
    
    if model_name == backend.models[0]:
        # Start training course similarity model
        params['enrolled_course_ids'] = selected_courses_df['COURSE_ID']
        
        with st.spinner('Training...'):
            time.sleep(0.5)
            backend.train(model_name,params)
            
        st.success('Done!')
    # TODO: Add other model training code here
    elif model_name == backend.models[2]:
        
        params['enrolled_course_ids'] = selected_courses_df['COURSE_ID']
        
        with st.spinner('Training...'):
            time.sleep(0.5)
            kmeans= backend.train(model_name,params)
            
        st.success('Done!')
        
    else:
        st.success('Done!')


def predict(model_name, params):
    res = None
    # Start making predictions based on model name, test user ids, and parameters
    with st.spinner('Generating course recommendations: '):
        time.sleep(0.5)
        res = backend.predict(model_name, params)
    st.success('Recommendations generated!')
    return res

def show_train_button():
    # Training
    st.sidebar.subheader('3. Training: ')
    training_button = st.sidebar.button("Train Model")
    training_text = st.sidebar.text('')
    # Start training process
    if training_button:
        train(model_selection, params)
        
    pred_head=4

    
    return pred_head


# ------ UI ------
# Sidebar
st.sidebar.title('Personalized Learning Recommender')
# Initialize the app
selected_courses_df = init__recommender_app()

# Model selection selectbox
st.sidebar.subheader('1. Select recommendation models')
model_selection = st.sidebar.selectbox(
    "Select model:",
    backend.models
)

# Hyper-parameters for each model
params = {}
st.sidebar.subheader('2. Tune Hyper-parameters: ')

#sidebar header for prediction

pred_head = 3

MaxReturnedCourses=50

# Course similarity model
if model_selection == backend.models[0]:
    
    # Add a slide bar for selecting top courses
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=MaxReturnedCourses,
                                    value=10, step=1)
    
    # Add a slide bar for choosing similarity threshold
    course_sim_threshold = st.sidebar.slider('Course Similarity Threshold %',
                                             min_value=0, max_value=100,
                                             value=50, step=10)
    params['top_courses'] = top_courses
    params['sim_threshold'] = course_sim_threshold
    
    
# TODO: Add hyper-parameters for other models
# User profile model
elif model_selection == backend.models[1]:
    
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=MaxReturnedCourses,
                                    value=10, step=1)
    
    
    params['top_courses'] = top_courses
    
# Clustering model
elif model_selection == backend.models[2] or model_selection == backend.models[3]:
    
    cluster_no = st.sidebar.slider('Number of Clusters',
                                   min_value=0, max_value=50,
                                   value=16, step=1)
    
    params['cluster_no'] = cluster_no
    
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=MaxReturnedCourses,
                                    value=10, step=1)
    
    params['top_courses'] = top_courses
    
    if model_selection == backend.models[3]:
        
        npc = st.sidebar.slider('Principal Components',
                                        min_value=0, max_value=14,
                                        value=9, step=1)
        
        params['npc'] = npc 
        
        
        ncomp=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        evr=[0.288, 0.463, 0.576, 0.649, 0.719, 0.788, 0.843,
              0.894, 0.927, 0.954, 0.973, 0.987, 0.998, 1.0]
        
        evr_df=pd.DataFrame({'Number of Components': ncomp[2:10], "Explained Varience Ratio": evr[2:10]})
        
        st.sidebar.table(evr_df)
     
elif model_selection == backend.models[4]:
    
    top_courses = st.sidebar.slider('Top courses',
                                    min_value=0, max_value=MaxReturnedCourses,
                                    value=10, step=1)
    
    
    params['top_courses'] = top_courses
       
else:
    pass

# Prediction
st.sidebar.subheader(f'{pred_head}. Prediction')

# Start prediction process
pred_button = st.sidebar.button("Recommend New Courses")
if pred_button and selected_courses_df.shape[0] > 0:
    # Create a new id for current user session
    params = backend.add_new_ratings(selected_courses_df['COURSE_ID'].values, params)
    
    user_df=params['user_df']
    
    res_df = predict(model_selection, params)
    res_df = res_df[['COURSE_ID', 'SCORE']]
    course_df = load_courses()
    res_df = pd.merge(res_df, course_df, on=["COURSE_ID"])#.drop('COURSE_ID', axis=1)
    st.table(res_df)
    
elif pred_button and selected_courses_df.shape[0] == 0:
    assert 1==0, 'No Courses Selected!'
