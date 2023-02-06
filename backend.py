import pandas as pd
import numpy as np

import streamlit as st

#kmeans
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#surprise
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

models = ("Course Similarity",
          "User Profile",
          "Clustering",
          "Clustering with PCA",
          "KNN")
          # "NMF",
          # "Neural Network",
          # "Regression with Embedding Features",
          # "Classification with Embedding Features")

"""
idx_id_dict - dictionary of key:doc_index to value:doc_id
id_idx_dict - dictionary of key:doc_id to values:doc_index
sim_matrix - matrix where RC corresponds to the similiarity between two courses
             IBM was never clear about what this was.  Seems COSINE similiarity between 

"""

def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")

def load_course_genres():
    return pd.read_csv("course_genres.csv")

def load_courses():
    df = pd.read_csv("course_processed.csv")
    df['TITLE'] = df['TITLE'].str.title()
    return df

def load_bow():
    return pd.read_csv("courses_bows.csv")

def load_profiles():
    return pd.read_csv('profile_df.csv')

# Create course id to index and index to id mappings
def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(['doc_index', 'doc_id']).max().reset_index(drop=False)
    idx_id_dict = grouped_df[['doc_id']].to_dict()['doc_id']
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def add_new_ratings(new_courses, params):
    res_dict = {}
    if len(new_courses) > 0:
        # Create a new user id, max id + 1 in the ratings.csv file
        ratings_df=load_ratings()
        # Set new ID
        new_id=ratings_df['user'].max() + 1
        # Give each course selected a rating of 3
        users=[new_id] * len(new_courses)
        ratings=[3.0] * len(new_courses)
        # Store new user results into a new Dataframe 
        res_dict['user']=users
        res_dict['item']=new_courses
        res_dict['rating']=ratings
        user_df = pd.DataFrame(res_dict)
        #
        # The following ensures that the same user doesn't get added twice.
        #
        #convert the courses of the last entry in the ratings df to an array for comparison
        comp1=ratings_df[ratings_df['user']==max(ratings_df.user)]['item'].values
        
        #do the same for the new user
        comp2=user_df['item'].values
        
        #write to the dataframe only if these values are not equal
        if not np.array_equal(comp1,comp2):
            updated_ratings = pd.concat([ratings_df, user_df])
            updated_ratings.to_csv("ratings.csv", index=False)        
        else:
            new_id=new_id - 1
        
        profile=build_profile_vector(new_courses,new_id)
        
        params['profile']=profile
        params['new_user_id']=new_id
        params['user_df']=user_df
        
        return params

def build_profile_vector(courses,new_id):
    
    course_genres_df=load_course_genres()
    profile_df=load_profiles()
    
    profile=np.zeros(14) #empty profile series
    
    # Populate the new users profile vector, which indicates their interest in each genre.
    for course in courses:
        profile=profile + np.array(course_genres_df[course_genres_df['COURSE_ID']==course].iloc[0,2:])*3.0 
    
    cp=np.insert(profile, [0], new_id)    
    dft=pd.DataFrame(cp.reshape(1,-1),columns=profile_df.columns)
    
    #
    # Ensure we are not adding the same profile two times in a row.
    #

    if not np.array_equal(dft.iloc[-1,1:], profile_df.iloc[-1,1:]):
        updated_profiles=pd.concat([profile_df, dft])
        updated_profiles.to_csv('profile_df.csv', index=False)
       
    return profile    
    
def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    # Create a dictionary to store your recommendation results
    res = {} #res for recommendation result
    # First find all enrolled courses for user
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res

def generate_recommendation_scores_user_profile(params):
    
    users = []
    courses = []
    scores = []
    
    user_df=params['user_df']
    
    score_threshold = 0.6
    
    idx_id_dict, id_idx_dict=get_doc_dicts()
    profile_df=load_profiles()
    
    test_user_ids=[params['new_user_id']]
    all_courses = set(idx_id_dict.values())
    
    course_genres_df = load_course_genres()
        
    profile=params['profile']
    
    unselected_course_ids = all_courses.difference(set(user_df['item']))
    
    for user_id in test_user_ids:
        
        # get user vector for the current user id

        test_user_vector=profile
        
        # get the unknown course ids for the current user id
        enrolled_courses = list(user_df['item'])
        unknown_courses = all_courses.difference(enrolled_courses)
        unknown_course_df = course_genres_df[course_genres_df['COURSE_ID'].isin(unknown_courses)]
        unknown_course_ids = unknown_course_df['COURSE_ID'].values
        
        # user np.dot() to get the recommendation scores for each course
        unknown_course_genres=unknown_course_df.iloc[:,2:].values

        # np.dot is the same as the matrix product for appropriately sized arrays.
        
        recommendation_scores = np.dot(test_user_vector,unknown_course_genres.T)
        
        
        # Append the results into the users, courses, and scores list
        for i in range(0, len(unknown_course_ids)):
            score = recommendation_scores[i]
            # Only keep the courses with high recommendation score
            if score >= score_threshold:
                users.append(user_id)
                courses.append(unknown_course_ids[i])
                scores.append(score)
                
        res_df=build_results_df(users, courses, scores, params)        
        
    return res_df 
       
def top_courses(params, courses, res_df):
    
    if "top_courses" in params and params['top_courses'] <= len(courses):
            res_df=res_df.iloc[0:params['top_courses'],:]
    
    return res_df

def combine_cluster_labels(kmeans_ids, labels):
    labels_df = pd.DataFrame(labels)
    cluster_df = pd.merge(kmeans_ids, labels_df, left_index=True, right_index=True)
    cluster_df.columns = ['user', 'cluster']
    return cluster_df


# Model training
def train(model_name, params):
    # TODO: Add model training code here
    if model_name==models[0]: 
        pass
    elif model_name==models[2]:
        pass
    else:
        pass
        
def build_results_df(users, courses, scores, params):
    res_dict = {}
    res_dict['USER'] = users
    res_dict['COURSE_ID'] = courses
    res_dict['SCORE'] = scores
    res_df = pd.DataFrame(res_dict, columns=['USER', 'COURSE_ID', 'SCORE'])
    
    res_df=top_courses(params, courses, res_df.sort_values(by=['SCORE'], ascending=False)) 
    
    return res_df

# Prediction
def predict(model_name, params):   
    
    
    if model_name==models[0]: # Course Similarity model
        
        sim_threshold = 0.6
        
        if "sim_threshold" in params:
            sim_threshold = params["sim_threshold"] / 100.0
            
        idx_id_dict, id_idx_dict = get_doc_dicts()
        sim_matrix = load_course_sims().to_numpy()
        users = []
        courses = []
        scores = []
        
        #initialize list for iteration.  This is only to reuse the existing code
        current_user_id=[params['new_user_id']]
    
        for user_id in current_user_id:
            
            if model_name == models[0]:
                ratings_df = load_ratings()
                user_ratings = params['user_df']
                enrolled_course_ids = user_ratings['item'].to_list()
                res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
                for key, score in res.items():
                    if score >= sim_threshold:
                        users.append(user_id)
                        courses.append(key)
                        scores.append(score)
            
        res_df=build_results_df(users, courses, scores, params) 
        
        return res_df
    
    elif model_name==models[1]:# User Profile Model
        return generate_recommendation_scores_user_profile(params)
    
    elif model_name==models[2] or model_name==models[3]: # Kmeans Model
        
        #isolate data for this user
        user_df=params['user_df']
        
        #train the model on existing data.  Use it for labelling
        kmeans, cluster_df, params=train_cluster(model_name,params)
        
        #grab the profile vector for the current user
        profile=params['profile']
        
        #predict the label of the current user
        label=float(kmeans.predict(profile.reshape(1,-1)))
        
        #load the user/enrolled courses data
        test_users_df=load_ratings()[['user','item']]
        
        #label the rating data with clusters
        test_users_labelled=pd.merge(test_users_df, cluster_df, left_on='user', right_on='user')
        
        #keep only the data of the cluster of interest
        labelled_df=test_users_labelled[test_users_labelled['cluster']==label]

        #add a count column for aggregation
        labelled_df['count']=[1]*len(labelled_df)
        
        #aggregate the number of counts
        count_df=labelled_df.groupby(['item']).agg('count').sort_values(by='count',ascending=False)
        count_df=count_df[['count']]
        
        #drop the courses the user has already taken
        count_df.drop(labels=user_df['item'], errors='ignore',inplace=True)
           
        #list of courses and the number of times they appeared in cluster
        courses=list(count_df.index)
        scores=list(count_df['count'])

        #this is only required so build_results_df can be called
        users=[params['new_user_id']]*len(courses)
        
        #build a df containing recommendation results
        res_df=build_results_df(users, courses, scores, params) 
             
        return res_df
    
    elif model_name == models[4]:
        
        user_df=params['user_df']
        ratings_df=load_ratings()
        
        #load user profile vectors
        user_profile_df=load_profiles()
        
        # Read the course rating dataset with columns user item rating
        reader = Reader(
                line_format='user item rating', sep=',', skip_lines=1, rating_scale=(2, 3))

        course_dataset = Dataset.load_from_file("ratings.csv", reader=reader)
        
        #we will be using the full training set for the application
        trainset=course_dataset.build_full_trainset()
        
        #hyperparameter grid.  
        sim_options={'name': 'pearson', 'user_based': False, 'k':10, 'min_k':1}

        knn=KNNBasic(sim_options=sim_options)

        
        # - Train the KNNBasic model on the trainset, and predict ratings for the testset
        knn.fit(trainset)
        
        pd.DataFrame(knn.compute_similarities()).to_csv('sim_pearson.csv')
        # read in the data on the current user
        enrolled_courses=set(user_df.item)
        
        #set of all courses
        all_courses=set(ratings_df.item)
       
        #courses that user has not interacted with
        unknown_courses=all_courses.difference(enrolled_courses)
     
        #current user id
        user_id=params['new_user_id']
       
        courses=[]
        scores=[]
        users=[]
        predictions=[]
       
        #get a prediction for every course in the unknown courses
        for course in unknown_courses:
            predictions.append(knn.predict(uid=str(user_id),iid=course))
            
        
        for prediction in predictions:

            if not prediction.details['was_impossible']:
                scores.append(prediction.est)
                courses.append(prediction.iid)
                users.append(prediction.uid)
                
                   
        #build a df containing recommendation results
        res_df=build_results_df(users, courses, scores, params) 
             
        return res_df
    
   
def train_cluster(model_name, params):
    
    #load and scale user profile vectors
    user_profile_df=load_profiles()
    feature_names = list(user_profile_df.columns[1:])
    scaler = StandardScaler()
    user_profile_df[feature_names]=scaler.fit_transform(user_profile_df[feature_names])
    
    #separate features for model from user ids
    features=user_profile_df.loc[:, user_profile_df.columns != 'user']
    kmeans_ids=user_profile_df.loc[:, user_profile_df.columns == 'user']
    
    #transform the profile vector to fit the model
    profile_transformed=scaler.transform(params['profile'].reshape(1,-1))
    
    if model_name==models[3]:
        
         #initialize PCA object
        pca=PCA(params['npc'])
        
        #apply pca to the features for clustering
        features=pd.DataFrame(pca.fit_transform(features),columns=[f'PC{i}' for i in range(1, params['npc'] + 1)])
        
        #transform the user profile vector
        profile_transformed=pca.transform(profile_transformed)
        
        
    #fit KNN and label user data with the appropriate clusters
    kmeans=KMeans(n_clusters=params['cluster_no'])
    kmeans.fit(features)
    cluster_df=combine_cluster_labels(kmeans_ids, kmeans.labels_)
    
    #the following is to write the distribution of clusters to the screen if desired
    agg_clust=cluster_df.groupby('cluster').size().reset_index().rename({0:'instances'},axis=1)    
        
    params['profile']=profile_transformed
    
    return kmeans, cluster_df, params
    