# IBMOnlineCourseRecommender

This project is a Streamlit web application of recommender systems to recommend new courses after the user selects courses they find interesting.   

The project and larger dataset were part of the capstone requirement of IBM's Machine Learning Professional Certificate offered on Coursera.  

Interfacing with the GUI is described below:

1) Scroll through the list of available courses and select courses you have an interest in.  Note, this data was implemented as received from IBM.  There are effective duplicate courses that vary by one character in the title or the description.

2) Select your choice of model.  At a high level, the models work as follow:

	1) Course Similarity -> A similarity matrix is built from a textual analysis of the course titles and descriptions.  Recommendations are based on how similar the words describing the courses are to the words describing the selected courses.
	   The score is how similar the recommended course is to the most similar course in the users list.

	2) User Profile -> A user profile is made by tabulating the genres of the selected courses.  Unseen courses whose genres are in line with the user profile are recommended. 
	   The score is the result of the dot product of the user's profile vector and the recommended courses genre vector.

	3) Clustering -> A user profile is made by tabulating the genres of the selected courses. The KMeans algorithm is used to assign clusters to the whole dataset, including the current user.
	   The members of the user’s cluster are extracted and the courses are wrangled.  Any courses taken by members of this cluster not seen by the current user are recommended. 
	   The score is the number of times the course was taken by members of the cluster.

 	4) Clustering with PCA -> Identical workflow to the clustering algorithm, but the dimension of the user profile dataset is reduced with PCA.  Try this if clustering is too slow.

	5) KNN -> An item based KNN approach using the Pearson similarity metric between courses.

3)  Tune the hyperparameters.  All models allow you to choose how many results you would like to see.  	
I find it overwhelming to look at too many results at once.

4) Click “Recommend New Courses” to view the results.

