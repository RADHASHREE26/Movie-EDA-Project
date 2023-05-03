#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis On Popular Films Dataset Of Last 2 Decades

# ![films.jpg](attachment:films.jpg)

# ## About the Dataset:
# 
# This is a Kaggle Dataset. The dataset contains most 100 popular movies for each year in the interval 2003-2022.
# The Data is Ideal for Exploratory Data Analysis.
# Every single information has been collected by web scraping and can be found on iMDB.

# ## About the Project:
# 
# As the title signifies, this is an Exploratory Analysis on the "TOP 100 POPULAR MOVIES FROM 2003 TO 2022 IMDB" dataset.
# I have tried and derived interesting and meaningful insights from the available data.
# Observations like these might help filmmakers in deciding on factors while preparing for or releasing their next work.

# ## Downloading the Dataset
# 
# Working on a the "TOP 100 POPULAR MOVIES FROM 2003 TO 2022 IMDB" dataset

# In[2]:


get_ipython().system('pip install jovian opendatasets --upgrade --quiet')


# Let's begin by downloading the data, and listing the files within the dataset.

# In[3]:


# Change this
dataset_url = 'https://www.kaggle.com/datasets/georgescutelnicu/top-100-popular-movies-from-2003-to-2022-imdb' 


# In[4]:


import opendatasets as od
od.download(dataset_url)


# The dataset has been downloaded and extracted.

# Let us save and upload our work to Jovian before continuing.

# In[5]:


project_name = "zerotopandas-course-project-eda-movies"


# In[6]:


get_ipython().system('pip install jovian --upgrade -q')


# In[7]:


import jovian


# In[7]:


jovian.commit(project=project_name)


# ## Data Preparation and Cleaning
# 
# Cleaning and preparing the data for perfect analysis and visualizations.
# 
# 

# Importing numpy and pandas libraries for the operations to be performed.

# In[8]:


import numpy as np
import pandas as pd


# Reading the csv file

# In[9]:


movies_df=pd.read_csv("top-100-popular-movies-from-2003-to-2022-imdb/movies.csv")
movies_df


# Let's get some basic information about the rows and columns of the dataset.

# In[10]:


movies_df.info()


# Here we can see that the dataset is pretty clean other than some null values in the certificate column and 1 missing value in the ratings column.
# Since I am not going to use the certificate column for this EDA, I am dropping it.

# In[11]:


movies_df.drop(['Certificate'], axis=1, inplace=True)
movies_df


# We will now tailor the data as per our analytic requirements.

# Let's take a look at a sample from the dataset to understand the dataset better.

# In[12]:


movies_df.sample(5)


# Viewing the one row for which we have a null value in Rating column

# In[13]:


movies_df[movies_df['Rating'].isnull()]


# In[14]:


movies_df.describe()


# Since rating has a standard deviation of 0.9 i.e less than 1, I am considering it as a low standard deviation and I am replacing the one empty row in rating column with the mean of the whole column.

# In[15]:


movies_df.at[85,'Rating']=movies_df['Rating'].mean()
movies_df['Rating']=movies_df['Rating'].round(decimals=1)
movies_df.loc[85]


# In[16]:


movies_df.sample(10)


# Now, we can see that for some of the rows, the budget and income columns have the value Unknown. Replacing these with NaN.

# In[17]:


movies_df['Budget'].replace("Unknown", np.nan, inplace=True)
movies_df


# In[18]:


movies_df['Income'].replace("Unknown", np.nan, inplace=True)
movies_df


# Now changing the Runtime, Budget and Income columns to numeric datatype

# In[19]:


movies_df['Budget']=movies_df['Budget'].str.replace('$','').str.strip()
movies_df['Income']=movies_df['Income'].str.replace('$','').str.strip()
movies_df['Budget']=movies_df['Budget'].str.replace(',','').str.strip()
movies_df['Income']=movies_df['Income'].str.replace(',','').str.strip()
movies_df


# In[20]:


movies_df['Budget']=movies_df['Budget'].str.extract('(\d+)', expand=False)
movies_df['Budget']=pd.to_numeric(movies_df['Budget'])

movies_df['Income']=movies_df['Income'].str.extract('(\d+)', expand=False)
movies_df['Income']=pd.to_numeric(movies_df['Income'])

movies_df['Runtime']=movies_df['Runtime'].str.extract('(\d+)', expand=False)
movies_df['Runtime']=pd.to_numeric(movies_df['Runtime'])


# In[21]:


movies_df


# Checking for nan values:

# In[22]:


movies_df['Budget'].isna().sum()


# In[23]:


movies_df['Income'].isna().sum()


# In[24]:


movies_df.describe()


# In[25]:


movies_df.info()


# Since removing these such huge amount of columns that have unknown values would highly impact the analysis, I am replacing it with some normalization.
# 
# Since the distribution seems skewed and the standard deviation is quite high, I am using median instead of mean.

# In[26]:


movies_df['Budget']=movies_df['Budget'].replace(np.nan, movies_df['Budget'].median())
movies_df['Income']=movies_df['Income'].replace(np.nan, movies_df['Income'].median())
movies_df['Runtime']=movies_df['Runtime'].replace(np.nan, movies_df['Runtime'].median())
movies_df


# Taking another overview of the data before proceeding with further cleaning and tailoring.

# In[27]:


movies_df.info()


# Making a new dataframe where I am altering the columns with multiple values to be in a list format for further proceedings with the data.

# In[28]:


new_movies_df=movies_df.copy(deep=True)
new_movies_df


# In[29]:


new_movies_df['Directors']=new_movies_df['Directors'].str.split(",")

new_movies_df['Stars']=new_movies_df['Stars'].str.split(",")

new_movies_df['Genre']=new_movies_df['Genre'].str.split(",")

new_movies_df['Country_of_origin']=new_movies_df['Country_of_origin'].str.split(",")


# Lets take a look at the newly created dataset.

# In[30]:


new_movies_df


# Also let's take a look at our un-altered original dataset.

# In[31]:


movies_df.sample(20)


# For films with unknown filming location, putting country of origin as the filming location.

# In[32]:


new_movies_df['Filming_location']=new_movies_df['Filming_location'].mask(new_movies_df['Filming_location']=="Unknown", new_movies_df['Country_of_origin'].str[0])
movies_df['Filming_location']=new_movies_df['Filming_location']
new_movies_df


# Checking if the piece of code works properly for an instance where Filming location was initially unknown (as seen in the sample).

# In[33]:


movies_df.loc[301]


# We are done with cleaning and preparation of data. Now we can proceed for Exploratory Analysis with the data.

# In[34]:


import jovian


# In[35]:


jovian.commit()


# ## Exploratory Analysis and Visualization
# 
# Exploring the dataset for fun, interesting and meaningful insights.
# 
# 

# Let's begin by importing`matplotlib.pyplot` and `seaborn`.

# In[44]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# ### Showing the mean, sum and range for numeric columns (that make sense)

# In[36]:


movies_df.describe()


# In[37]:


movies_df['Budget'].sum()


# In[38]:


movies_df['Income'].sum()


# Subtracting minimum and maximum values of the columns to get the range of data.

# In[39]:


movies_df['Budget'].max()-movies_df['Budget'].min()


# In[40]:


movies_df['Income'].max()-movies_df['Income'].min()


# In[41]:


movies_df['Runtime'].max()-movies_df['Runtime'].min()


# ### Exploring distributions of numeric columns using histograms.

# ##### Checking distribution of Ratings

# In[45]:


# #movies_df['Rating'].hist(bins=10);
# cmap=get_cmap('viridis')
# n, bins, patches = plt.hist(movies_df['Rating'], bins=10, color=cmap((n-n.min())/(n.max()-n.min())))

plt.figure(figsize=(10,6))

plt.xlabel("Ratings")
plt.ylabel("#Films")
plt.title("Distribution of Range")

#plt.hist(movies_df['Rating'], bins=10);
plt.hist(movies_df['Rating'], bins=10, color='#1C2951');


# The above graph shows the distribution of Range. As we can see, a major chunk of the films tend to score a rating between 6 and 8.

# ##### Checking distribution of Budget

# Taking only upto a particular range such that graph isn't distorted due to outliers.

# In[48]:


plt.figure(figsize=(10,6))
range_min = 1.0e+01
range_max = 1.5e+08

plt.xlabel("Budget in millions")
plt.ylabel("#Films")
plt.title("Distribution of Budget")

plt.hist(movies_df['Budget'], bins=10, range=(range_min, range_max), color='#1C2951');


# The above graph shows the distribution of Budget. As we can see, a major chunk of the films have Budget less than half a millions.

# ##### Checking distribution of Income

# Again taking only upto a particular range such that graph isn't distorted due to outliers.

# In[49]:


#(movies_df['Income']).hist(bins=10)
plt.figure(figsize=(10,6))
range_min = 1.0e+01
range_max = 8.0e+08

plt.xlabel("Income in millions")
plt.ylabel("#Films")
plt.title("Distribution of Income")

plt.hist(movies_df['Income'], bins=10, range=(range_min, range_max), color='#1C2951');


# The above graph shows the distribution of Income. As we can see, most of the films have Income less than 3 million.

# 
# Now exploring the data for further interesting insights.

# ### Total no. of different genres and Movies distribution Genre-wise

# First viewing all the individual genres in which films are being made.

# In[46]:


m1_df=new_movies_df
m1_df


# In[47]:


unique_values = list(set([val.strip() for sublist in m1_df['Genre'] for val in sublist]))
unique_values


# In[48]:


len(unique_values)


# Therefore we have 20 different genres 

# Plotting below the Top 10 trending Genre combinations revenue-wise: the darkest color representing highest revenue and the lighter colors representing lower revenues.

# In[49]:


plt.figure(figsize=(12,8))
top_genres=movies_df['Genre'].value_counts()[:10]
cmap=plt.get_cmap('viridis')
plt.pie(top_genres, explode=np.full(10,0.1),autopct='%0.2f%%',shadow=True, labels=top_genres.index, colors=cmap(np.linspace(0, 1, len(top_genres))))
plt.title('Movies distribution Genre-wise');


# This shows that "Action, Adventure, Sci-Fi" genre has been the post popular genre combination in the last two decades contributing to approximately 15% of the total films made.

# ### Co-relation between Rating and Runtime of movies

# In[50]:


movies_df.info()


# In[51]:


movies_df['Runtime'].corr(movies_df['Runtime'])


# This shows that Runtime and Rating are perfectly correlated.

# Which might signify that too short films aren't interesting to audience as well as too lengthy films seem to bore people.

# In[50]:


plt.figure(figsize=(15,8))
colors = np.random.rand(2000)
plt.scatter(movies_df['Runtime'], movies_df['Rating'], c=colors, cmap='viridis')
plt.ylabel("Ratings")
plt.xlabel("Runtime")
plt.title("Relation between Runtime and Rating")
plt.colorbar();


# The clustered graph represents audience preference with particular rating and runtime.
# It shows most of the films have Rating between 5 and 8 and a Runtime between 75ms and 150ms.

# Let us save and upload our work to Jovian before continuing

# In[53]:


import jovian


# In[54]:


jovian.commit()


# ## Asking and Answering Questions
# 
# TODO - Asking and Answering Questions regarding the dataset.

# #### Q1: Who is the Director with highest rated movies for the past 2 decades and what is his most worked genre?

# Improvising the dataset to a new one to get the answer.

# In[52]:


dir_df=new_movies_df
dir_df=dir_df.explode("Directors")
dir_df


# Grouping by Rating and Income:

# In[53]:


q1_df=dir_df.groupby('Directors')['Rating','Income'].sum()
q1_df


# Sorting the dataset in descending to get the most rated Director:
# (Performing a mod operation on the Income to have values that are more workable and good for visualization)

# In[54]:


sorted_df1=q1_df.sort_values('Rating', ascending=False)

for i in range(0,10):
    sorted_df1['Income']=sorted_df1['Income']%1000000
#plt.bar(sorted_df1.index,sorted_df1['Rating'])
#sorted_df1
sorted_df1


# Thus we now know that **Ridley Scott** is the highest rated Director of the last 2 decades.

# Exploiting the dataset further to get Ridley Scott's most contributed genre:

# In[55]:


df1=dir_df[dir_df['Directors']=='Ridley Scott']
df1


# In[56]:


df2=df1.explode("Genre")
df2


# In[57]:


genre_counts = df2['Genre'].value_counts()
genre_counts


# In[58]:


plt.figure(figsize=(20,8))
plt.xlabel("Genres")
plt.ylabel("#Films")
plt.title("No. of films of each Genre by Ridley Scott in past 2 decades")
plt.bar(genre_counts.index, genre_counts.values, width=0.6, color='#1C2951');


# From the above plotting, we can conclude that the Director with highest box-office rating is Ridley Scott and the Genre he contributes most is Drama followed by Action.

# #### Q2: How is Revenue generated related to Rating of the movie?

# Finding the correlation using corr() function:

# In[59]:


movies_df['Rating'].corr(movies_df['Income'])


# The value shows very mild correlation between the Rating and Revenue generation of movies.

# In[61]:


plt.figure(figsize=(12,6))
#sorted_df1.Rating.plot(kind="bar", width=0.4)
#sorted_df1.Income.plot()
sns.scatterplot(x='Rating', y='Income', data=movies_df, hue='Month', palette='viridis')
plt.title("Income vs Rating")
plt.legend(loc='upper left', fontsize='small', ncol=2);


# Hence we might conclude that Rating of the movie does not define the Revenue earned from the movie.
# 
# Movies with reasonably high ratings are often times found with more moderate Income. 

# But we might also safely conclude from the plotting that movies with lower Ratings generally do not generate very high Revenues.

# #### Q3: What is the most popular filming locations?

# To find this we are first extracting the Filming locations into a dataframe and then counting number of occurences of each location.

# In[62]:


new_movies_df['Filming_location']


# In[63]:


q3_df=new_movies_df['Filming_location'].value_counts().head(10)
q3_df


# Now plotting for the Top 10 filming locations (as plotting all the locations would give a unnecessarily messy visualization).

# In[64]:


sns.barplot(x=q3_df.values,y=q3_df.index, palette='viridis')
plt.xlabel("No. of movies")
plt.ylabel("Filming Locations")
plt.title("Top 10 Filming Locations");


# We hence can infer that USA is the most popular Filming Location followed by Canada but USA leads any other country way too ahead.

# #### Q4: Which year, which month highest number of movies have been made?

# To get the answer to this question, I am deriving a new dataset where I am grouping the items based on Year and Month.

# In[65]:


movies_counts=new_movies_df.groupby([new_movies_df['Year'],new_movies_df['Month']]).size()
movies_counts


# In[66]:


new_movies_df


# Now extracting the values for the months that have maximum films produced, yearwise.

# In[67]:


q5_df=new_movies_df.groupby('Year')['Month'].value_counts().reset_index(name='Values')
q5_df


# In[68]:


max_movies=q5_df.groupby('Year')['Values'].idxmax()
q5_max_df = q5_df.loc[max_movies]
q5_max_df


# Plotting the graph for the same:

# In[69]:


plt.figure(figsize=(18,8))
sns.barplot(x=q5_max_df['Year'],y=q5_max_df['Values'], data=q5_max_df, palette='viridis')
ax = plt.gca()
for rect, label in zip(ax.patches, q5_max_df['Month']):
    height = rect.get_height()
    width = rect.get_width()
    x = rect.get_x()
    y = rect.get_y()
    ax.text(x + width / 2, y + height / 2, label, ha='center', va='center', rotation=90)
plt.title('Yearly Highest Movie producing Months');


# Surprisingly 2022 has been the most productive year in the industry.
# 
# Also the Yearly Highest Movie producing Months generally include the later parts of the year, namely October to December.

# #### Q5: What is the Budget-Income Relationship?

# In[70]:


plt.figure(figsize=(10,6))
plt.xlabel("Budget")
plt.ylabel("Income")
plt.title("Budget vs Income")
plt.scatter(x='Budget', y='Income', data=movies_df);


# Removing outliers for better visualization:

# In[71]:


new_data_df=movies_df[(movies_df['Budget']<1.5e+08) & (movies_df['Income']<8.0e+08)]


# Now plotting a better a graph for the same:

# In[72]:


plt.figure(figsize=(14,8))
plt.xlabel("Budget")
plt.ylabel("Income")
plt.title("Budget vs Income")
plt.scatter(x='Budget', y='Income', data=new_data_df, color='#003151');


# We can easily visualize here that better budget definitely contributes to better revenue generation by the film.

# Let us save and upload our work to Jovian before continuing.

# In[94]:


import jovian


# In[73]:


jovian.commit()


# ## Inferences and Conclusion
# 
# I have inferred and concluded on the questions and queries raised by me while working on this particular dataset.

# In[106]:


import jovian


# In[74]:


jovian.commit()


# ## References and Future Work
# 
# Reference -
# 1) For some coding related doubts reffered - https://datatofish.com/
# 2) For knowing more about pandas and it's functions in detail - https://pandas.pydata.org
# 3) For knowing more about matplotlib - https://matplotlib.org
# 4) For knowing more about seaborn - https://seaborn.pydata.org/
# 
# Future Work-
# I am planning to add more perspectives to this particular project in the future to ask and answer better questions and hence obtain further insights. Therefore, the project will be updated from time to time.

# In[77]:


import jovian


# In[ ]:


jovian.commit()


# In[ ]:




