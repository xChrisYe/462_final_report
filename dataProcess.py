# -*- coding: utf-8 -*-
"""
Created on Wed May 12 22:36:03 2021

@author: xChrisYe
"""
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import seaborn as sns
from replay import rePlayBaseModel,AB_TEST, epsilonGreddy,UCB,thompsonSample
from pandas import DataFrame

# import data
def getData():
    fpath_rating = os.path.join(os.path.dirname(__file__),  'data', 'ratings.csv')
    fpath_link = os.path.join(os.path.dirname(__file__),  'data', 'links.csv')
    fpath_movie = os.path.join(os.path.dirname(__file__), 'data', 'movies.csv')
    fpath_tag = os.path.join(os.path.dirname(__file__),  'data', 'tags.csv')
    
    data_rating = pd.read_csv(fpath_rating, header=0, usecols=['userId','movieId', 'rating'])
    #data_link = pd.read_csv(fpath_link, header=0, usecols=['movieId', 'imdbId'])
    data_movie = pd.read_csv(fpath_movie, header=0)
    #data_tag = pd.read_csv(fpath_tag, header=0,usecols=['userId', 'movieId', 'tag'])
    
    #return data_rating, data_link, data_movie, data_tag
    return data_rating,  data_movie

# clean title (remove the year) 
def cleaningTitle(title):
    try :
        return title[:title.rindex('(')]
    except:
        return title
    
# add title into data according to the movie_id    
#def processData(data_rating, data_link, data_movie, data_tag):
def processData(data_rating,  data_movie, ):
    data_movie['title'] = data_movie['title'].apply(lambda tcol : cleaningTitle(tcol))
    return pd.merge(data_rating, data_movie,left_on='movieId', right_on='movieId')[['userId', 'movieId', 'rating', 'title']] 

def plotRatio(movie):
    df_1 =movie[(movie['counts']>=63000)]['counts'].sum()
    
    df_2 =movie[ (movie['counts']<63000)&(movie['counts']>=30000)]['counts'].sum()
    
    df_3 =movie[ (movie['counts']<30000)&(movie['counts']>=20000)]['counts'].sum()
  
    df_4 =movie[(movie['counts']<20000)]['counts'].sum()
        
    explode = (0,0,0,0.05)
    plt.figure(figsize=(6,8))
    frames = [df_1, df_2, df_3,df_4]
    lable_list =['>63000','30000-63000','20000-30000','<20000']
    colors = ['#99FFFF', '#FFCCCC', '#FF9900','#33CC33']
    patches, l_text, p_text = plt.pie(frames,explode=explode,colors =colors,  labels=lable_list, labeldistance=1.1, autopct="%1.1f%%", shadow=False, startangle=90, pctdistance=0.8)
    
    for t in l_text:
        t.set_size(20)
    for t in p_text:
        t.set_size(15)
    plt.axis("equal") 
    plt.legend(bbox_to_anchor=(1.02, 0.15), loc=2, labelspacing=-2.3,frameon=False)
    plt.title('Percentage of the number of reviews')
    plt.show()
   
    movie_counts_3 = movie[movie['counts']>=20000]
    plt.title('the number of movie with review >=20000')
    plt.hist(movie_counts_3['counts'], histtype='bar', rwidth=0.1)
    plt.xlabel('number of review')
    plt.ylabel('frequent')
    plt.show()   
 
# describe the star for each movie    
def plotRating(df):
    sns.set(style='darkgrid',font_scale=1.5)
    k = df.groupby(['title','rating']).size().reset_index(name='counts').sort_values(by=['counts'],ascending=False)

    rating_v_count_ax = sns.factorplot(data=k, x='rating', y='counts', hue='title',  size=15, aspect=1.0, kind='bar', legend=True)

# transform the star to whether user like or not.     
def transform(rating):
    if rating>=4.0:
        return 1
    else:
        return 0
    
# plot the picture of how its replay process within different algorithms    
def plotLine(random_Result,ab_result_1k_Result,ab_result_3k_Result,greddy_1_Result,greddy_2_Result,greddy_3_Result,UCB_Result,thompsonSample_Result):
    
    # calculate the mean in N times iterations
    random = DataFrame(random_Result).groupby(['index'], as_index=False).mean()
    ab_result_1k =ab_result_1k_Result.groupby(['index'], as_index=False).mean()
    ab_result_3k =ab_result_3k_Result.groupby(['index'], as_index=False).mean()
    greddy_1 = greddy_1_Result.groupby(['index'], as_index=False).mean()
    greddy_2 = greddy_2_Result.groupby(['index'], as_index=False).mean()
    greddy_3 = greddy_3_Result.groupby(['index'], as_index=False).mean()
    UCB_df = UCB_Result.groupby(['index'], as_index=False).mean()
    thompson_df = thompsonSample_Result.groupby(['index'], as_index=False).mean()
    
    
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set(font_scale=2.5)

    #sns.palplot(color_15)
    fig, ax = plt.subplots(figsize=(15,12))
    
    for (avg_results_df, style,name) in [(random,'k-','radom'),
                                    (ab_result_1k, 'r-','AB Test_1k'),
                                    (ab_result_3k, 'r--','AB Test_3k'),
                                    (greddy_1, 'b-','$epsilon$ = 0.05'),
                                    (greddy_2, 'b--','$epsilon$ = 0.1'),
                                    (greddy_3, 'b-.','$epsilon$ = 0.2'),
                                    (UCB_df, 'g-','UCB'),
                                    (thompson_df,'y-','thompson')
                                ]:

        #avg_results_df_used  = avg_results_df.query('index <= 2000')
        #ax.plot(avg_results_df_used.index, avg_results_df_used.fraction_relevant, style, linewidth=3.5)
        ax.plot(avg_results_df.index, avg_results_df.fraction_relevant, style,label = name, linewidth=3.5)
    
    # add a line for the optimal value -- 0.5575 for Star Wars (from exploration noteboook)
    #ax.axhline(y=0.5575, color='k', linestyle=':', linewidth=2.5)
    ax.legend(fontsize='20', title_fontsize='20')
    #plt.legend(loc="upper right")   # 与plt.legend(loc=1)等价
    ax.set_title('Percentage of Liked Recommendations')
    ax.set_xlabel('Recommendation process')
    ax.set_ylabel('Percentage of Liked')
    
    ax.set_xticks(range(0,26000,5000))
    #ax.set_xticks(range(0,2000,500))
    ax.set_ylim(0.2, 0.6)
    ax.set_yticks(np.arange(0.2, 0.8, 0.1))
    
    # rescale the y-axis tick labels to show them as a percentage
    ax.set_yticklabels((ax.get_yticks()*100).astype(int))
    
    plt.tight_layout()
    plt.show()

# show the process of recommendation
def plotRecommendation(movie_list,movie_titles,result,name):
    
    color_15 = ['#99FFFF', '#FFCCCC', '#CCCCCC','#33CC33','#FFFF99','#FFCC99','#00FFFF','#009999',
                '#FFFF00','#FF66CC','#3399FF','#FF3300','#FF9900','#333399','#6666FF']

    color_map = dict(zip(movie_list, color_15))
    
    result['movieSelect']=1                              
    result_ttl = result[result.iteration==10]\
                                    .pivot(index='index', columns='movieId', values='movieSelect')\
                                    .fillna(0)\
                                    .cumsum(axis=0)\
                                    .reset_index()\
                                    .rename(columns=movie_titles)
    
    return_resutl = result_ttl.values[-1,1:]
    
    result_ttl.iloc[:,1:] = result_ttl.iloc[:,1:].div((result_ttl.index + 1)/100, axis=0)

    fig, ax = plt.subplots(figsize=(15,14))
    
    ax.stackplot(result_ttl.index,
                 result_ttl.iloc[:,1:16].T,
                 labels=result_ttl.iloc[:,1:16].columns.values.tolist(),
                 colors=[color_map[x] for x in result_ttl.iloc[:,1:].columns.values]
                )
    
    ax.set_xlim(0,1000)
    ax.set_xticks(range(0, 27000, 5000))
    
    ax.set_title('%s Algorithm'%name)
    ax.set_xlabel('Recommendation test times')
    ax.set_ylabel('Recommendation success ratio')
    
    lgd = plt.legend(bbox_to_anchor=(1.02, 0.15), loc=2, labelspacing=-2.3,frameon=False)
    lgd.get_frame().set_facecolor('w')
    ax.set_facecolor('w')
    
    plt.tight_layout()
    plt.show()
    return return_resutl

# show the result of reccomendation
def plotRecomResult(movie_titles,recomm_result_random,recomm_result_ab,recomm_result_greddy,recomm_result_ucb,recomm_result_ts):
    color_15 = ['#99FFFF', '#FFCCCC', '#CCCCCC','#33CC33','#FFFF99','#FFCC99','#00FFFF','#009999',
                '#FFFF00','#FF66CC','#3399FF','#FF3300','#FF9900','#333399','#6666FF']
    algorithmList=['random','ab_test','greedy','UCB','thompsonSample']
    bars=[recomm_result_random.tolist(),recomm_result_ab.tolist(),recomm_result_greddy.tolist(),recomm_result_ucb.tolist(),recomm_result_ts.tolist()]
    
    plt.figure(figsize=(6,6.5))
    df = pd.DataFrame(np.asarray(bars),index=algorithmList, columns=list(movie_titles.values()))
    df.plot.bar(stacked=True,color=color_15)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,frameon=False,fontsize='xx-small')
    lgd.get_frame().set_facecolor('w')
    plt.show()


if __name__ == "__main__":
    #data_rating, data_link, data_movie, data_tag=getData()
    #df = processData(data_rating, data_link, data_movie, data_tag)
    data_rating,data_movie=getData()
    df = processData(data_rating,  data_movie)
    
    #movie_counts = df.groupby('movieId')
    #movie_counts_1 =  df.groupby(['movieId','rating']).size().reset_index(name='counts')
    movie_counts =  df.groupby(['movieId']).size().reset_index(name='counts').sort_values(by=['counts'],ascending=False)

    plotRatio(movie_counts)


    movieIds = movie_counts[:15]['movieId'] 
    df = df[df['movieId'].isin(movieIds)]
    plotRating(df)
    #df['title'] = df['title'].apply(lambda tcol : cleaningTitle(tcol))
    #df = trasformToLiked(df,4)
    reward_threshold =4
    df['liked'] = df['rating'].apply(lambda rating : 1 if rating>=reward_threshold else 0)

    #sns.FacetGrid(tips,X col = 'size',  row = 'smoker', hue = 'day')
    ct = pd.crosstab(df.title, df.liked)
    ct.plot.bar(stacked=True)
    plt.legend(title='liked')
    plt.show()
    
    #solve data
    #fpath_saving = os.path.join(os.path.dirname(__file__),  'data', 'movie_rating.csv')
    #df.to_csv(fpath_saving)

      
    # this is the code how to process the replay, which cost much time, 
    # instead we use the output file as input to analysis the recommendation process
    """
    random_Result = rePlayBaseModel(data = df).replay()
    DataFrame(random_Result).to_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'random_Result.csv'))
    
    ab_result_1k_Result = AB_TEST(data = df,n_test = 1000).replay()
    DataFrame(ab_result_1k_Result).to_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'ab_result_1k_Result.csv'))

    
    ab_result_3k_Result = AB_TEST(data = df,n_test = 3000).replay()
    DataFrame(ab_result_3k_Result).to_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'ab_result_3k_Result.csv'))
      
    greddy_1_Result = epsilonGreddy(data= df,epsilon=0.05).replay()
    DataFrame(greddy_1_Result).to_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'greddy_1_Result.csv'))
    
    greddy_2_Result = epsilonGreddy(data= df,epsilon=0.1).replay()
    DataFrame(greddy_2_Result).to_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'greddy_2_Result.csv'))
    
    greddy_3_Result = epsilonGreddy(data= df,epsilon=0.2).replay()
    DataFrame(greddy_3_Result).to_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'greddy_3_Result.csv'))
    
    UCB_Result = UCB(data= df).replay()
    DataFrame(UCB_Result).to_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'UCB_Result.csv'))
    
    thompsonSample_Result = thompsonSample(data= df).replay()
    DataFrame(thompsonSample_Result).to_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'thompsonSample_Result.csv'))
    
    """
    movie_list = df.title.sort_values().unique().tolist()
    
    movie_titles = df.groupby(['movieId','title']).size().to_frame() \
                                    .reset_index('title').title \
                                    .to_dict()
                           
    # import output file as data
    random_Result =  pd.read_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'random_Result.csv'), header=0,index_col=0)
    ab_result_1k_Result =  pd.read_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'ab_result_1k_Result.csv'), header=0,index_col=0) 
    ab_result_3k_Result =  pd.read_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'ab_result_3k_Result.csv'), header=0,index_col=0)
    greddy_1_Result =  pd.read_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'greddy_1_Result.csv'), header=0,index_col=0)
    greddy_2_Result =  pd.read_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'greddy_2_Result.csv'), header=0,index_col=0)
    greddy_3_Result =  pd.read_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'greddy_3_Result.csv'), header=0,index_col=0)
    UCB_Result =  pd.read_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'UCB_Result.csv'), header=0,index_col=0)
    thompsonSample_Result =  pd.read_csv(os.path.join(os.path.dirname(__file__),  'data','output', 'thompsonSample_Result.csv'), header=0,index_col=0)
    
    plotLine(random_Result,ab_result_1k_Result,ab_result_3k_Result,greddy_1_Result,greddy_2_Result,greddy_3_Result,UCB_Result,thompsonSample_Result)
    
    recomm_result_random = plotRecommendation(movie_list,movie_titles,random_Result,'random')
    recomm_result_ab = plotRecommendation(movie_list,movie_titles,ab_result_3k_Result,'ab_test')
    recomm_result_greddy = plotRecommendation(movie_list,movie_titles,greddy_3_Result,'greedy')
    recomm_result_ucb = plotRecommendation(movie_list,movie_titles,UCB_Result,'ucb')
    recomm_result_ts = plotRecommendation(movie_list,movie_titles,thompsonSample_Result,'thompsonSample')
    
    plotRecomResult(movie_titles,recomm_result_random,recomm_result_ab,recomm_result_greddy,recomm_result_ucb,recomm_result_ts)

    
    

