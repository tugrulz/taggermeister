import mltg
from mltg import *


filepath1 = './tag_relevance.dat'
filepath2 = './tags.dat'
filepath3 = './movies.dat'

relevance_table = tag_relevance(filepath1)
tag_table = pd.read_csv(filepath2, sep='\t', header=None,
                    names=['TagID', 'Tag', 'NumTaggings'])


#movies_table = movies(filepath3)

# import pickle
# o1 = open('relevance.pkl', 'wb')
# o2 = open('tags.pkl', 'wb')
# o3 = open('movies.pkl', 'wb')
# pickle.dump(relevance_table, o1)
# pickle.dump(tag_table, o2)
# pickle.dump(movies_table, o3)

movies = pd.read_csv(filepath3, sep='\t', header=None,
                        names=['MovieID', 'Title', 'NumRatings'])

#print(movies_table)
#print(movies)

#print tag_table.columns.values
#print tag_table
row =  tag_table.loc[tag_table['Tag'] == 'bleak']

tagID =  row.iloc[0]['TagID']
numTag = row.iloc[0]['NumTaggings']
#print relevance_table['TagID'] == tagID
rows = relevance_table.loc[(relevance_table['TagID'] == tagID)] #& (relevance_table['TagRelevance'].values > 0.01)]
#print rows.shape
rows = rows.sort('TagRelevance', ascending = False)
#print rows['TagRelevance']
ids = rows.ix[:,'MovieID':'MovieID']
print ids.iloc[1]['MovieID']

#print relevance_table.shape
for i in range(numTag):
     print movies.loc[movies['MovieID'] == ids.iloc[i]['MovieID']]['Title'], rows.iloc[i]['TagRelevance']

print numTag

#    print movies.loc['MovieID' == ids[i]]
#    print movies.get_value(ids[i]-1, 1, True)
