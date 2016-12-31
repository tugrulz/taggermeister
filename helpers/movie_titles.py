from imdb import IMDb
from os import listdir
import pickle

ia = IMDb()

years = listdir('/media/tugrulz/Yeni Birim/ml data/en/OpenSubtitles2016/xml/en')

id_dict = {}
fails = []

for year in years:
    titles = listdir('/media/tugrulz/Yeni Birim/ml data/en/OpenSubtitles2016/xml/en/' + year)
    for title in titles:
        zero_count = 7 - len(title);
        q_title = title;
        for i in range(zero_count):
            q_title = '0' + q_title
        movie = ia.get_movie(q_title)
        if (movie.data.has_key('title')):
            print('ID: ' + title + ' Movie Name: ' + movie['title'])
            id_dict[title] = movie['title']
        else:
            print('Failed to retrieve movie of id ' + title)
            fails.append(title)


output = open('titles.pkl', 'wb')

pickle.dump(id_dict, output)

print(fails)

output.close()