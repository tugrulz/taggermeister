from imdb import IMDb
from imdb import IMDbError
from os import listdir
import pickle

ia = IMDb()


id_dict = {}
fails = []

movies_f = open('movies.dat', 'rb')

movies = movies_f.readlines()

start = 3000
end = 4500

for i in range(start,end):
    name = movies[i].split('\t')[1]
    try:
        results = ia.search_movie(name)
        if (len(results) > 0):
            result = results[0]
            id_dict[name] = result.getID()
            print("Fetched : " + str(result))
        else:
            print('Filed to fetch ' + name)
            fails.append(name)
        print(str(end - i) + " to go.")
    except(IMDbError):
        print('Filed to fetch ' + name)
        fails.append(name)



output = open('data'+str(start)+'.pkl', 'wb')

pickle.dump(id_dict, output)

output2 = open('fails'+str(start)+'.pkl', 'wb')

pickle.dump(fails, output2)

movies_f.close()

print(fails)

output.close()