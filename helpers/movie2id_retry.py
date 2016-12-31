import pickle
from imdb import IMDb
from imdb import IMDbError

ia = IMDb()

start = 3000

pkl_file = open('data/fails' + str(start) + '.pkl')

failler = pickle.load(pkl_file)

fails = []

id_dict = {}

i = 0
end = len(failler)

for fail in failler:
    name = fail.split('(')[0]
    print(name)
    try:
        results = ia.search_movie(name)
        print(results)
        if (len(results) > 0):
            result = results[0]
            id_dict[fail] = result.getID()
            print("Fetched : " + str(result))
        else:
            print('Filed to fetch ' + name)
            fails.append(fail)
        print(str(end - i) + " Did it.")
    except(IMDbError):
        print('Filed to fetch ' + name)
        fails.append(fail)
    i = i+1


output = open('data_f'+str(start)+'.pkl', 'wb')

pickle.dump(id_dict, output)

output2 = open('fails_f'+str(start)+'.pkl', 'wb')

pickle.dump(fails, output2)

print(fails)

output.close()

output2.close()
