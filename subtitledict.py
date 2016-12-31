import pickle

codes = ['0', '3000', '4500', '6000', '_f0', '_f3000', '_f4500', '_f6000']

masterdict = {}

for code in codes:
    pkl_file = open('data/data' + str(code) + '.pkl')
    dict = pickle.load(pkl_file)
    for key in dict:
        masterdict[key] = dict[key]
    pkl_file.close()

# manually add some of the failed keys
masterdict['Sex: The Annabel Chong Story'] = '0181810'
masterdict['Hellsing Ultimate OVA Series (2006)'] = '0495212'
masterdict["Family Guy Presents: It's a Trap (2010)"] = '1701991'
masterdict["Family Guy Presents: It's a Trap (2010)"] = '1701991'
masterdict["Bad Timing: A Sensual Obsession (1980)"] = '0080408'

subf = open('subdict.pkl', 'wb')

pickle.dump(masterdict, subf)
subf.close()