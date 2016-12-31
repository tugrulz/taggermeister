#!/usr/bin/env python

import fileinput
from os import listdir
import sys
import glob
import os.path

# years = range(0, 1985)
years = range(1984, 1985)

# can't even deal right now
# 1951 deleted
# 1953 deleted

import os
years = os.listdir('/media/tugrulz/Yeni Birim/mldata/en/OpenSubtitles2016/xml/en/')

import pickle
inp = open('subdict.pkl', 'rb')
iddict = pickle.load(inp)
dict = {v: k for k, v in iddict.iteritems()}

for xyear in years:
    print(xyear)
    year = '/media/tugrulz/Yeni Birim/mldata/en/OpenSubtitles2016/xml/en/' + xyear

    for movieDir in glob.glob(year + '/*' * 1):
        movieFiles = listdir(movieDir)
        try:
            script = filter(lambda x: x.endswith('.xml'), movieFiles)[0]
        except (IndexError):
            print(movieFiles)
            continue

        id = movieDir.split('/')[-1]
        zeros = 7 - len(id)
        for i in range(0,zeros):
            id = '0' + id

        if (dict.has_key(id) == False): #| (id == '0056801') | (id == '0058604')):
            continue
        else:
            name = dict[id]
            name = name.replace('/', "slash")
            if (id == '0056801'):
                name = '8 1slash2 (8slash) (1963)'
            if(os.path.isfile('/media/tugrulz/Yeni Birim/mldata/subtitles/' + name + '.txt')):
                continue
            text = ""
            f=open(movieDir+'/'+script)
            for line in f.readlines():
                text += line

            print movieDir+'/'+script

            from lxml import etree
            root = etree.fromstring(text)
            result = ""

            already = True
            for sentence in root:
                for w in sentence:
                    if(w.text is None):
                        if (already == False):
                            result += "\n"
                            already = True
                    else:
                        if(w.text[0] >= 'A' and w.text[0] <= 'z') or (w.text[0] >= '0' and w.text[0] <= '9'):
                            result += " "
                            already = False
                        result += w.text
                if(already == False):
                    result += "\n"
                    already = True

            # tmp = []
            # for x in root.xpath('//document/s/w'):
            #     tmp.append(x.text)




            # for i in range(len(tmp)-1):
            #     result += tmp[i]
            #     if tmp[i+1] == None:
            #         continue
            #     char = tmp[i+1][0]
            #     if (char >= 'A' and char <='z') or (char >= '0' and char <='9'):
            #         result += ' '

            #print result
            print('Sub id:' + id + " Name: " + name + ".txt")
            with open('/media/tugrulz/Yeni Birim/mldata/subtitles/'+ name + '.txt', 'w') as g:
                g.write(result.encode('utf-8', 'ignore'))


