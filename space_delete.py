import os

liste = os.listdir('/media/tugrulz/Yeni Birim/mldata/subtitles/')

path = '/media/tugrulz/Yeni Birim/mldata/subtitles/'

for p in liste:
    result = ""
    f = open(path+p, 'rb')
    for line in f.readlines():
        if line != "\n":
            result += line
    g = open('/media/tugrulz/Yeni Birim/mldata/subtitles2/'+p, 'wb')
    g.write(result)
    g.close()
    f.close()