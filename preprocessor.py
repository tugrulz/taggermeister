from bs4 import BeautifulSoup
import time

input = open("data/3775942.xml", "rb")

xl = input.read()

soup = BeautifulSoup(xl)

sentences = soup.find_all('s id')
print(sentences)

result = ""

start = time.time()

for s in sentences:
    #print(s)
    words_soup = BeautifulSoup(str(sentences))
    words = words_soup.find_all('w')
    for w in words:
        #print(w)
        #print(w.getText())
        result += str(w.getText()) + " "
    #print(result)

end = time.time()

print("Took " + str(end - start) + " to preprocess.")

output = open("data/3775942.txt", "wb")

#output.write(result)

end2 = time.time()

print("Took " + str(end2 - end) + " to write.")



