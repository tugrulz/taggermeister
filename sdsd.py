print(train)

result_train = []
for t in train:
    result_train += t

result_test = []
for t in train:
    result_test += t

train = np.array(result_train, dtype="int")
test = np.array(result_test, dtype="int")