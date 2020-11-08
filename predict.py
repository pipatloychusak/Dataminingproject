import pickle

pkl_filename = "model_adidas_SVM.pkl"
with open(pkl_filename, 'rb') as file:
    pickled_model = pickle.load(file)

x_new = [[1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0]]
print(pickled_model.predict(x_new))

# score = pickled_model.score(x, y)
# print("Test score: {0:.2f} %".format(100 * score))

