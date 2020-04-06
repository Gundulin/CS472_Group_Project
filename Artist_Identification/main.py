from tools.arff import Arff
from sklearn import tree

if __name__ == '__main__':
    mat = Arff("datasets/artist_identification.arff", label_count=1)  ## label_count = 0 because clustering is unsupervised.
    data = mat.data[:, 0:-1]
    labels = mat.data[:, -1].reshape(-1, 1)
    DT = tree.DecisionTreeClassifier()
    DT.fit(data, labels)
    score = DT.score(data, labels) # TODO: put in test data instead of just data (same for labels)

    print("score: {}".format(score))


    # print("DEBUG:\n\n**Lenses**\n")
    # mat = Arff("../data/lenses.arff", label_count=1)
    # counts = []
    # for i in range(mat.data.shape[1]):
    #     counts += [mat.unique_value_count(i)]
    # data = mat.data[:, 0:-1]
    # labels = mat.data[:, -1].reshape(-1, 1)
    # DT = Dtree.DTClassifier(counts)
    # DT.fit(data, labels)
    # test_mat = Arff("../data/all_lenses.arff", label_count=1)
    # test_data = test_mat.data[:, 0:-1]
    # test_labels = test_mat.data[:, -1].reshape(-1, 1)
    # prediction = DT.predict(test_data)
    # score = DT.score(test_data, test_labels)
    # np.savetxt("pred_lenses.csv", prediction, delimiter=",")
    # print("Accuracy =  [{:.2f}]".format(score))