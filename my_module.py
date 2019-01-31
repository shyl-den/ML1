import operator
from collections import Counter

def euclidDist(a,b):
    return np.sqrt(np.sum((a-b)**2))

def my_knn(trainSet, xTest, K):
    predictions=[]
    
    for x in range(len(xTest)):
        
        allDistancesToTrainSamples = []
        numberOfTrainSamples = len(trainSet)
        for i in range(numberOfTrainSamples):
            d = euclidDist(trainSet[i,0:2], xTest[x,0:2])
            allDistancesToTrainSamples.append((trainSet[i,3], d))

        allDistancesToTrainSamples.sort(key = operator.itemgetter(1))
        kNearestNeighbours = allDistancesToTrainSamples[:K]

        kNearestNeighboursClasseLabels = [neighbour[0] for neighbour in kNearestNeighbours]
        count = Counter(kNearestNeighboursClasseLabels)
       
        predictedClassLevel = count.most_common()[0][0];
        predictions.append(predictedClassLevel)

    return  predictions

def lin_reg_na(df, col):
    df1 = df.dropna()
    df2 = df1.drop([col],axis = 1)
    
    non_null = df[df.col.notnull()]
    is_null = df[df.col.isnull()]
    X_col = [c for c in df.columns if c != col]
    X_train = df2[:]
    y_train = df1[col]
    linear_regr = LinearRegression()
    linear_regr.fit(X_train, y_train)
    X_pred = is_null[X_col]
    predict = linear_regr.predict(X_pred)
    is_null[col] = predict
    df1 = pd.concat([is_null, non_null], axis=0)
    return df1


def my_dropna_column(*args, df):
    for i in range(len(args)):
        df.drop([args[i]], axis=1, inplace=True)
    return df


def my_dropna_row(*args, df):
    for i in range(len(args)):
        df.drop([args[i]], inplace=True)
    return df


def my_replacena_mean(*args, df):
    for i in range(len(args)):
        df[args[i]].fillna(df.loc[:, args[i]].mean(), inplace=True)
    return df


def my_replacena_mode(*args, df):
    for i in range(len(args)):
        df[args[i]].fillna(df.loc[:, args[i]].mode()[0], inplace=True)
    return df


def my_replacena_median(*args, df):
    for i in range(len(args)):
        df[args[i]].fillna(df.loc[:, args[i]].median(), inplace=True)
    return df


def my_standartize(column, df):
    for i in range(len(df[column])):
        df[column].replace(df[column][i], abs((df[column][i] - df[column].mean()) / df[column].std()), inplace=True)


def my_normalize(column, df):
    min = df[column].min()
    max = df[column].max()
    for i in range(len(df[column])):
        df[column].replace(df[column][i], (df[column][i] - min) / (max - min), inplace=True)
