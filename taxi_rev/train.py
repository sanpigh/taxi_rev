
from taxi_rev.encoders import compute_rmse
from taxi_rev.data import clean_data, get_data
from taxi_rev.encoders import set_pipeline
from sklearn.model_selection import train_test_split


# implement train() function
def train(X_train, y_train, pipeline):
    '''returns a trained pipelined model'''
    pipeline.fit(X_train, y_train)
    return pipeline



# implement evaluate() function
def evaluate(X_test, y_test, pipeline):
    '''returns the value of the RMSE'''
    y_pred = pipeline.predict(X_test)
    rmse = compute_rmse(y_pred, y_test)
    return rmse


if __name__ == '__main__':
    # df = clean_data(get_data(100))
    # y = df.pop('fare_amount')
    # pipe = set_pipeline()
    # pipe.fit(df, y)
    # print(pipe.score(df, y))
    # print(evaluate(df, y, pipe))
    
    # store the data in a DataFrame
    df = clean_data(get_data())

    # set X and y
    y = df.pop('fare_amount')

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.3)

    # build pipeline
    pipeline = set_pipeline()

    # train the pipeline
    train(X_train, y_train, pipeline)

    # evaluate the pipeline
    rmse_train = evaluate(X_train, y_train, pipeline)     
    rmse_val   = evaluate(X_val, y_val, pipeline)    

    print(rmse_train, rmse_val)