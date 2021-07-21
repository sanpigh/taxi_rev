from taxi_rev.encoders import compute_rmse
from taxi_rev.data import clean_data, get_data
from taxi_rev.encoders import set_pipeline, set_preproc_pipe
from sklearn.model_selection import train_test_split

from memoized_property import memoized_property

import mlflow
from mlflow.tracking import MlflowClient


# self.experiment_name: name of the experiment
# self.mlflow_run: RUN object (class mlflow.entities.Run).
#   mlflow.entities.Run: contains the attribute .Data that contains
#   the run data, including metrics, parameters, and tags.


class Trainer:
    MLFLOW_URI = "https://mlflow.lewagon.co/"
    models = [
        "linear_regression",
        "decision_tree_regression",
        "random_forest_regressor",
    ]

    def __init__(self, experiment_name):
        self.experiment_name = experiment_name

    @memoized_property
    def mlflow_client(self):
        # Generates the MlflowClient object that will be used below, memoized allows to
        # create only one object in despite of the multiple calls
        mlflow.set_tracking_uri(self.MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id

    def mlflow_create_run(self):
        # creates the attribute mlflow_run
        self.mlflow_run = self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    # implement evaluate() function
    def evaluate(self, X_test, y_test, pipeline):
        """returns the value of the RMSE"""
        y_pred = pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def train(self):
        # store the data in a DataFrame
        df = clean_data(get_data())

        # set X and y
        y = df.pop("fare_amount")

        # hold out
        X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.3)

        for model in models:

            # build pipeline
            pipeline = set_pipeline("linear_regression")

            # train the pipeline
            pipeline.fit(X_train, y_train)

            # evaluate the pipeline
            rmse_train = self.evaluate(X_train, y_train, pipeline)
            rmse_val = self.evaluate(X_val, y_val, pipeline)

            self.mlflow_create_run()
            self.mlflow_log_metric("rmse_train", rmse_train)
            self.mlflow_log_metric("rmse_val", rmse_val)
            self.mlflow_log_param("model", model)


trainer = Trainer("[FR] [Bordeaux] [sanpigh] test_many_models v0")
trainer.train()


# implement train() function
# def train(X_train, y_train, pipeline):
#     '''returns a trained pipelined model'''
#     pipeline.fit(X_train, y_train)
#     return pipeline


# if __name__ == '__main__':
#     # df = clean_data(get_data(100))
#     # y = df.pop('fare_amount')
#     # pipe = set_pipeline()
#     # pipe.fit(df, y)
#     # print(pipe.score(df, y))
#     # print(evaluate(df, y, pipe))

#     # store the data in a DataFrame
#     df = clean_data(get_data())

#     # set X and y
#     y = df.pop('fare_amount')

#     # hold out
#     X_train, X_val, y_train, y_val = train_test_split(df, y, test_size=0.3)

#     # build pipeline
#     pipeline = set_pipeline()

#     # train the pipeline
#     train(X_train, y_train, pipeline)

#     # evaluate the pipeline
#     rmse_train = evaluate(X_train, y_train, pipeline)
#     rmse_val   = evaluate(X_val, y_val, pipeline)

#     print(rmse_train, rmse_val)
