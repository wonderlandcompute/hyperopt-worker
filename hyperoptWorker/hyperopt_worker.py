from hyperoptWorker import wonderland_pb2
from hyperoptWorker import Job, ListJobsRequest, new_client
import base64
import os
import json
import shutil
import traceback
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
import modelgym.metrics
from modelgym.utils import XYCDataset
from hyperopt import STATUS_OK, STATUS_FAIL
from modelgym.utils.evaluation import crossval_fit_eval
from azure.storage.file import FileService
import time
import logging

# TODO implement non-modegym learning.
# TODO combine MODEL_CLASSES and MODEL_PATH into one obj with the following struct model={'CtBClassifier':{model: 'CtBClassifier', path: 'path_to_exported_model.pickle'}}

MODEL_CLASSES = {
    'CtBClassifier': modelgym.models.CtBClassifier,
    'CtBRegressor': modelgym.models.CtBRegressor,
    'XGBClassifier': modelgym.models.XGBClassifier,
    'XGBRegressor': modelgym.models.XGBRegressor,
    'LGBMClassifier': modelgym.models.LGBMClassifier,
    'LGBMRegressor': modelgym.models.LGBMRegressor,
    'RFClassifier': modelgym.models.RFClassifier,
}

# TODO add non-modelgym metrics(need to rework modegym's function).
PREDEFINED_METRICS = {
    # classification metrics
    "accuracy": modelgym.metrics.Accuracy,
    "f1": modelgym.metrics.F1,
    "recall": modelgym.metrics.Recall,
    "log_loss": modelgym.metrics.Logloss,
    "precision": modelgym.metrics.Precision,
    "roc_auc": modelgym.metrics.RocAuc,

    # regression metrics
    # "mae": mean_absolute_error,
    "mse": modelgym.metrics.Mse,
    # "r2": r2_score,
}

MODEL_FILE = "model.pickle"
stub = new_client()
file_service = FileService(account_name='mylake',
                           account_key='nTYA+KhHEIuy2DVyG8uGuNev3qKGJ8Qm975hCkMgm+hGc7AW17RhnygFTKSNho5Iu8s3zwYcqxgrmte0tROBog==')
os.environ["AFSSHARE"] = "myshare"
os.environ["REPO_STORAGE"] = "~/repo-storage-worker"
repo_storage = Path(os.getenv('REPO_STORAGE')).expanduser()


def get_metric_function(metric):
    """Returns a predefined metric function object if metric corresponds to one,
    otherwise returns deserialized function object."""
    metric_function = PREDEFINED_METRICS.get(metric)

    if not metric_function:
        metric_function = pickle.loads(base64.b64decode(metric))

    return metric_function


def assess_model(metrics, model, train_objects, train_target):
    """Evaluates provided metrics for a trained model."""
    model_results = []
    prediction = model.predict(XYCDataset(train_objects))

    for metric in metrics:
        metric_function = get_metric_function(metric)
        score = metric_function(train_target, prediction)
        model_results.append(np.asscalar(score))

    return model_results


def cleanup_catboost():
    """Removes Catboost-generated files from current directory."""

    print("Removing Catboost-generated test/ and train/ folder !")
    shutil.rmtree('test', ignore_errors=True)
    shutil.rmtree('train', ignore_errors=True)

    os.remove('learn_error.tsv')
    os.remove('meta.tsv')
    os.remove('time_left.tsv')


def load_data(csv_path):
    """Loads train data from CSV file specified by its path. The target column must be 'y' """
    dataframe = pd.read_csv(csv_path)
    train_target = dataframe['y'].values
    train_objects = dataframe.drop(['y'], axis=1).values
    return train_target, train_objects


def process_job(job):
    input = json.loads(job.input)
    out_folder = Path(input["model_path"]).parent
    data_path = afs_download(input["data_path"])
    model_path = afs_download(input["model_path"])
    try:
        train_target, train_objects = load_data(data_path)
        dataset = XYCDataset(train_objects, train_target)
        with open(model_path, "rb") as f:
            models = json.load(f)
        cv = dataset.cv_split(models['cv'])

        for model in models['models']:
            if MODEL_CLASSES.get(model['type']):
                metrics = [PREDEFINED_METRICS.get(metric)() for metric in models['metrics']]
                result = crossval_fit_eval(model_type=MODEL_CLASSES[model['type']],
                                           params=model.get('params'),
                                           cv=cv,
                                           metrics=[PREDEFINED_METRICS.get(metric)() for metric in models['metrics']],
                                           verbose=False)

                res_model = MODEL_CLASSES[model['type']](params=model.get('params')).fit(dataset)
            else:
                raise NotImplementedError('Classifier %s is not supported')

        result["status"] = STATUS_OK
        losses = [cv_result[metrics[-1].name]
                  for cv_result in result["metric_cv_results"]]
        result["loss_variance"] = np.std(losses)

        if model['type'] in (MODEL_CLASSES['CtBClassifier'], MODEL_CLASSES['CtBRegressor']):
            cleanup_catboost()

        afs_upload(pickle.dumps(res_model), out_folder / 'model.pickle')
        afs_upload(json.dumps(result).encode(), out_folder / 'output.json')

        job.output = json.dumps({'output': str(out_folder / 'output.json'),
                                 'result_model_path': str(out_folder / 'model.pickle')})

        job.status = Job.COMPLETED
        stub.ModifyJob(job)
    except Exception as exc:
        logging.exception(str(exc))
        traceback.print_exc()
        log = "Error:\n" + str(exc) + "\n\n\nTrace:\n" + traceback.format_exc()
        afs_upload(str(log).encode(), out_folder / 'error.log')
        job.output = json.dumps({'error': str(out_folder / 'error.log')})
        job.status = wonderland_pb2.Job.FAILED
        stub.ModifyJob(job)
        return

    return


def afs_download(afs_path):
    """Downloads from AFS

    :param afs_path: relative path for data in the AFS share
    :return:
    """
    logging.info("Downloading file from the AFS")
    afs_path = Path(afs_path)
    local_path = repo_storage / afs_path
    if local_path.exists():
        logging.info("Wow, the data is already here. Getting the data from the pantry.")
        return local_path
    else:
        path_to_folder = local_path.parent
        path_to_folder.mkdir(parents=True, exist_ok=True)
    file_service.get_file_to_path(share_name=os.getenv("AFSSHARE"),
                                  directory_name=afs_path.parent,
                                  file_name=afs_path.name,
                                  file_path=local_path)
    logging.info("Downloading was finished")
    return local_path


def afs_upload(bytes, afs_path):
    """Uploads to the AFS

    :param <bute> bytes: data
    :param <string> afs_path: relative path for data in the AFS share
    :return:
    """
    file_service.create_file_from_bytes(share_name=os.getenv("AFSSHARE"),
                                        directory_name=afs_path.parent,
                                        file_name=afs_path.name,
                                        file=bytes)


def sleep_at_work(last_work):
    if time.time() - last_work < 10:
        time.sleep(1)
        return
    if time.time() - last_work < 100:
        time.sleep(5)
        return
    time.sleep(5)


def main():
    last_work = time.time()
    logging.basicConfig(level=logging.INFO)
    while True:
        sleep_at_work(last_work)
        try:
            logging.info("Knock, knock, wonderland")
            pulled_jobs = stub.PullPendingJobs(ListJobsRequest(how_many=1, kind='hyperopt'))
            for job in pulled_jobs.jobs:
                last_work = time.time()
                logging.info("Gotcha!Learning...JOB_ID={}\n".format(job.id))
                process_job(job)
                logging.info("Processed:\n{}".format(job))
        except Exception as exc:
            logging.exception(exc)


if __name__ == '__main__':
    main()
