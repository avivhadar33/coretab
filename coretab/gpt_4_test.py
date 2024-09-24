from openai import OpenAI
api_key = 'put your key here'
client = OpenAI(api_key=api_key)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, roc_auc_score, f1_score
import time
from dataclasses import dataclass
import json

from coretab.coreset_algorithms import CoreTabDT, CoreTabXGB
from dataset_utils import null_columns, encode_categorical_features, Dataset


@dataclass
class GptResults:
    train_sets: dict
    test_set: tuple
    ft_objects: dict
    coretab_objs: dict
    predictions: dict
    scores: dict


def create_train_set(X_train, y_train, train_sample_per_class):
    X_train_s1 = X_train.sample(frac=1)[y_train == 1].sample(train_sample_per_class)
    X_train_s0 = X_train.sample(frac=1)[y_train == 0].sample(train_sample_per_class)
    X_train_s = pd.concat([X_train_s0, X_train_s1]).sample(frac=1)
    y_train_s = y_train[X_train_s.index]
    return X_train_s, y_train_s


class GptClassifier():

    def __init__(self, features_format, model=None, experiment_name=None, open_ai_file_id=None):
        self.X_train = None
        self.y_train = None
        self.features_format = features_format
        self.model = model
        self.experiment_name = experiment_name
        self.responses = []

        self.file_path = None if open_ai_file_id is not None else 'datasets/gpt_format/' + experiment_name + '.jsonl'
        self.open_ai_file_id = open_ai_file_id

    def row_to_text(self, row: pd.Series):
        features_encoded = ' | '.join([f'{c}:{row[c]}' for c in X_train.columns])
        full_text = self.features_format.format(features_encoded=features_encoded)
        return full_text

    @staticmethod
    def label_to_answer(label: int):
        text_label = "'Yes'" if label else "'No'"
        return f"<answer>{text_label}</answer>"

    def system_message(self):
        return {"role": "system", "content": self.system_content}

    def set_model(self, model: str):
        self.model = model

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        messages = self.parse_dataset(X, y)
        if not self.open_ai_file_id:
            with open(self.file_path, 'w') as f:
                for example in messages:
                    json.dump(example, f)
                    f.write('\n')

            open_ai_file = client.files.create(
                file=open(self.file_path, "rb"),
                purpose="fine-tune"
            )
            self.open_ai_file_id = open_ai_file.id

        if self.experiment_name is None or self.model is None:
            raise Exception('define experiment_name or model')

        finetune_object = client.fine_tuning.jobs.create(
            training_file=self.open_ai_file_id,
            model=self.model,
            suffix=self.experiment_name,
            hyperparameters={
                "batch_size": 30,
                "n_epochs": 1
            }
        )
        return finetune_object

    def parse_dataset(self, X, y):
        gpt_queries = []
        if len(X) > 5000:
            raise Exception(f'Too many: {len(X)}')

        for index, row in X.iterrows():
            if y is not None:
                messages = {'messages': [
                    {"role": "user", "content": self.row_to_text(row)},
                    {"role": "assistant", "content": self.label_to_answer(int(y[index]))}
                ]}
            else:
                messages = [{"role": "user", "content": self.row_to_text(row)}]

            gpt_queries.append(messages)

        return gpt_queries

    def predict(self, X_test, model=None):
        if len(X_test) > 3000:
            raise Exception(f'Too many: {len(X_test)}')

        cur_model = model if model else self.model
        if not cur_model:
            raise Exception(f'Not given a model')
        print(f'model used {cur_model}')

        y_pred = []
        for index, row in X_test.iterrows():
            response = client.chat.completions.create(
                model=cur_model,
                messages=[{"role": "user", "content": self.row_to_text(row)}],
                temperature=0,
                stop='</answer>',
                max_tokens=2048,
            )
            self.responses.append(response)
            answer = response.choices[0].message.content.split("<answer>")[-1].strip().replace("'", '')
            prediction = 1 if answer == 'Yes' else 0
            y_pred.append(prediction)
        return y_pred


def wait_to_models_fintune(ft_ids_list):
    print(f'waiting for those ft jobs: {ft_ids_list}')

    inital_length = len(ft_ids_list)
    finished = 0
    while finished < inital_length:
        finished_obj = None
        for ft_id in ft_ids_list:
            ft_obj = client.fine_tuning.jobs.retrieve(ft_id)
            if ft_obj.status == 'succeeded':
                finished += 1
                finished_obj = ft_obj
                break
            elif ft_obj.status == "failed":
                raise Exception(f'Finetune failed {str(ft_obj)}')
        if finished_obj:
            ft_ids_list.remove(ft_obj.id)
        time.sleep(30)


def test_dataset_gpt(dataset: Dataset, base_experiment_name, dataset_content, train_sample_per_class=1_500,
                     test_amount=2_000, encode_categorical=False):
    X_test_s = dataset.X_test.sample(frac=1).sample(test_amount)
    y_test_s = dataset.y_test[X_test_s.index]
    print(y_test_s.value_counts())

    X_train_s, y_train_s = create_train_set(dataset.X_train, dataset.y_train, train_sample_per_class)
    print(y_train_s.value_counts())

    if encode_categorical:
        df_coreset = encode_categorical_features(dataset.df)
        df_coreset = null_columns(df_coreset)
    else:
        df_coreset = dataset.df

    coretab_dt = CoreTabDT(examples_to_keep=15_000)
    X_dt, y_dt = coretab_dt.create_coreset(df_coreset.loc[dataset.X_train.index, :].drop(target_col, axis=1),
                                           dataset.y_train)
    X_train_s_dt, y_train_s_dt = create_train_set(dataset.X_train.loc[X_dt.index, :], y_dt, train_sample_per_class)

    coretab_xgb = CoreTabXGB(examples_to_keep=15_000)
    X_xgb, y_xgb = coretab_xgb.create_coreset(df_coreset.loc[dataset.X_train.index, :].drop(target_col, axis=1),
                                              dataset.y_train)
    X_train_s_xgb, y_train_s_xgb = create_train_set(dataset.X_train.loc[X_xgb.index, :], y_xgb, train_sample_per_class)

    clf_default = GptClassifier(dataset_content, experiment_name=base_experiment_name + '_default',
                                model='gpt-4o-mini-2024-07-18')
    ft_defualt_obj = clf_default.fit(X_train_s, y_train_s)

    clf_dt = GptClassifier(dataset_content, experiment_name=base_experiment_name + '_dt',
                           model='gpt-4o-mini-2024-07-18')
    ft_dt_obj = clf_dt.fit(X_train_s_dt, y_train_s_dt)

    clf_xgb = GptClassifier(dataset_content, experiment_name=base_experiment_name + '_xgb',
                            model='gpt-4o-mini-2024-07-18')
    ft_xgb_obj = clf_xgb.fit(X_train_s_xgb, y_train_s_xgb)

    wait_to_models_fintune([ft_defualt_obj.id, ft_dt_obj.id, ft_xgb_obj.id])

    ft_defualt_obj, ft_dt_obj, ft_xgb_obj = (client.fine_tuning.jobs.retrieve(ft_defualt_obj.id),
                                             client.fine_tuning.jobs.retrieve(ft_dt_obj.id),
                                             client.fine_tuning.jobs.retrieve(ft_xgb_obj.id))
    print('Training finished!')
    y_pred = clf_default.predict(X_test_s, model=ft_defualt_obj.fine_tuned_model)
    y_pred_dt = clf_dt.predict(X_test_s, model=ft_dt_obj.fine_tuned_model)
    y_pred_xgb = clf_xgb.predict(X_test_s, model=ft_xgb_obj.fine_tuned_model)

    model_dt = TrivialModel(pd.Series(data=y_pred_dt, index=X_test_s.index))
    y_pred_te_dt = coretab_dt.te_predict(df_coreset.loc[X_test_s.index, :].drop(target_col, axis=1), model_dt)

    model_xgb = TrivialModel(pd.Series(data=y_pred_xgb, index=X_test_s.index))
    y_pred_te_xgb = coretab_xgb.te_predict(df_coreset.loc[X_test_s.index, :].drop(target_col, axis=1), model_xgb)

    results = GptResults(
        train_sets={'default': (X_train_s, y_train_s), 'dt': (X_train_s_dt, y_train_s_dt),
                    'xgb': (X_train_s_xgb, y_train_s_xgb)},
        test_set=(X_test_s, y_test_s),
        ft_objects={'default': ft_defualt_obj, 'dt': ft_dt_obj, 'xgb': ft_xgb_obj},
        coretab_objs={'dt': coretab_dt, 'xgb': coretab_xgb},
        predictions={'default': y_pred, 'dt': y_pred_dt, 'xgb': y_pred_xgb},
        scores={
            'default': f1_score(y_test_s, y_pred), 'dt': f1_score(y_test_s, y_pred_dt),
            'xgb': f1_score(y_test_s, y_pred_xgb),
            'te-dt': f1_score(y_test_s, y_pred_te_dt), 'te-xgb': f1_score(y_test_s, y_pred_te_xgb)
        }
    )
    print(results.scores)
    return results


class TrivialModel:

    def __init__(self, y_pred):
        self.y_pred = y_pred

    def predict(self, X_test):
        return self.y_pred[X_test.index]


def print_results(results):
    mean_def, mean_dt, mean_xgb = (round(np.mean([r.scores['default'] for r in results]), 5),
                                   round(np.mean([r.scores['dt'] for r in results]), 5),
                                   round(np.mean([r.scores['xgb'] for r in results]), 5))
    std_def, std_dt, std_xgb = (round(np.std([r.scores['default'] for r in results]), 5),
                                round(np.std([r.scores['dt'] for r in results]), 5),
                                round(np.std([r.scores['xgb'] for r in results]), 5))

    mean_te_dt, mean_te_xgb = (round(np.mean([r.scores['te-dt'] for r in results]), 5),
                               round(np.mean([r.scores['te_xgb'] for r in results]), 5))
    std_te_dt, std_te_xgb = (round(np.std([r.scores['te-dt'] for r in results]), 5),
                             round(np.std([r.scores['te_xgb'] for r in results]), 5))

    print(f'result default: {mean_def} +- {std_def}')
    print(f'result dt: {mean_dt} +- {std_dt}')
    print(f'result xgb: {mean_xgb} +- {std_xgb}')
    print(f'result te-dt: {mean_te_dt} +- {std_te_dt}')
    print(f'result te-xgb: {mean_te_xgb} +- {std_te_xgb}')


gpt_promts = {
    'di': """Based on the following health and lifestyle information (features), determine if this person is likely to have diabetes. Please respond only with 'Yes' or 'No'
        
Here are the health and lifestyle information of the person (seperated by |):    
<features>{features_encoded}</feature>

Respond using this format:
<answer>'Yes' or 'No'</answer>""",
    'bf': """Based on the information (features), determine if this bank account is a fraud. Please respond only with 'Yes' or 'No' ('Yes' if it is fraud)
        
Here are the information of account (seperated by |):    
<features>{features_encoded}</feature>

Respond using this format:
<answer>'Yes' or 'No'</answer>"""
}


def main():
    df = pd.read_csv('Diabetes dataset path')
    target_col = 'Diabetes_binary'
    promt = gpt_promts['di']
    gpt_results = []

    for i in range(4):
        X_train, X_test, y_train, y_test = train_test_split(df.drop(target_col, axis=1), df[target_col],
                                                            test_size=0.2, shuffle=True)
        dataset_di = Dataset(df, X_train, y_train, X_test, y_test, target_col, f1_score)
        returned_obj = test_dataset_gpt(dataset_di, 'experiment_name' + '_di', promt)
        gpt_results.append(returned_obj)

    print_results(gpt_results)
    return gpt_results



