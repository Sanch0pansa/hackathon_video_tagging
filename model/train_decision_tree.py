import lightning as pl
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from classificators.decision_tree_classifier import DataModule, Classificator, RandForest
from classificators.decision_tree_classifier.DataModule import VideoDataModule
from sklearn.multioutput import MultiOutputClassifier

import xgboost as xgb

import numpy as np

def iou_loss(pred, target, eps=1e-6):
    """
    IoU Loss Function.
    
    pred: предсказанная маска (содержит вероятности классов).
    target: истинная маска (значения 0 или 1 для каждого пикселя).
    eps: малое значение для избежания деления на ноль.
    """
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return 1 - (intersection / (union + eps))

def calculate_iou_per_class(pred_mask, true_mask, num_classes):
    """
    Вычисляет IoU для каждого класса.
    pred_mask: предсказанная маска (2D массив с классами).
    true_mask: истинная маска (2D массив с классами).
    num_classes: количество классов.
    
    Возвращает массив IoU для каждого класса.
    """
    iou_list = np.zeros(num_classes)
    
    for cls in range(num_classes):
        # Пиксели, которые являются 1 для текущего класса
        pred_cls = (pred_mask == cls).astype(np.uint8)
        true_cls = (true_mask == cls).astype(np.uint8)
        
        # Область пересечения
        intersection = np.sum(pred_cls * true_cls)
        
        # Область объединения
        union = np.sum(pred_cls) + np.sum(true_cls) - intersection
        
        # IoU
        iou = intersection / union if union != 0 else 0
        iou_list[cls] = iou
        
    return iou_list

def calculate_mean_iou(pred_mask, true_mask, num_classes):
    """
    Вычисляет среднее значение IoU для всех классов.
    """
    iou_per_class = calculate_iou_per_class(pred_mask, true_mask, num_classes)
    mean_iou = np.mean(iou_per_class)
    return mean_iou, iou_per_class


def train_model(video_meta_file, categories_file, tensor_dir, max_depth = 10):
    # Инициализация логгера Wandb

    # data = pd.read_csv(video_meta_file)
    # categoric_data = pd.read_csv(categories_file)
    # embeddings = torch.load(tensor_dir + '/' + data['video_id'][2] + ".pt")
    # print(embeddings.shape)
    # ides = data['video_id']
    # print(ides)
    
    data_module = VideoDataModule(
        video_meta_file=video_meta_file,
        categories_file=categories_file,
        tensor_dir=tensor_dir
    )
    
    data_module.setup()

    data = data_module.dataset.video_meta_df.iloc[:, [0]]
    # print(data)
    y = []
    
    data['emb'] = 0
    embeddings = []

    for i in range(data.shape[0]):
        embedding = torch.load(tensor_dir + '/' + data['video_id'][i] + ".pt")
        label = data_module.dataset[i][1]
        y.append(label)
        embeddings += embedding
    
    data['emb'] = embeddings
    X = np.array(embeddings)
    y = np.array(y)
    print(y[3])

    # max_len = max(len(seq) for seq in X)
    # X_padded = np.array([np.pad(seq, (0, max_len - len(seq))) for seq in X])
    # print(X['emb'].shape)
    # print(np.array(y))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # model = xgb.XGBClassifier(n_estimators=5)
    # # Обучение
    # model.fit(X_train, y_train)
    # # Предсказания
    # y_pred = model.predict(X_test)

    # Создаем модель DecisionTree
    tree = RandForest.DecisionTree(max_depth=max_depth)
    # # tree.find_best_depth(X_train, y_train, len(y_train))
    # tree.model.fit(X_train, y_train, tree.max_depth)
    # y_pred = tree.model.predict(X_test)
    

    # MultiOutputClassifier
    # multi_target_clf = MultiOutputClassifier(tree.model, n_jobs=-1)
    # multi_target_clf.fit(X_train, y_train)
    # y_pred = multi_target_clf.predict(X_test)

    # bagging_model = BaggingClassifier(tree.model, n_estimators=10)

    # multi_target_clf = MultiOutputClassifier(bagging_model, n_jobs=-1)
    # multi_target_clf.fit(X_train, y_train)
    # y_pred = multi_target_clf.predict(X_test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # Задание параметров
    params = {
    'objective': 'multi:softmax',  # Используйте 'multi:softmax' для предсказания классов
    'num_class': len(data_module.dataset.category_to_idx2),                # Количество классов
    'eta': 0.1,                    # Скорость обучения
    'max_depth': 10,                # Максимальная глубина дерева
    'eval_metric': 'mlogloss',     # Метрика для оценки
    }

    num_rounds = 100
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class = 2, eta = 0.1, max_depth=5, eval_metric='mlogloss')
    multi_target_model = MultiOutputClassifier(xgb_model, n_jobs=-1)
    multi_target_model.fit(X_train, y_train)
    y_pred = multi_target_model.predict(X_test)

    # model = xgb.train(params, dtrain, num_rounds)

    # Предсказание классов
    # y_pred = model.predict(dtest)
    # #Метрика iou
    iou = calculate_mean_iou(y_test, y_pred, len(y_test))
    print("IOU")
    print(iou)
    # # Выводим предсказания
    print(y_pred[5])
    # print(y_pred[0])
    # print(y_pred.shape)

    # tree.find_best_depth(max_depth_range=range(4, max_depth))

    
    # Определяем Trainer
    # trainer = pl.Trainer(
    #     accelerator="auto",
    #     devices=1
    # )
    
    # Запуск обучения
    # trainer.fit(tree.model, data_module)

if __name__ == "__main__":
    train_model(
        video_meta_file=r".\model\train_dataset_tag_video\baseline\train_data_categories.csv",
        categories_file=r".\model\train_dataset_tag_video\baseline\IAB_tags.csv",
        tensor_dir=r".\model\embeddings_1536",
        max_depth = 5
    )


