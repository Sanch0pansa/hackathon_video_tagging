import lightning as pl
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import WandbLogger
from classificators.decision_tree_classifier import DataModule, Classificator
from classificators.decision_tree_classifier.DataModule import VideoDataModule


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

    X = data_module.dataset.video_meta_df.iloc[:, [0]]
    print(X)
    y = []
    
    X['emb'] = 0
    embeddings = []

    for i in range(X.shape[0]):
        embedding = torch.load(tensor_dir + '/' + X['video_id'][i] + ".pt")
        label = data_module.dataset[i][1]
        y.append(label)
        embeddings += embedding
    
    X['emb'] = embeddings
    y = np.array(y)
    # print(X['emb'].shape)
    # print(np.array(y))
        
        

    # Создаем модель
    tree = Classificator.DecisionTree(max_depth=max_depth)

    tree.model.fit(np.array(embeddings), y, tree.max_depth)
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
        max_depth = 11
    )


