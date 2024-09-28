import torch
from classificators.mlp_classifier.Classificator import MultiTaskClassifier
from classificators.mlp_classifier.DataModule import VideoDataset
import pandas as pd
import os

def load_model(checkpoint_path, input_dim=1536, num_classes=None):
    # Если количество классов не передано, загружаем его из чекпоинта
    if num_classes is None:
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        num_classes = checkpoint['hyper_parameters']['num_classes']
    
    # Загружаем модель с соответствующим количеством классов
    model = MultiTaskClassifier(input_dim=input_dim, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # Переводим модель в режим инференса
    return model

def get_num_classes(categories_file):
    # Читаем файл категорий и получаем количество уникальных категорий
    categories_df = pd.read_csv(categories_file)
    categories_df['full_category'] = categories_df.apply(
        lambda row: ': '.join(filter(lambda x: str(x) != 'nan', 
                                     [row['Уровень 1 (iab)'], row['Уровень 2 (iab)'], row['Уровень 3 (iab)']])), 
        axis=1
    )
    
    return len(categories_df['full_category'].unique())

def predict_tags(model, tensor_path, categories_file, threshold=0.5):
    # Загружаем тензор для видео
    tensor = torch.load(tensor_path, weights_only=True)
    tensor = tensor.view(tensor.shape[1])  # Убедимся, что тензор имеет нужную форму (1D)
    
    # Прогоняем тензор через модель для получения предсказаний
    with torch.no_grad():
        preds = model(tensor.unsqueeze(0))  # Добавляем batch размерность

    # Преобразуем вероятности в бинарные метки (теги)
    predicted_labels = (preds > threshold).squeeze().cpu().numpy()
    

    # Загружаем список категорий
    categories_df = pd.read_csv(categories_file)
    categories_df['full_category'] = categories_df.apply(
        lambda row: ': '.join(filter(lambda x: str(x) != 'nan', 
                                     [row['Уровень 1 (iab)'], row['Уровень 2 (iab)'], row['Уровень 3 (iab)']])), 
        axis=1
    )
    category_names = categories_df['full_category'].tolist()

    # Преобразуем бинарные метки в текстовые категории
    predicted_tags = [category_names[i] for i, val in enumerate(predicted_labels) if val == 1]

    return predicted_tags

if __name__ == "__main__":
    # Задаем пути
    checkpoint_path = "./checkpoints/final_model.ckpt"  # Путь к чекпоинту модели
    tensor_path = "./embeddings_1536/0a7a288165c6051ebd74010be4dc9aa8.pt"  # Путь к тензору видео
    categories_file = "./train_dataset_tag_video/baseline/IAB_tags.csv"  # Путь к CSV с категориями
    
    # Получаем количество классов из файла категорий
    num_classes = get_num_classes(categories_file)
    
    # Загрузим обученную модель
    model = load_model(checkpoint_path, input_dim=1536, num_classes=num_classes)
    
    # Прогоняем инференс и получаем текстовые теги
    predicted_tags = predict_tags(model, tensor_path, categories_file)
    
    # Выводим результат
    print(f"Predicted tags: {predicted_tags}")
