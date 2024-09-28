import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

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
    iou_list = np.zeros(num_classes, dtype=object)
    
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



class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.model = DecisionTreeClassifier(max_depth=self.max_depth)

    def fit(self, X, y):
        """
        Обучение модели 
        на данных X и метках y.
    
        
        :param X: Признаки (матрица, где строки - образцы, колонки - признаки)
        :param y: Метки классов (вектор)
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Предсказание классов для новых данных X.
        
        :param X: Новые данные (матрица, где строки - образцы, колонки - признаки)
        :return: Вектор предсказанных классов
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Оценка точности модели на тестовых данных.
        
        :param X: Признаки тестовых данных
        :param y: Истинные метки тестовых данных
        :return: Точность модели
        """
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def find_best_depth(self, X, y, num_classes, max_depth_range=range(4, 7)):
        """
        Метод для подбора оптимальной глубины дерева с использованием кросс-валидации.
        
        :param X: Признаки
        :param y: Метки классов
        :param max_depth_range: Диапазон глубин для поиска
        :return: Оптимальная глубина дерева
        """
        mean_scores = []
        for depth in max_depth_range:
            # Создаем временную модель для каждой глубины
            clf = DecisionTreeClassifier(criterion=iou_loss, max_depth=depth)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
            clf.fit(X_train, y_train, depth)
            # Кросс-валидация
            y_pred = self.model.predict(X_test)
            scores = calculate_mean_iou(y_pred, y_test, num_classes)
        
            # Среднее значение точности
            # mean_scores.append(np.mean(scores))
            print(f"Глубина {depth}: точность {np.mean(scores):.4f}")
    
        # Лучшая глубина
        best_depth = max_depth_range[np.argmax(mean_scores)]
        print(f"\nОптимальная глубина дерева: {best_depth} с точностью {max(mean_scores):.4f}")
        
        # Обновляем модель с лучшей глубиной
        self.max_depth = best_depth
        self.model = DecisionTreeClassifier(max_depth=self.max_depth)
        return best_depth