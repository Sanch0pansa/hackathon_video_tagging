import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

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
    
    def find_best_depth(self, X, y, max_depth_range):
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
            clf = DecisionTreeClassifier(max_depth=depth)
            
            # Кросс-валидация
            scores = cross_val_score(clf, X, y, cv=5)
            
            # Среднее значение точности
            mean_scores.append(np.mean(scores))
            print(f"Глубина {depth}: точность {np.mean(scores):.4f}")
    
        # Лучшая глубина
        best_depth = max_depth_range[np.argmax(mean_scores)]
        print(f"\nОптимальная глубина дерева: {best_depth} с точностью {max(mean_scores):.4f}")
        
        # Обновляем модель с лучшей глубиной
        self.max_depth = best_depth
        self.model = DecisionTreeClassifier(max_depth=self.max_depth)
        return best_depth