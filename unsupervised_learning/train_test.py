import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Функция для нахождения контуров
from images_preparing import load_images, preprocess_image


def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Функция для выделения сетки и ячеек
def extract_grid_and_cells(image, contours):
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 25 and h > 25:  # Отфильтруем слишком маленькие контуры
            cell = image[y:y+h, x:x+w]
            cell = cv2.resize(cell, (50, 50))  # Изменяем размер ячейки для унификации
            cells.append((x, y, w, h, cell.flatten(), cv2.contourArea(contour), w / h))
    return cells

# Функция для нахождения линий на изображении
def find_lines(image):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=80, minLineLength=25, maxLineGap=10)
    return lines

# Функция для классификации ячеек
def classify_cells(image, cells):
    classifications = []
    for idx, (x, y, w, h, cell, area, aspect_ratio) in enumerate(cells):
        cell_img = image[y:y + h, x:x + w]
        lines = find_lines(cell_img)

        if lines is None:
            classifications.append('O')
        else:
            if len(lines) >= 2:
                classifications.append('X')
            else:
                classifications.append('O')

    largest_area = max(cells, key=lambda cell: cell[5])[5]
    for idx, cell in enumerate(cells):
        if cell[5] == largest_area:
            classifications[idx] = 'grid'
            break

    return classifications

# Загрузка изображений
image_folder = 'unsupervised_learning/board_images/'
images = load_images(image_folder)

# Извлечение признаков ячеек из всех изображений
all_cells = []
for img in images:
    preprocessed_img = preprocess_image(img)
    contours = find_contours(preprocessed_img)
    cells = extract_grid_and_cells(preprocessed_img, contours)
    all_cells.extend(cells)

# Преобразование всех ячеек в массив numpy для кластеризации
cell_data = np.array([cell[4] for cell in all_cells])

# Кластеризация ячеек
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(cell_data)
labels = kmeans.labels_

# Визуализация кластеров
def plot_clusters(cells, labels):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_cells = [cells[i][4] for i in range(len(cells)) if labels[i] == label]
        plt.figure(figsize=(10, 10))
        for i in range(min(25, len(cluster_cells))):
            plt.subplot(5, 5, i + 1)
            plt.imshow(cluster_cells[i].reshape(50, 50), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Cluster {label}')
        plt.show()

plot_clusters(all_cells, labels)
