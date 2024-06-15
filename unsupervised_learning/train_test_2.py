import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from images_preparing import load_images, preprocess_image

def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def extract_grid_and_cells(image, contours):
    cells = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 25 and h > 25:  # filter very small items
            cell = image[y:y+h, x:x+w]
            cell = cv2.resize(cell, (50, 50))  # unify cells
            cells.append((x, y, w, h, cell.flatten(), cv2.contourArea(contour), w / h))
    return cells

def find_lines(image):
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=5)
    return lines

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

# extra cluster 1 grouping
def reclassify_cluster_1_in_images(image_cells, labels, all_cells):
    start_idx = 0
    for cells in image_cells:
        cluster_1_indices = [i for i in range(start_idx, start_idx + len(cells)) if labels[i] == 1]
        if len(cluster_1_indices) > 1:
            largest_area_idx = max(cluster_1_indices, key=lambda i: all_cells[i][5])
            for idx in cluster_1_indices:
                if idx != largest_area_idx:
                    labels[idx] = 2  # move to cluster 2
        start_idx += len(cells)

def clustering():
    image_folder = 'unsupervised_learning/board_images/'
    images = load_images(image_folder)

    # extracting cell features from all images
    all_cells = []
    image_cells = []
    for img in images:
        preprocessed_img = preprocess_image(img)
        contours = find_contours(preprocessed_img)
        cells = extract_grid_and_cells(preprocessed_img, contours)
        all_cells.extend(cells)
        image_cells.append(cells)

    # convert all cells to numpy array for clustering
    cell_data = np.array([cell[4] for cell in all_cells])

    # cell clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(cell_data)
    labels = kmeans.labels_

    reclassify_cluster_1_in_images(image_cells, labels, all_cells)

    return {"all_cells": all_cells, "labels": labels}


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


clusters = clustering()
plot_clusters(clusters["all_cells"], clusters["labels"])
