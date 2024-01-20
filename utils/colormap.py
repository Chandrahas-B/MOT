import numpy as np
import pickle

def generate_class_colors(num_classes= 91):
    # Generate random colors for each class
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    color_map = {i: tuple([int(color[0]), int(color[1]), int(color[2])]) for i, color in enumerate(colors)}
    return color_map

colormaps = generate_class_colors()

file = open('./utils/colormaps.pkl', 'wb')
pickle.dump(colormaps, file)
file.close()