import pickle

with open('./utils/colormaps.pkl', 'rb') as pkl_file:
    colormaps = pickle.load(pkl_file)