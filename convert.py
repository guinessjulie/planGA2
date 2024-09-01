from PIL import Image
import os
import tkinter as tk
from tkinter import filedialog

# todo info to use parent selection for weights value of random.choices()
def min_max_scaling(values):
    min_val = min(values)
    max_val = max(values)

    scaled_fitness = [(f - min_val) / (max_val - min_val) for f in values]
    return scaled_fitness
def min_max_exp_scaling(values, n):
    min_max_scaled = min_max_scaling(values, n)
    adjusted_fitness = [sf ** n for sf in min_max_scaled]
    return adjusted_fitness