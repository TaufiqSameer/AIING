import numpy as np;
import pickle;
import pandas as pd;
import streamlit as st;

from PIL import Image;

pickle_in = open("./classifier.pkl","rb");
classifier = pickle.load(pickle_in);

def welcome():
    return "Welcome";
