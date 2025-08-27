import streamlit as st
from plt_one_addpt_onclick import plt_one_addpt_oncliick
from plt_one_addpt_oneclick2 import plt_one_addpt_onclick
import numpy as np
# %matplotlib widget
x_train = np.array([0., 1, 2, 3, 4, 5])
y_train = np.array([0,  0, 0, 1, 1, 1])

w_in = np.zeros((1))
b_in = 0

list1 = ["False", "True"]

value_pm = st.selectbox("Select payment methed", list1)
# "New folder/streamlit.py"
if value_pm == "False":
    plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=True)
