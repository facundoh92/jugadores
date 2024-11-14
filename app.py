import streamlit as st
from PIL import Image
from utils import model_interp,vae_loaded,show_interp

import random

index1 = 14
index2 = 42

interp_result = model_interp(model = vae_loaded, index1 = index1, index2 = index2).unbind(0)
imgs = [img.permute(1,2,0).cpu() for img in interp_result]
show_interp(imgs,index1,index2, scale=2);