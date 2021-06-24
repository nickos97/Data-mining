#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("healthcare-dataset-stroke-data.csv")
smoke=dataset.groupby('smoking_status')['id'].count()
gender=dataset.groupby('gender')['id'].count()
residence=dataset.groupby('Residence_type')['id'].count()
hyper=dataset.groupby('hypertension')['id'].count()
work=dataset.groupby('work_type')['id'].count()
married=dataset.groupby('ever_married')['id'].count()
heart=dataset.groupby('heart_disease')['id'].count()
smoke.plot(kind="bar")
gender.plot(kind="bar")
residence.plot(kind="bar")
hyper.plot(kind="bar")
work.plot(kind="bar")
married.plot(kind="bar")
heart.plot(kind="bar")
x = dataset.get('age')
#plt.hist(x, bins = 80)
#plt.show()

dataset.describe()
dataset.plot(x='avg_glucose_level',y='bmi',style='o',figsize=(27, 13))
plt.figure(figsize=(25, 12))


    # %%
