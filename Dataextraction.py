# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 16:37:33 2021

@author: Divyasha Pradhan
"""
import pandas as pd
from glob import glob

df_reader = pd.read_json('Clothing_Shoes_and_Jewelry.json',lines=True,nrows=50000000,chunksize=1000000)
# for cols in data:
#     cols
#     break

# cols.columns
print("Jason Read Finished")
Counter=1
print('Counter Initiated')

for chunk in df_reader:
    new_df=pd.DataFrame(chunk[['overall','reviewText','summary']])
    print("Into the Loop and Three Columns Extrcted to DataFrame")
    new_df5=new_df[new_df['overall']==5].sample(4000)
    new_df4=new_df[new_df['overall']==4].sample(4000)
    new_df3=new_df[new_df['overall']==3].sample(8000)
    new_df2=new_df[new_df['overall']==2].sample(4000)
    new_df1=new_df[new_df['overall']==1].sample(4000)
    
    new_df6=pd.concat([new_df1,new_df2,new_df3,new_df4,new_df5],axis=0, ignore_index=True)
    print("DataFrame Concatenation Done")
    
    new_df6.to_csv(str(Counter)+".csv",index=False)
    print("File Created - "+str(Counter)+".csv")               
    Counter+=1
    
# Step 2 : Combining extracted chunks into a single file
   
filenames =glob("*.csv")

dataframes = [pd.read_csv(f) for f in filenames]

finalframe = pd.concat(dataframes,axis=0, ignore_index= True)

finalframe.to_csv("balanced_reviews.csv", index=False)