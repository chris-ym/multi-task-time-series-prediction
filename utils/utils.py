#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import os
import pickle

def col_append(columns,num_dup):
    '''
    return all columns, all columns deleted categroical 
    '''
    for i in range(num_dup):
        if i==0:
            column=list(columns)
        else:
            column=column+list(map(lambda x: x+'_'+str(i),columns))
    return column

def next_time_pred(df,target,date):
    df['new_target']=df.groupby(date)[target].shift(-1)
    df=df.dropna(subset=['new_target'])
    return df

def time_series_trans(pid,date,new_target,features,time_length):
    total_list=[]
    for pno2 in df[pid].unique():    
        print(str(list(df[pid].unique()).index(pno2)+1)+ '/'+ str(len(list(df[pid].unique())))+'........\n')
        df=temp[pno2][features]
        for date in df[date].unique():   
            print(str(list(df[date].unique()).index(date)+1)+ '/'+ str(len(list(df[date].unique())))+'........\n')
            date_df=df[df[date]==date].reset_index(drop=True)
            if len(date_df)<=1:
                print('Error: Time length of record of %s is less than 2 in the patient ID %s',(str(pno2),str(date)))
            for num in range(len(date_df)):
                #situation 1, the first record of the day
                if num == 0:
                    temp_list=list(pd.concat([date_df.drop([new_target],axis=1)[0:1]]*time_length, ignore_index=True).values.reshape(-1))
                    temp_list.append(date_df[new_target][num])             
                    if total_list==[]:                
                        total_list=[temp_list]
                    else:
                        total_list.append(temp_list)
                #situation 2, the transformation which have duplicate first record
                elif 1<=num<=time_length:   
                    temp_list=list(pd.concat([date_df.drop([new_target],axis=1)[0:1]]*(time_length-num), ignore_index=True).values.reshape(-1))
                    temp_list.extend(date_df.drop([new_target],axis=1)[1:num+1].values.reshape(-1))
                    temp_list.append(date_df[new_target][num])
                    total_list.append(temp_list)
                #situation 3, no duplicate record 
                else:
                    temp_list=list(date_df.drop([new_target],axis=1)[num-(time_length-1):num+1].values.reshape(-1))
                    temp_list.append(date_df[new_target][num])
                    total_list.append(temp_list)
        print('-'*30)            
        small_df=pd.DataFrame(total_list,columns=columnnn)           
        small_df.to_csv('new_fixed_length_partof.csv')

    print('Final csv file saved.')
    final_df=pd.DataFrame(total_list,columns=columnnn)
    final_df.to_csv('new_fixed_length_new.csv')
    return final_df
