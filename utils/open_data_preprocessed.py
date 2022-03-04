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

# country (group)  AQI (2 target)
def next_time_pred(df,target,date):
    # target is a list that have multiple target
    for lbl in target:
        df[lbl+'_new']=df.groupby(date)[lbl].shift(-1)
    #df=df.dropna(subset=['new_target'])
    df=df.dropna()
    return df
  
def time_series_trans(temp,country,city,site,new_target,features,time_length):
    total_list=[]
    for pno2 in temp[country].unique():    
        print('Country: '+str(list(temp[country].unique()).index(pno2)+1)+ '/'+ str(len(list(temp[country].unique())))+'============\n')
        df=temp[temp[country] == pno2][features]

        for one_city in df[city].unique():   
            print('City: '+str(list(df[city].unique()).index(one_city)+1)+ '/'+ str(len(list(df[city].unique())))+'---------\n')
            city_df=df[df[city]==one_city].reset_index(drop=True)

            for one_site in city_df[site].unique():   
                print('Site: '+str(list(city_df[site].unique()).index(one_site)+1)+ '/'+ str(len(list(city_df[site].unique())))+'-------\n')
                date_df=city_df[city_df[site]==one_site].reset_index(drop=True)

                for num in range(len(date_df)):
                    print(str(num+1)+ '/'+ str(len(date_df))+'........\n')
                    #situation 1, the first record of the day
                    if num == 0:
                        temp_list=list(pd.concat([date_df.drop([new_target[0],new_target[1]],axis=1)[0:1]]*time_length, ignore_index=True).values.reshape(-1))
                        temp_list.append(date_df[new_target[0]][num]) 
                        temp_list.append(date_df[new_target[1]][num]) 
                        if total_list==[]:                
                            total_list=[temp_list]
                        else:
                            total_list.append(temp_list)
                    #situation 2, the transformation which have duplicate first record
                    elif 1<=num<time_length-1:   
                        temp_list=list(pd.concat([date_df.drop([new_target[0],new_target[1]],axis=1)[0:1]]*(time_length-num), ignore_index=True).values.reshape(-1))
                        temp_list.extend(date_df.drop([new_target[0],new_target[1]],axis=1)[1:num+1].values.reshape(-1))
                        temp_list.append(date_df[new_target[0]][num])
                        temp_list.append(date_df[new_target[1]][num])
                        total_list.append(temp_list)
                    #situation 3, no duplicate record 
                    else:
                        temp_list=list(date_df.drop([new_target[0],new_target[1]],axis=1)[num-(time_length-1):num+1].values.reshape(-1))
                        temp_list.append(date_df[new_target[0]][num])
                        temp_list.append(date_df[new_target[1]][num])
                        total_list.append(temp_list)
        print('-'*30)            
        small_df=pd.DataFrame(total_list,columns=time_col)           
        small_df.to_csv('final_processed_partof.csv')
        
    print('Final csv file saved.')
    final_df=pd.DataFrame(total_list,columns=time_col)
    final_df.to_csv('final_processed_pollution.csv')

    return final_df
