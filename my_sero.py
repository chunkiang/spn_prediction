import pandas as pd
import os

# Import xgboost
import xgboost as xgb
import numpy   as np
from sklearn.model_selection import train_test_split
from sklearn.metrics         import confusion_matrix, classification_report

# 这里定义使用哪个k-mer进行计算，越大的k值结果越准确，但是计算时间越长。
#path = '/data/home/zhaocj/spn_forcast/df数据整理/../7mer_output/RBIND_7mer'
#path = '/data/home/zhaocj/spn_forcast/df数据整理/../9mer_output/RBIND_9mer'
#path = '/data/home/zhaocj/spn_forcast/df数据整理/../11mer_output/RBIND-11mer'
#path = '/data/home/zhaocj/spn_forcast/df数据整理/../11mer_output/test5.txt'
path = '/data/home/zhaocj/spn_forcast/df数据整理/../10mer_output/RBIND'

# 这里定义使用哪个药物来进行计算
# kmer数据和药物数据合并的时候会根据drug里有没有‘_MIC’来判断是和ris表合并还是和mic表合并。
#drug = 'PEN_NM'
#drug = 'AMC_NM'
#drug = 'CRO_NM'
#drug = "ERY_NM"
#drug = "LVX_NM"
#drug = "MFX_NM"
#drug = "SXT_NM"
drug = "CLI_NM"

#drug = 'PEN_MIC'
#drug = 'AMC_MIC'
#drug = 'CRO_MIC'
#drug = "ERY_MIC"
#drug = "LVX_MIC"
#drug = "MFX_MIC"
#drug = "SXT_MIC"
#drug = "CLI_MIC"

try:
    df = pd.read_table(path, header=None, names=['SEQ', 'COUNT', 'ID'])
    id_number = pd.read_table('/data/home/zhaocj/spn_forcast/df数据整理/ID.txt')
    ris = pd.read_table('/data/home/zhaocj/spn_forcast/df数据整理/ris.txt')
    mic = pd.read_table('/data/home/zhaocj/spn_forcast/df数据整理/mic.txt')
    sero = pd.read_table('/data/home/zhaocj/seroba/summary.tsv', header=None, names=['ID', 'SEROTYPE', 'COMMENT'])
    sero2 = pd.read_table('sero2.txt')
    sero2['COMMENT'] = 'NaN'
    sero = pd.concat([sero, sero2])
except:
    print('数据导入失败！/(ㄒoㄒ)/~~')
    print()
else:
    print()
    print(path)
    print('数据导入成功！')
    #print(df.head())
    #print(id_number.head())
    #print(ris.head())
    print()


try:
    # 将长型数据转换为宽型数据
    df = df.pivot_table(index=['ID'], columns='SEQ', values='COUNT', fill_value=0).reset_index()

    # 将df的菌株号（ID）字段修改为字符型，表格pivot后是数值型和字符型混合的。
    for i in range(df.shape[0]):
        df.iloc[i, 0] = str(df.iloc[i, 0])

    # 选取id_number数据表里有的菌
    df = df.set_index('ID')
    df = df.loc[list(id_number['ID'])].reset_index()
   
    # 选取id_number数据表里有的血清型
    sero = sero.set_index('ID')
    sero = sero.loc[list(id_number['ID'])].reset_index()
 
    # 合并表型数据结果
    if 'MIC' in drug:
        mic = mic[['ID', drug]]
        # mic数值需要修改为数值型的，不然混淆矩阵无法实现
        for i in range(mic.shape[0]):
            mic.iloc[i, 1] = str(mic.iloc[i, 1])
        # 如果用血清型就用这个，然后把血清型用dummy来计算
        df = pd.merge(df, sero.iloc[:,:-1], how = 'left', on=['ID'])
        
        df = pd.merge(df, mic, how = 'left', on=["ID"])
    else:
        ris = ris[['ID', drug]]
        # 如果用血清型就用这个，然后把血清型用dummy来计算
        df = pd.merge(df, sero.iloc[:,:-1], how = 'left', on=['ID'])
        
        df = pd.merge(df, ris, how = 'left', on=["ID"])
 
    # 去掉第一列，第一列是菌株号
    ID = df.iloc[:,0]
    df = df.iloc[:,1:]

except:
    print('出错了，哪里错了，不知道。')
else:
    print('运气真好，做对了！已将长型数据转换为宽型数据。')
    print(df.head())
    print()

# 血清型作为dammy数据
df_dummies = pd.get_dummies(df['SEROTYPE'], drop_first=True)
# 三段数据分别是kmer数据，dummy数据，mic数据
df = pd.concat([df.iloc[:,:-2], df_dummies, df.iloc[:,-1]],  axis=1)


# 下边进入xgboost相关计算

# Create arrays for the features and the target: X, y
X, y = df.iloc[:,:-1], df.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

# 输出训练集和测试集的大小 
print('test_size=0.2')
print('训练集大小：')
print(X_train.shape)
print()
print('测试集大小：')
print(X_test.shape)
print()

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print()
print(drug)
print("accuracy: %f" % (accuracy))
print()

# 输出预测结果
print('预测结果')
print(preds)
print()

# 输出实际结果
print('实际结果')
print(np.array(y_test))
print()

# 输出菌株号
print('菌株号')
print(np.array(ID.loc[X_test.index]))
print()

# 混淆矩阵
print('混淆矩阵')
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
print()

# 导出结果
list_dict = {
    '菌株号': list(np.array(ID.loc[X_test.index])),
    '预测结果': list(preds),
    '实际结果': list(np.array(y_test))
}

pd.DataFrame(list_dict).to_csv(path.split('/')[-2].split('_')[0] + '_' +  drug + '_sero.csv')


