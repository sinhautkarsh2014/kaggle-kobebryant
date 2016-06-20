import pandas as pd
from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
df=pd.read_csv('data.csv')
#renaming categorical data with int
df["action_type_ctg"]=df["action_type"].astype("category")
#print df["action_type_ctg"].cat.rename_categories(range(1,58)) # this doesn't happen permanently
df["action_type_ctg"]=df["action_type_ctg"].cat.rename_categories(range(1,58))
#print df["action_type_ctg"]

df["combined_shot_type_ctg"]=df["combined_shot_type"].astype("category")
df["combined_shot_type_ctg"]=df["combined_shot_type_ctg"].cat.rename_categories([1,2,3,4,5,6])


df["season_ctg"]=df["season"].astype("category")
df["season_ctg"]=df["season_ctg"].cat.rename_categories(range(1,21))

df["shot_type_ctg"]=df["shot_type"].astype("category")
df["shot_type_ctg"]=df["shot_type_ctg"].cat.rename_categories(range(1,3))

df["shot_zone_area_ctg"]=df["shot_zone_area"].astype("category")
df["shot_zone_area_ctg"]=df["shot_zone_area_ctg"].cat.rename_categories(range(1,7))

df["shot_zone_basic_ctg"]=df["shot_zone_basic"].astype("category")
df["shot_zone_basic_ctg"]=df["shot_zone_basic_ctg"].cat.rename_categories(range(1,8))

df["shot_zone_range_ctg"]=df["shot_zone_range"].astype("category")
df["shot_zone_range_ctg"]=df["shot_zone_range_ctg"].cat.rename_categories(range(1,6))

df["matchup_ctg"]=df["matchup"].astype("category")
df["matchup_ctg"]=df["matchup_ctg"].cat.rename_categories(range(1,75))

df["opponent_ctg"]=df["opponent"].astype("category")
df["opponent_ctg"]=df["opponent_ctg"].cat.rename_categories(range(1,34))

df["minutes_remaining"]=df["minutes_remaining"].astype("object")
df["seconds_remaining"]=df["seconds_remaining"].astype("object")

# segregating the training data from to-be predicted fields

df_traindat=df[df.shot_made_flag.notnull()]
df_preddat=df[df.shot_made_flag.isnull()]

# assigning X's & y's
Xtr_sparse=pd.DataFrame(enc.fit_transform(df_traindat[["action_type_ctg","combined_shot_type_ctg","season_ctg","shot_type_ctg","shot_zone_area_ctg","shot_zone_basic_ctg","shot_zone_range_ctg","matchup_ctg","opponent_ctg"]]).toarray())
Xtr_float=df_traindat[["shot_id","game_event_id","game_id","loc_x","loc_y","minutes_remaining","period","playoffs","seconds_remaining","shot_distance"]]

#print Xtr_float.shape
Xtr_float.index=Xtr_sparse.index

#print X_train.shape
#X_train=df_traindat[["action_type_ctg","combined_shot_type_ctg","season_ctg","shot_type_ctg","shot_zone_area_ctg","shot_zone_basic_ctg","shot_zone_range_ctg","matchup_ctg","opponent_ctg","shot_id","game_event_id","game_id","lat","loc_x","loc_y","lon","minutes_remaining","period","playoffs","seconds_remaining","shot_distance"]]
y_train=df_traindat["shot_made_flag"]
Xpr_sparse=pd.DataFrame(enc.transform(df_preddat[["action_type_ctg","combined_shot_type_ctg","season_ctg","shot_type_ctg","shot_zone_area_ctg","shot_zone_basic_ctg","shot_zone_range_ctg","matchup_ctg","opponent_ctg"]]).toarray())
#print Xpr_sparse.shape
Xpr_float=df_preddat[["shot_id","game_event_id","game_id","loc_x","loc_y","minutes_remaining","period","playoffs","seconds_remaining","shot_distance"]]
# assigning X's & y's
Xpr_sparse=pd.DataFrame(enc.fit_transform(df_traindat[["action_type_ctg","combined_shot_type_ctg","season_ctg","shot_type_ctg","shot_zone_area_ctg","shot_zone_basic_ctg","shot_zone_range_ctg","matchup_ctg","opponent_ctg"]]).toarray())
Xpr_float=df_traindat[["shot_id","game_event_id","game_id","loc_x","loc_y","minutes_remaining","period","playoffs","seconds_remaining","shot_distance"]]

#print Xpr_float.shape
Xpr_float.index=Xpr_sparse.index

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
sel = SelectKBest(chi2, k=175)
Xtr_sparse=pd.DataFrame(sel.fit_transform(Xtr_sparse,y_train))
Xpr_sparse=pd.DataFrame(sel.transform(Xpr_sparse))


k=Xpr_float['shot_id']
sel2 = SelectKBest(f_classif, k=10)
Xtr_float=pd.DataFrame(sel2.fit_transform(Xtr_float,y_train))
Xpr_float=pd.DataFrame(sel2.transform(Xpr_float))



X_train=pd.concat([Xtr_float,Xtr_sparse], axis=1)


X_pred=pd.concat([Xpr_float,Xpr_sparse], axis=1)

print X_train.shape
print X_pred.shape

#print X_pred.shape
# gaussian nb as model
from sklearn.naive_bayes import GaussianNB
#from sklearn.linear_model import LogisticRegression 
#from sklearn.calibration import CalibratedClassifierCV 
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier



#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.metrics import log_loss
#from sklearn.cross_validation import train_test_split

#Xtr,Xts,ytr,yts=train_test_split(X_train,y_train, test_size=0.00005)
###model=SVC(probability=True, verbose=True)
###model=CalibratedClassifierCV(method='isotonic') 
model=GaussianNB()
#model.fit(Xtr, ytr)
#ypr=model.predict_proba(Xts)
##model.fit(X_train,y_train)
##y_pred=model.predict_proba(X_pred)
#print log_loss(yts,ypr)
pred=[]
j=0
for i in k:
	model.fit(X_train[:i],y_train[:i])
	y_pred=model.predict_proba(X_pred[j])
	j=j+1
	print i
	pred.extend(y_pred[:,0])
	
out=pd.DataFrame()
out['shot_id']=k
out['shot_made_flag']=pred
#print out
out.to_csv('out.csv', index=False)












