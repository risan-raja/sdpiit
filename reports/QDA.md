## QDA
(sklearn.linear_discriminant...QuadraticDA)

 - No hyper parameters to tune
 - Good performance for all classes
 - Can be good for Stacking Model.
 
```python
# ohe = OneHotEncoder(min_frequency=0.0001, handle_unknown='infrequent_if_exist', sparse=False,dtype=np.int32)
# X_train_t = ohe.fit_transform(nominal_data)
fig, ax = plt.subplots(3,1,figsize=(10,10))
i = 0
for categories in [nominal, ordinal]:
    X = final_data.loc[:,categories]
    y = final_data.target.to_numpy().reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=10, test_size=0.3)
    feature_names = categories
    model = QuadraticDiscriminantAnalysis(priors=class_priors,store_covariance=True,reg_param=0.0001 )
    # model = De
    # ovr_qda.fit(X_train, y_train)
    # y_pred = QDA_.fit(X_train, y_train)
    cv_= RepeatedStratifiedKFold(n_splits=3,n_repeats=5, random_state=10)
    pipe =  Pipeline(steps=[('polynomialwrapper',
                 PWrapper(feature_encoder=WOEEncoder())),
                (model.__class__.__name__,
                 QuadraticDiscriminantAnalysis(priors=class_priors,
                                               reg_param=0.0001,
                                               store_covariance=True))], memory=mem)
    def analyze_model(ax=ax,i = i,X=X,y=y, pipe=pipe, feature_names=categories):
        with parallel_backend('multiprocessing'):
            cv_model = cross_validate(pipe, X, y, cv = cv_,return_train_score=True,n_jobs=-1)
            ax[i].plot(np.arange(15), cv_model['test_score'], label=f'{feature_names[0].split("__")[0].upper()}  Data')
            ax[i].legend()
            ax[2].plot(np.arange(15), cv_model['test_score'], label=f'{feature_names[0].split("__")[0].upper()}  Data')
            ax[2].legend()
    analyze_model()
    y_pred = model.fit(X_train, y_train).predict(X_test)
    print(classification_report(y_test,y_pred))
    break
    i+=1
```

 - **__Nominal Data__** Performance
 
 
 
 
 ```
    precision    recall  f1-score   support

           0       0.48      0.31      0.38       382
           1       0.52      0.82      0.64       552
           2       0.35      0.04      0.07       205

    accuracy                           0.51      1139
   macro avg       0.45      0.39      0.36      1139
weighted avg       0.48      0.51      0.45      1139
 ```

 - **__Ordinal Data__** Performance
 
 
 
 ```
  precision    recall  f1-score   support

           0       0.65      0.63      0.64       382
           1       0.73      0.81      0.77       552
           2       0.33      0.26      0.29       205

    accuracy                           0.65      1139
   macro avg       0.57      0.56      0.57      1139
weighted avg       0.63      0.65      0.64      1139

```