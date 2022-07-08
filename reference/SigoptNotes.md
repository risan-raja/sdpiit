# SigOpt Notes

### get the best Runs for the Experiment
```python
    MODEL_NAME = "OneVsRestClassifier(XGBoostClassifier)"
    DATASET_NAME = "Sklearn Wine"
    FEATURE_ENG_PIPELINE_NAME = "Sklearn Standard Scalar"
    PREDICTION_TYPE = "Multiclass"
    DATASET_SRC = "sklearn.datasets"

    def evaluate_xgboost_model(X, y,
                               number_of_cross_val_folds=5,
                               max_depth=6,
                               learning_rate=0.3,
                               min_split_loss=0):
        t0 = time.time()
        classifier = OneVsRestClassifier(XGBClassifier(
            objective = "binary:logistic",
            max_depth =    max_depth,
            learning_rate = learning_rate,
            min_split_loss = min_split_loss,
            use_label_encoder=False,
            verbosity = 0
        ))
        cv_accuracies = cross_val_score(classifier, X, y, cv=number_of_cross_val_folds)
        tf = time.time()
        training_and_validation_time = (tf-t0)
        return numpy.mean(cv_accuracies), training_and_validation_time

    def run_and_track_in_sigopt():
        (features, labels) = get_data()

        sigopt.log_dataset(DATASET_NAME)
        sigopt.log_metadata(key="Dataset Source", value=DATASET_SRC)
        sigopt.log_metadata(key="Feature Eng Pipeline Name", value=FEATURE_ENG_PIPELINE_NAME)
        sigopt.log_metadata(key="Dataset Rows", value=features.shape[0]) # assumes features X are like a numpy array with shape
        sigopt.log_metadata(key="Dataset Columns", value=features.shape[1])
        sigopt.log_metadata(key="Execution Environment", value="Colab Notebook")
        sigopt.log_model(MODEL_NAME)
        sigopt.params.max_depth = numpy.random.randint(low=3, high=15, dtype=int)
        sigopt.params.learning_rate = numpy.random.random(size=1)[0]
        sigopt.params.min_split_loss = numpy.random.random(size=1)[0]*10

        args = dict(X=features,
                    y=labels,
                    max_depth=sigopt.params.max_depth,
                    learning_rate=sigopt.params.learning_rate,
                    min_split_loss=sigopt.params.min_split_loss)

        mean_accuracy, training_and_validation_time = evaluate_xgboost_model(**args)

        sigopt.log_metric(name='accuracy', value=mean_accuracy)
        sigopt.log_metric(name='training and validation time (s)', value=training_and_validation

    %%run My_First_Run
    run_and_track_in_sigopt()   
    experiment = sigopt.create_experiment(
      name="Keras Model Optimization (Python)",
      type="offline",
      parameters=[
        dict(name="hidden_layer_size", type="int", bounds=dict(min=32, max=128)),
        dict(name="activation_function", type="categorical", categorical_values=["relu", "tanh"]),
      ],
      metrics=[dict(name="holdout_accuracy", objective="maximize")],
      parallel_bandwidth=1,
      budget=30,
    )

    for run in experiment.loop():
      with run:
        holdout_accuracy = execute_keras_model(run)
        run.log_metric("holdout_accuracy", holdout_accuracy)

    best_runs = experiment.get_best_runs()
```


### _Metrics Definition Example_

 - Format 1

```python

dict(
  name="name",
  objective="minimize",
  strategy="store",
  threshold= num
)
```

 - Format 2
 
```python
dict(
  name="name",
  objective="minimize",
  strategy="store",
  threshold= num
)  
```
 - Format 3
```python
dict(
  name="name",
  objective="minimize",
  strategy="store",
  threshold= num
)  
```


### Strategy:
```python
    strategy=["store", "optimize", "constraint"]
```

### Wrap Up Your Experiment
```python
    best_runs_list = sigopt.get_experiment(EXPERIMENT_ID).get_best_runs()
```



### Experimentation Config

 - Format 1
```python
sigopt.create_experiment(
  name="Single metric optimization with linear constraints",
  type="offline",
  parameters=[
            {'name': 'alpha', 'type': 'double', 'bounds': {'min': 0, 'max': 1}},
            {'name': 'beta', 'type': 'double', 'bounds': {'min': 0, 'max': 1}},
            {'name': 'gamma', 'type': 'double', 'bounds': {'min': 0, 'max': 1}}
  ],
  metrics=[{'name': 'holdout_accuracy', 'strategy': 'optimize', 'objective': 'maximize'}],
  linear_constraints=[
                      { 
                        'type': 'less_than',
                        'threshold': 1,
                        'terms': [
                            {'name': 'alpha', 'weight': 1},
                            {'name': 'beta', 'weight': 1},
                            {'name': 'gamma', 'weight': 1}
                        ]
                      },
                      {  'type': 'greater_than',
                        'threshold': 0,
                        'terms': [{'name': 'alpha', 'weight': 3}, {'name': 'beta', 'weight':5 }]}]

                    ,
                    parallel_bandwidth=1,
                    budget=30
                    )
```

```python

        {'conditionals': [{'name': 'num_conv_layers', 'values': ['1', '2', '3']}],
         'parameters': [{'name': 'layer_2_num_filters',
           'conditions': {'num_conv_layers': ['2', '3']},
           'type': 'int',
           'bounds': {'min': 100, 'max': 300}}]}
```
### Define and Set Up Parameter Space





#### Floating Point

```python

{'name': 'parameter',
 'type': 'double',
 'grid': [1e-05, 0.001, 0.33, 0.999],
 'transformation': 'log'}
```
#### Integer Point

```python

{'name': 'parameter', 'type': 'int', 'grid': [3, 4, 7, 10]}
```
#### categorical_values Point

```python

{'name': 'parameter',
 'categorical_values': ['val_a', 'val_b', 'val_c'],
 'type': 'categorical'}
```
