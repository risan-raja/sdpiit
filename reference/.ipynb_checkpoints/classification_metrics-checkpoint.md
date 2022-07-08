Accuracy classification score.
```python 
metrics.accuracy_score(y_true, y_pred, *[, ...])
```

Compute Area Under the Curve (AUC) using the trapezoidal rule.
```python 
metrics.auc(x, y) 
```

Compute average precision (AP) from prediction scores.
```python 
metrics.average_precision_score(y_true, ...)
```


Compute the balanced accuracy.
```python 
metrics.balanced_accuracy_score(y_true, ...)
```


Compute the Brier score loss.
```python 
metrics.brier_score_loss(y_true, y_prob, *)
```


Build a text report showing the main classification metrics.
```python 
metrics.classification_report(y_true, y_pred, *)
```


Cohen's kappa: a statistic that measures inter-annotator agreement.
```python 
metrics.cohen_kappa_score(y1, y2, *[, ...])
```


Compute confusion matrix to evaluate the accuracy of a classification.
```python 
metrics.confusion_matrix(y_true, y_pred, *)
```


Compute Discounted Cumulative Gain.
```python 
metrics.dcg_score(y_true, y_score, *[, k, ...])
```


Compute error rates for different probability thresholds.
```python 
metrics.det_curve(y_true, y_score[, ...])
```


Compute the F1 score, also known as balanced F-score or F-measure.
```python 
metrics.f1_score(y_true, y_pred, *[, ...])
```


Compute the F-beta score.
```python 
metrics.fbeta_score(y_true, y_pred, *, beta)
```


Compute the average Hamming loss.
```python 
metrics.hamming_loss(y_true, y_pred, *[, ...])
```


Average hinge loss (non-regularized).
```python 
metrics.hinge_loss(y_true, pred_decision, *)
```


Jaccard similarity coefficient score.
```python 
metrics.jaccard_score(y_true, y_pred, *[, ...])
```


Log loss, aka logistic loss or cross-entropy loss.
```python 
metrics.log_loss(y_true, y_pred, *[, eps, ...])
```


Compute the Matthews correlation coefficient (MCC).
```python 
metrics.matthews_corrcoef(y_true, y_pred, *)
```


Compute a confusion matrix for each class or sample.
```python 
metrics.multilabel_confusion_matrix(y_true, ...)
```


Compute Normalized Discounted Cumulative Gain.
```python 
metrics.ndcg_score(y_true, y_score, *[, k, ...])
```


Compute precision-recall pairs for different probability thresholds.
```python 
metrics.precision_recall_curve(y_true, ...)
```


Compute precision, recall, F-measure and support for each class.
```python 
metrics.precision_recall_fscore_support(...)
```


Compute the precision.
```python 
metrics.precision_score(y_true, y_pred, *[, ...])
```


Compute the recall.
```python 
metrics.recall_score(y_true, y_pred, *[, ...])
```


Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
```python 
metrics.roc_auc_score(y_true, y_score, *[, ...])
```


Compute Receiver operating characteristic (ROC).
```python 
metrics.roc_curve(y_true, y_score, *[, ...])
```


Top-k Accuracy classification score.
```python 
metrics.top_k_accuracy_score(y_true, y_score, *)
```


Zero-one classification loss.
```python 
metrics.zero_one_loss(y_true, y_pred, *[, ...])
```

