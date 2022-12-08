---
title: "XGBoost Evaluation Classes"
date: 2022-12-08T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["python", "xgboost"]
comments: true
showReadingTime: false
toc: false 
---

```python
import abc
from sklearn import metrics
import xgboost as xgb
import numpy as np

class BaseEval(metaclass=abc.ABCMeta):
    '''Base class for creating eval_metrics for XGBoost supporting binary classification threshold
    
    metric() must be overridden
    
    '''
    def __init__(self, thresh:float=0.5):
        self.thresh = thresh

    def __call__(self, predt:np.ndarray, dtest:xgb.DMatrix):
        y_thresh = deepcopy(predt)
        y_thresh[y_thresh > self.thresh] = 1
        y_thresh[y_thresh <= self.thresh] = 0 

        return 'recall', self.metric(dtest.get_label(), y_thresh) 

    @abc.abstractmethod
    def metric(self, y_true, y_pred):
        '''Define evaluation function here'''
        pass


class RecallEval(BaseEval):
    def metric(self, y_true, y_pred):
       
        return metrics.recall_score(y_true, y_pred) 


class PrecisionEval(BaseEval):
    def metric(self, y_true, y_pred):
       
        return metrics.precision_score(y_true, y_pred) 


class F2Eval(BaseEval):
    def metric(self, y_true, y_pred):
       
        return metrics.fbeta_score(y_true, y_pred, beta=2.0) 
```

Used as follows:
```python

eval_fn = F2Eval(0.6) # example threshold

booster = xgb.train(
    dtrain=dtrain,
    num_boost_round=100,
    feval=eval_fn,
    evals=[(dtrain, 'train'), (dvalid, 'eval')],
    verbose_eval=True
)
```