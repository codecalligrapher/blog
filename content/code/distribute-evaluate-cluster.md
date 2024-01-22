---
title: "Distribute Cluster Evaluation"
date: 2024-01-03T00:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
tags: ["clustering", "machine-learning"]
comments: true
showReadingTime: false
---


from https://www.databricks.com/notebooks/segment-p13n/sg_03_clustering.html 

```python
X_broadcast = sc.broadcast(X)
 
# function to train model and return metrics
def evaluate_model(n):
  model = KMeans( n_clusters=n, init='k-means++', n_init=1, max_iter=10000)
  clusters = model.fit(X_broadcast.value).labels_
  return n, float(model.inertia_), float(silhouette_score(X_broadcast.value, clusters))
 
 
# define number of iterations for each value of k being considered
iterations = (
  spark
    .range(100) # iterations per value of k
    .crossJoin( spark.range(2,21).withColumnRenamed('id','n')) # cluster counts
    .repartition(sc.defaultParallelism)
    .select('n')
    .rdd
    )
 
# train and evaluate model for each iteration
results_pd = (
  spark
    .createDataFrame(
      iterations.map(lambda n: evaluate_model(n[0])), # iterate over each value of n
      schema=['n', 'inertia', 'silhouette']
      ).toPandas()
    )
 
# remove broadcast set from workers
X_broadcast.unpersist()
```