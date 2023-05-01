---
title: "Kedro 0.17.X Hook Spec"
date: 2023-05-01T17:00:00-00:00
ShowCodeCopyButtons: true
hideSummary: true
summary: false
keywords: ["python", "kedro"]
comments: true
showReadingTime: false
---
```python
from kedro.framework.hooks import hook_impl

class <hook_name>:

    def __init__(self):
        pass

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        pass

    @hook_impl
    def after_pipeline_run(self, run_params, pipeline, catalog):
        pass
    @hook_impl
    def on_pipeline_error(self, error, run_params, pipeline, catalog):
        pass

    @hook_impl
    def after_catalog_created(
        self,
        catalog,
        conf_catalog,
        conf_creds,
        feed_dict,
        save_version,
        load_versions,
        run_id,
    ):
        pass

    @hook_impl
    def before_node_run(self, node, catalog, inputs, is_async, run_id):
        pass

    @hook_impl
    def after_node_run(self, node, catalog, inputs, outputs, is_async, run_id):
        pass

    @hook_impl
    def on_node_error(self, error, node, catalog, inputs, is_async, run_id):
        pass
```