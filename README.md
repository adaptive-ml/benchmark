# ML Benchmark tool

This tool is invoked from github action in https://github.com/adaptive-ml/adaptive

## Process

* Create a k8s job (supervisor)

Inside the job:
1. Create an argo App (adaptive-ml-bench)
1. Wait 4 app to be ready
1. Launch Benchmark scenarios
1. Collect result and send slack message
1. Delete argo App
