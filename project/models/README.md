# Optional Models for Advanced Scanner Features

Place the following files in this folder to enable all advanced analysis features:

## Facial Landmark Localization (LBF)
- `lbfmodel.yaml`

## Demographic Analysis (Age/Gender via OpenCV DNN Caffe models)
- `age_deploy.prototxt`
- `age_net.caffemodel`
- `gender_deploy.prototxt`
- `gender_net.caffemodel`

If these files are missing, scanner still runs and attendance recognition still works.
Only the corresponding advanced overlays will show as unavailable.
