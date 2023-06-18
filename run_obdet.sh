echo "$(tput setaf 1)$(tput bold)***Building the model!***"
bazel build src/app/tf_infer:tf_infer
echo "$(tput setaf 2)$(tput bold)***BUILT!***"
echo "$(tput setaf 2)$(tput bold)***Starting Predictions.***"
GLOG_logtostderr=$1 bazel-bin/src/app/tf_infer/tf_infer $2 $3
echo "$(tput setaf 1)$(tput bold)**STOPPED!**$(tput sgr0)"