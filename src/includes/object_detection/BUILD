load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "object_detection_helper",
    srcs = ["object_detection_helper.cc"],
    hdrs = ["object_detection_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":bounding_box",
        ":common_dependencies",
        ":tflite_dependencies",
    ],
)

cc_library(
    name = "classifier_helper",
    srcs = ["classifier_helper.cc"],
    hdrs = ["classifier_helper.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":common_dependencies",
        ":tflite_dependencies",
    ],
)

cc_library(
    name = "non_max_supression",
    srcs = ["non_max_supression.cc"],
    hdrs = ["non_max_supression.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":bounding_box",
        ":common_dependencies",
    ],
)

cc_library(
    name = "bounding_box",
    srcs = ["bounding_box.cc"],
    hdrs = ["bounding_box.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "common_dependencies",
    deps = [
        "@com_github_gflags_gflags//:gflags",
        "@glog",
        "@linux_opencv//:opencv",
    ],
)

cc_library(
    name = "tflite_dependencies",
    deps = [
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/delegates/xnnpack:xnnpack_delegate",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)
