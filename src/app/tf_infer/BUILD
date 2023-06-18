load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "tf_infer",
    srcs = ["tf_infer.cc"],
    deps = [
        "//src/includes/object_detection:bounding_box",
        "//src/includes/object_detection:classifier_helper",
        "//src/includes/object_detection:non_max_supression",
        "//src/includes/object_detection:object_detection_helper",
    ],
)

exports_files(
    srcs = [
        "hand_1.jpg",
    ],
)
