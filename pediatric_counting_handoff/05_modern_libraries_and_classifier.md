# Modern libraries and the compact classifier

## How does MobileNet learn adult versus child?

MobileNet begins with general visual knowledge learned from a large pretraining dataset. It does **not** begin with a built-in concept of adult or child. We provide labeled examples:

```text
person_crop_001.jpg → adult
person_crop_002.jpg → child
person_crop_003.jpg → adult
```

During training, the network produces probabilities, compares them with the correct label, calculates a loss, and adjusts its parameters to reduce the loss. Repeating this over many labeled crops teaches the model visual patterns that are useful for the particular dataset. A simplified classification loss for one crop is cross-entropy:

\[
L=-\sum_{c} y_c\log p_c,
\]

where `y_c` is 1 for the correct class and 0 otherwise, and `p_c` is the model’s predicted probability for class `c`.

Transfer learning means that we reuse the early visual feature extractor and replace or fine-tune the final classification layer for `adult` and `child`. With limited data, it is usually safer to start from pretrained weights and freeze most layers, then optionally unfreeze a small number of later layers after the baseline is working [1].

The classifier receives **one crop at a time**. The base detector finds the person region; the classifier does not need to find the person again. The system should train it on crops that resemble deployment crops, including the CCTV viewpoint, blur, distance, lighting, and partial visibility. If the training set contains only close-up, unobstructed images, the network may learn shortcuts that fail in the hospital room.

The label should be understood as a visual class for the project, not a precise biological age measurement. Cases where the child is fully hidden should be labeled `not_observable` or excluded from direct classification rather than forced into `adult` or `child`.

## Modern library map

| Responsibility | Recommended library | Why use it | Keep local or delegate? |
|---|---|---|---|
| Video reading/writing | OpenCV, optionally Supervision video helpers | Mature frame I/O and broad format support | Thin local wrapper |
| Detector inference | Existing Ultralytics package | Already matches the base YOLO model and project experiments | Delegate model inference |
| Standard detection representation | Supervision `Detections` | Converts outputs from Ultralytics and other frameworks into a common structure [2] | Delegate conversion; keep project metadata |
| Debug overlays | Supervision annotators | Boxes, labels, traces, and visual inspection without custom drawing [2] | Delegate rendering |
| Basic tracking baseline | Ultralytics ByteTrack or Supervision ByteTrack | Reusable tracker implementation with persistent IDs [2] [3] | Delegate association; keep lifecycle/count policy local |
| Alternative tracker experiments | Ultralytics trackers or BoxMOT | Compare trackers and optional ReID without writing each algorithm [3] [4] | Delegate algorithm; record configuration |
| Detector-agnostic custom tracking | Norfair | Lightweight tracker accepting generic detections and custom distance functions [5] | Optional research comparison |
| Ground-truth video annotation | CVAT | Keyframes, interpolation, persistent track labels, and manual correction [6] | Use for annotation, not runtime |
| Dataset inspection/error analysis | FiftyOne | Browse videos, labels, predictions, embeddings, and mistakes; integrates with CVAT [7] | Optional research tool |
| Standard MOT metrics | TrackEval | HOTA, CLEAR MOT, identity metrics, and benchmark formats [8] | Delegate standard metrics; add count metrics locally |
| Compact classifier training | PyTorch transfer learning or existing Ultralytics classification | Reuse pretrained features and train adult/child head [1] [9] | Choose one training path and document it |

## Recommended stack for the first stable release

Use OpenCV behind a small project wrapper for reading and writing frames. Use the existing Ultralytics person detector. Convert its outputs into Supervision’s `Detections` representation for consistent boxes and annotations. Use one ByteTrack implementation, not two at once. Keep the project’s lifecycle manager, crop transform, track-level label smoothing, uncertainty policy, and distinct-child counter in local code because these are the thesis-specific responsibilities.

Use CVAT to create persistent reference tracks and visibility/occlusion labels. Use TrackEval when the annotation and prediction files can be converted to its supported format. Use FiftyOne later to inspect false positives, false negatives, difficult crops, and model disagreements. Use BoxMOT only if a systematic tracker/ReID comparison is needed after the baseline. Norfair is a useful alternative when a custom distance function is central, but it is not required for the first release.

## What not to outsource blindly

A convenience library may offer a “count objects” function, but the thesis needs to explain exactly when an ID becomes countable, how long a lost ID is retained, and when an ID is retired. Therefore, use libraries for infrastructure while keeping the research policy visible in project code and logs.

Do not assume a library’s default classification crop is appropriate for this CCTV geometry. Ultralytics warns that square cropping can remove important content from extreme aspect-ratio images and suggests preserving proportions with a resize-based transform when needed [9]. The chosen crop margin, resize mode, input size, and padding must be configured, tested, and recorded.

## Starter experiment for MobileNet

Create a crop dataset from the base detector and manually review it. Each crop should have a label `adult`, `child`, or `not_observable/uncertain`. Split by video session, not by adjacent frames. Train a small pretrained MobileNet-family classifier in two stages:

1. Freeze the feature extractor and train the new adult/child classification head.
2. If validation results justify it, unfreeze only the final block and fine-tune with a low learning rate.

Compare it with the current fine-tuned classifier using the same held-out crops and the same crop transform. Report child recall, adult recall, macro-F1, confusion matrix, inference latency, and final distinct-child count error. If MobileNet is not better, that is a valid result; the thesis can conclude that the existing classifier is preferable under the tested conditions.

## References

[1]: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html "Transfer Learning for Computer Vision Tutorial"

[2]: https://supervision.roboflow.com/latest/ "Supervision documentation"

[3]: https://docs.ultralytics.com/modes/track "Ultralytics multi-object tracking documentation"

[4]: https://github.com/mikel-brostrom/boxmot "BoxMOT repository"

[5]: https://tryolabs.github.io/norfair/2.2/ "Norfair documentation"

[6]: https://www.cvat.ai/academy/track-mode "CVAT Track Mode"

[7]: https://docs.voxel51.com/index.html "FiftyOne documentation"

[8]: https://github.com/JonathonLuiten/TrackEval "TrackEval repository"

[9]: https://docs.ultralytics.com/tasks/classify "Ultralytics image classification documentation"
