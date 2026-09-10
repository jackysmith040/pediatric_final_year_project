# Learning and mathematics guide
## Pediatric Counting with Interactive Computer Vision

### 1. The simple answer about MobileNet

A neural network classifier is a function that receives an image and returns probabilities for classes. In this project, the input is one standardized crop of a person and the output might be:

```text
adult: 0.82
child: 0.15
uncertain: 0.03
```

A **small classifier** is simply a classifier designed to use less computation than a large detector. It is not a replacement for the base person detector. The base detector answers **where is a person?** The crop classifier answers **does this particular crop look more like an adult or a child?**

MobileNet is one possible small classifier family. It uses efficient convolutional operations so that it can run with less computation. We do not need to invent a neural network from scratch. The safer approach is transfer learning: begin with a model that already knows general visual patterns, replace its final output layer with our classes, and train using our adult/child crops. PyTorch documents the same two broad strategies: fine-tune a pretrained network, or freeze most of it and train only a new final classifier layer [1]. MobileNet was designed for efficient vision on devices with limited compute [2].

The important distinction is this:

| Component | Question it answers | Output |
|---|---|---|
| Base YOLO detector | Where are visible people? | Bounding boxes |
| Crop classifier | Is this one crop adult or child? | Class label and confidence |
| Tracker | Which detection belongs to which temporary identity? | Track ID |
| Counting logic | Has this confirmed child already been counted? | Count update |

A small classifier is worth adding only as a controlled experiment. It should not be added because “more AI” sounds better. The experiment is: **does a lightweight crop classifier improve adult/child classification and final distinct-child counting compared with the current fine-tuned classifier?**

### 2. The mathematical story you can defend

Your thesis does not need advanced mathematics to be mathematically legitimate. The mathematical contribution is the formalization of a sequence of transformations, decisions, and measurable errors.

Let a video be a sequence of frames:

\[
V = (I_1, I_2, \ldots, I_T),
\]

where `I_t` is the image at frame `t` and `T` is the number of frames.

The base detector is a function:

\[
D(I_t) = \{(b_{tj}, s_{tj})\}_{j=1}^{n_t},
\]

where `b_tj` is a bounding box for detected person `j`, `s_tj` is the detector confidence, and `n_t` is the number of detected people in frame `t`.

A box can be represented as:

\[
b=(x_1,y_1,x_2,y_2),
\]

with width `w = x_2 - x_1` and height `h = y_2 - y_1`.

The crop function expands the box by a configurable margin `m`:

\[
C_m(b) = (x_1-mw,\ y_1-mh,\ x_2+mw,\ y_2+mh),
\]

followed by clipping to the image boundaries. This explains exactly what “give the classifier some context” means. The system can test `m = 0`, `0.10`, and `0.20` rather than hiding a magic number in code.

The resize function maps the crop into a fixed input size `S × S`. With letterboxing, the scale is:

\[
q = \min(S/w_c, S/h_c),
\]

where `w_c` and `h_c` are the crop dimensions. The resized dimensions are `(q w_c, q h_c)` and the remaining space is padding. This preserves the crop’s aspect ratio instead of stretching a tall person into a square.

The classifier is a function:

\[
P(y \mid z_t) = f_\theta(z_t),
\]

where `z_t` is the resized crop, `y` is the class, and `\theta` represents the learned model parameters. The class prediction is usually:

\[
\hat y_t = \arg\max_{y \in \{adult, child\}} P(y \mid z_t).
\]

Because one frame may be uncertain, the track-level label should use multiple observations. For a track `k` with recent classifier probabilities, a simple average is:

\[
\bar P_k(y) = \frac{1}{r}\sum_{i=1}^{r}P(y \mid z_{k,i}),
\]

and the track label is:

\[
\hat y_k = \arg\max_y \bar P_k(y).
\]

This is why a still person can be handled: time supplies repeated evidence even when movement is small.

The tracker creates a set of track IDs:

\[
\mathcal{T}_t = \{\tau_{t1}, \tau_{t2}, \ldots\},
\]

and associates new detections with existing tracks using spatial and possibly appearance similarity. The intersection-over-union of two boxes is:

\[
IoU(A,B)=\frac{|A\cap B|}{|A\cup B|}.
\]

High IoU means that two boxes occupy similar regions. A tracker can also use motion prediction, often represented by a state estimate such as position and velocity. You do not need to derive the Kalman filter in the defense unless asked; you can explain that it predicts where a temporarily missing person is likely to appear next.

The count is a set cardinality, not a sum over frames. Let `K_child` be the set of confirmed track IDs whose smoothed label is child. Then the distinct-child count is:

\[
N_{child}=|K_{child}|.
\]

This is the key mathematical distinction between occupancy and unique counting. If one child appears in 1,000 frames with the same ID, the count is still 1, not 1,000.

### 3. The lifecycle as a finite-state model

Each track has a state in:

\[
S=\{Tentative, Confirmed, Lost, Retired\}.
\]

The state changes according to observations:

| Current state | Condition | Next state |
|---|---|---|
| No track | Plausible detection appears | Tentative |
| Tentative | Repeated matching detections reach `N_confirm` | Confirmed |
| Tentative | Evidence disappears before confirmation | Retired |
| Confirmed | Detection is temporarily missing | Lost |
| Lost | Matching detection returns before timeout | Confirmed |
| Lost | Lost timeout expires | Retired |
| Confirmed | Child label becomes sufficiently reliable | Add ID to counted set once |

The transition rules are the formal replacement for vibe-coded conditions. `N_confirm`, detector thresholds, crop margin, and lost timeout are parameters. The harness should record them and compare their effects.

### 4. What can and cannot be inferred

If a baby is fully hidden behind a caregiver, the frame contains no visible child pixels. The system may preserve an already-established track through a short occlusion, but it cannot directly detect unseen information from that frame. This is an observability limitation, not merely a model weakness.

Therefore, classify outcomes as **visible**, **partially visible**, **temporarily occluded**, or **not observable**. This is more scientifically honest than forcing every frame into adult or child.

### 5. Defense answers

**Why use two stages?** The base detector is strong on the CCTV domain and finds visible people. The specialist classifier receives a standardized person crop, reducing the amount of irrelevant scene content and separating localization from adult/child classification.

**Why not train one model to do everything?** A single model may be possible, but the two-stage design is easier to inspect and lets us test which part fails: person localization, adult/child classification, identity association, or counting.

**Why use a small classifier?** It may reduce computation for repeated crop classification and offers a controlled comparison. Its value must be demonstrated using classification accuracy, latency, and final count error.

**Why not promise perfect carried-baby detection?** A fully hidden child is not observable in an RGB frame. The system can preserve a known identity during temporary occlusion, but it cannot prove a previously unseen hidden child exists without additional learned contextual evidence.

**What is the mathematical contribution?** The system formalizes frame-to-box detection, box-to-crop transformation, probabilistic classification, identity association, finite-state track management, and set-based distinct counting. It also defines measurable error quantities.

## References

[1]: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html "Transfer Learning for Computer Vision Tutorial"

[2]: https://arxiv.org/abs/1704.04861 "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"

[3]: https://docs.ultralytics.com/tasks/classify "Image Classification with Ultralytics YOLO"
