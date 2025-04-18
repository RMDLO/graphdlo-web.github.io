---
abstract: |
  This work introduces GraphDLO, a graph-based learning framework for
  predicting the future trajectories of Deformable Linear Objects (DLOs)
  under both prehensile (grasping) and non-prehensile (pushing)
  interactions. A data set collected over 300 hours of interactions
  among each of three distinct rope objects is collected remotely using
  a cloud robotics platform and automatically labeled with rope and
  gripper state information from perception. A Graph Neural Network
  (GNN) is trained on this dataset to take as input the current rope
  state and a gripper trajectory to predict a trajectory of future rope
  states. The GraphDLO-predicted trajectories exhibit close qualitative
  agreement with ground truth trajectories across a prediction horizon
  of up to ten steps, demonstrating its potential for accurate
  long-horizon prediction in deformable object manipulation.
author:
- |
  Holly Dinkel^1^, Muhammad Zahid^2^, Bhumsitt Pramuanpornsatid^1^,\
  Brian Coltin^3^, Trey Smith^3^, Florian Pokorny^2^, and Timothy
  Bretl^1^\
  [^1] [^2] [^3]
bibliography:
- IEEEabbrv.bib
- references.bib
title: "**GraphDLO: Graph-Based Neural Dynamics for Deformable Linear
  Object Trajectory Prediction**"
---

# Introduction {#sec:intro}

Robots interact with increasingly diversified objects in the real world.
Manipulation robots--including humanoids, manipulation arms,
manipulation drones, and other dexterous devices--will soon be
ubiquitous, and the types of objects with which they interact must
become better understood for the automation to be worthwhile. One
category of manipulable objects present in nearly every human
environment is Deformable Linear Objects (DLOs). Future autonomous
manipulation systems may manipulate DLOs into desired shapes to route
cables [@zhou2020cable], suture wounds [@pedram2021suture], install
wires [@lagneau2020shapecontrol], knit or braid
string [@xiang2023multidlo], or tie
knots [@lu2016dynamic; @dinkel2024knotdlo]. One open problem specific to
DLO shape control with dexterous robots is trajectory prediction. This
problem is difficult because the shape of the entire object may change
as the robot grasps and moves only one part of it, where the magnitude
of the change in shape at points along the object typically depends on
their distances to the rigidly grasped
point [@berenson2013manipulation; @ruan2018rigidity]. To address this
problem, this work makes the following listed contributions:

1.  A graph-based dynamics model that explicitly encodes the grasped
    region of the deformable linear object (DLO), enabling accurate
    prediction of future DLO states given the current configuration and
    a planned gripper trajectory.

2.  A scalable framework for automatic labeling of planar rope
    configurations during real-world robotic manipulation, implemented
    on a cloud robotics platform to facilitate large-scale data
    collection for learning deformable object dynamics.

3.  A set of ablation studies highlight the trade-offs between model
    performance and prediction horizon, providing insights into the
    temporal limits of dynamics-based forecasting for DLOs.

# Related Work

![**Predicting DLO Trajectories with GraphDLO.** GraphDLO predicts the
future trajectory of a DLO, $\hat{\bm{\mathrm{X}}}^{t+1:t+T+1}$, given
its current state $\bm{\mathrm{X}}^t$ and a sequence of actions
$\bm{\mathrm{a}}^{t:T}$. Predicted states are visualized as filled
circles, where larger and brighter circles denote states closer in time,
and smaller, darker circles represent states further in the future. The
color gradient of DLO states is scaled from teal to gold based on the
node order, indicating the structure in the representation of the DLO.
Gripper actions are visualized as planar Cartesian axes at each step,
with the red $x$-axis and the green $y$-axis indicating the gripper
orientation. ](figures/first-page){#fig:first-page
width="\\columnwidth"}

![**Collecting Data at Scale with CloudGripper.** Three grippers from
the CloudGripper cloud robotics platform are used to collect interaction
data for three different DLOs. In each workcell, a base-mounted camera
captures an occlusion-free bottom-up view of the DLO resting on a
transparent plexiglass plate. A 3D-printed enclosure surrounds the
manipulation area to constrain the rope within the gripper's workspace
during interaction.](figures/cloudgripper){#fig:robots
width="\\textwidth"}

<figure id="fig: distribution">
<p><img src="figures/durations" alt="image" /> <img
src="figures/lengths" alt="image" /> <img src="figures/objects"
alt="image" /></p>
<figcaption><strong>Estimating Large-Scale Data Statistics.</strong>
(Top left) Each interaction episode spans approximately 10 minutes and
consists of up to 100 commanded gripper waypoints. (Top right) During
automated data collection, rope length is estimated in each image as a
proxy for rope labeling accuracy. The resulting distribution of measured
rope lengths across all episodes remains tightly concentrated,
indicating consistent and reliable labeling. (Bottom) Three ropes of
approximately equal length but varying thickness are used throughout
data collection and model training.</figcaption>
</figure>

Graph-based representations enable local state information sharing
through connected edges, making them well-suited for modeling the
dynamics of non-rigid and granular
objects [@li2019learning; @zhang2024adaptigraph; @longhini2024clothsplatting; @gu2025learning].
Prior work on learning object dynamics has primarily focused on
revealing properties such as inertial parameters or friction through
interaction [@xu2019densephysnet; @bianchini2023], or applying
analytical techniques like mass-spring systems and position-based
dynamics (PBD) to evolve object shape over
time [@lloyd2007identification; @macklin2014pbd; @bender2015position; @yin2021domanipulation].
In contrast, GraphDLO learns to directly predict rope
trajectories---tasks that traditionally relied on numerically
integrating classical dynamics models, often with trade-offs between
accuracy and computational cost. Accurate prediction of DLO trajectories
supports a range of downstream applications, including shape control and
planning [@moll2006path; @yu2023shapecontrol; @yu2022shapecontrol; @yu2023generalizable; @gu2025learning],
digital twin development [@jiang2025phystwin; @abou2025real], and
physics-informed state
estimation [@schulman2013deformable; @tang2017state; @longhini2024clothsplatting; @abouchakra2024physically; @zhang2024dynamic].
Predicting DLO trajectories also enables longer-horizon planning in
environments with obstacles or interaction
constraints [@mitrano2021learning; @huang2023deformable].

Although recent works have demonstrated success in learning deformable
dynamics, they predominantly rely on synthetic datasets generated in
physics-based
simulators [@hu2019taichi; @hu2019difftaichi; @macklin2014pbd; @macklin2022warp].
The models trained on synthetic data are deployed directly on data
acquired in the real world for inference, and it is still unclear how
transferrable this approach is when more complex environment effects
such as non-prehensile interaction with environment obstacles are
introduced. Advances in cloud robotics provide a promising alternative
by enabling large-scale, real-world data collection through distributed
fleets of identical robots operating in synchronized
environments [@zahid2024cloudgripper]. This capability is especially
valuable for gathering datasets involving rich contact dynamics or
complex deformable
behaviors [@hoque2022learning; @brohan2022rt; @jin2024pushing; @ahn2024autort].
Additionally, recent topology-grounded developments in DLO perception
and tracking enable automatic labeling of such data at
scale [@keipour2022efficient; @xiang2023trackdlo].

While many robot learning approaches for deformables have focused on
non-prehensile pushing
interactions [@wang2023pile; @zahid2024cloudgripper; @zhang2024adaptigraph; @zhang2024dynamic3dgs],
key challenges remain unaddressed. One such gap is preserving
topological consistency, which is essential for many DLOs that are
visually or functionally asymmetric along their length. Representing a
DLO as an unordered set of points can lead to ambiguities where point
order inverts between time steps despite the object itself remaining
unchanged, particularly under rotation. Furthermore, non-prehensile
manipulation does not capture rotational dynamics within a grasp. This
work directly addresses these limitations by incorporating grasp-aware
modeling, ordered graph structures, and real-world physical interactions
into deformable object prediction.

# The GraphDLO Method

![**Automating DLO Prehensile Interactions.** The data generation
process with CloudGripper automates prehensile manipulation of a rope
across hundreds of interaction episodes. Dataset collection for a single
episode proceeds as follows: **Estimate.** The configuration of the DLO
is first estimated in image space as an ordered sequence of pixel
coordinates, captured from a base-mounted camera that provides an
unobstructed bottom-up view [@keipour2022efficient; @xiang2023trackdlo].
**Map Pixel-Position.** These pixel coordinates are mapped to the
gripper frame using the calibration from pixels in the image space of
the base camera to positions in the workspace of the gripper. **Grasp.**
A planar grasp is planned by uniformly sampling a point along the rope
and computing a grasp pose based on the local geometry around the
selected point. **Explore.** Once the rope is grasped, the robot
executes a sequence of randomized planar translations and rotations,
enabling the generation of diverse rope configurations under realistic
manipulation. After each executed gripper waypoint, the system records
the estimated DLO state in both image and gripper space, the base camera
image, and the position of the gripper in both image and gripper
frames.](figures/data-generation){#fig: data-generation
width="\\textwidth"}

The GraphDLO model predicts a trajectory of future states of a DLO based
on its current state, the grasp location, and the planned gripper
trajectory as shown in Figure [1](#fig:first-page){reference-type="ref"
reference="fig:first-page"}. In contrast to prior approaches that use on
past object states and gripper motions to forecast the next state,
GraphDLO relies on the Markov assumption and models the future states as
dependent solely on the current
state [@zhang2024dynamic3dgs; @zhang2024adaptigraph]. This simplifies
the input representation and reduces the dimensionality of the model.

## Graph Model

For a planar DLO represented as a graph of $N$ nodes and $N-1$ edges in
two dimension, the GraphDLO algorithm learns to predict the 2D
trajectories of nodes as

$$\hat{\bm{\mathrm{X}}}^{t+1:t+T+1} = f(\bm{\mathrm{X}}^{t}, \bm{\mathrm{g}}^{t}, \bm{\mathrm{a}}^{t:t+T}),$$

where
$\hat{\bm{\mathrm{X}}}^{t+1:T+1} \subset [0, 1] ^{T \times N \times 2}$
is the predicted trajectory,
$\bm{\mathrm{X}}^{t} \subset [0, 1]^{N \times 2}$ is the estimated DLO
state, $\bm{\mathrm{g}}^t \subset \{0, 1\}^{N}$ encodes which node is
grasped such that $\sum_i \bm{\mathrm{g}}_i^t = 1$, and
$\bm{\mathrm{a}}^t = \{x_g^t, y_g^t, \theta_g^t\} \subset [0, 1]^{1 \times 3}$
is the gripper state. All model inputs and outputs are normalized to
prevent features with larger scales from dominating in backpropagation
to improve convergence speed and stability. The superscript $t:t+T$
indicates a temporally consecutive sequence indexed at start time $t$
and with horizon $T$, so

$$\begin{matrix}
    \hat{\bm{\mathrm{X}}}^{t+1:t+T+1} = \left[\hat{\bm{\mathrm{X}}}^{t+1}, \dots, \hat{\bm{\mathrm{X}}}^{t+T+1}\right]^{\intercal} \\
    \bm{\mathrm{a}}^{t:t+T} = \left[\bm{\mathrm{a}}^t, \dots, \bm{\mathrm{a}}^{t+T}\right]^{\intercal}
\end{matrix}.$$

This work uses $K$-hop message passing to model interactions in the
graph. First, the degree matrix $D$ is computed from the adjacency
matrix $\bm{\mathrm{A}}$ and $\mathcal{N}(i)$, the set of neighbors for
node $i$, where each diagonal element $D_{ii}$ represents the number of
neighbors for node $i$, computed as
$$D_{ii} = \sum_{j \in \mathcal{N}(i)} A_{ij}.$$ The adjacency matrix
$\bm{\mathrm{A}}$ can cause large variations in node features after
aggregation, and nodes with many connections (high $D_{ii}$) can
dominate. To stabilize aggregation, symmetric normalization is applied
to the adjacency matrix as $$\label{eq: symm-norm-adj}
    \tilde{\bm{\mathrm{A}}} = \bm{\mathrm{D}}^{-\frac{1}{2}}\bm{\mathrm{A}}\bm{\mathrm{D}}^{-\frac{1}{2}}$$
where $\bm{\mathrm{D}}^{-\frac{1}{2}}$ is the inverse square root of the
degree matrix and is computed as
$$D_{ii}^{-\frac{1}{2}} = \frac{1}{\sqrt{D_{ii}}},$$ where if
$D_{ii} = 0$, $D_{ii}^{-\frac{1}{2}}$ is set to $0$ to avoid division by
zero. If an edge exists between node $i$ and node $j$, its weight is
scaled by $D_{ii}^{-\frac{1}{2}}D_{jj}^{-\frac{1}{2}}$ to distribute
information evenly across all nodes. After computing
$\tilde{\bm{\mathrm{A}}}$, the node states are updated for
$k = 1,\dots, K$ as

$$\bm{\mathrm{x}}_i^{k+1} = \sum_{j \in \mathcal{N}(i)} \tilde{\bm{\mathrm{A}}}_{ij}\bm{\mathrm{x}}_j^k.$$

After message passing, the feature vector
$\bm{\mathrm{F}}^t = [\bm{\mathrm{X}}^t, \bm{\mathrm{g}}^t, \bm{\mathrm{a}}^{t:t+T}]$
is passed through a neural network with hidden layers
$\bm{\mathrm{h}}_1 \subset [0, 1]^{T, N \times 2 + 3 \times T}$ with
structure

$$\begin{matrix}
    \bm{\mathrm{h}}_1 = ReLU(\bm{\mathrm{W}}_1\bm{\mathrm{F}}^t + \bm{\mathrm{b}}_1) \\
    \bm{\mathrm{h}}_2 = ReLU(\bm{\mathrm{W}}_2\bm{\mathrm{h}}_1 + \bm{\mathrm{b}}_2) \\
    \hat{\bm{\mathrm{X}}}^{t+1:t+T+1} = \bm{\mathrm{W}}_3 \bm{\mathrm{h}}_2 + \bm{\mathrm{b}}_3
    \end{matrix}.$$

## Loss Function

The loss function is selected to balance the desire for the model to
learn DLO states that are low in distance to the target states as well
as shapes that are similar in geometry to the target shapes. This is
achieved by combining the Mean-Squared Error (MSE) with the contrastive
Cosine Embedding (CE) loss functions. The MSE loss is

$$\begin{gathered}
    \mathcal{L}_{MSE}(\hat{\bm{\mathrm{X}}}^t, \bm{\mathrm{X}}^t) = \frac{1}{N} \sum_{i=1}^N \left(\hat{\bm{\mathrm{x}}}_i^t, \bm{\mathrm{x}}_i^t\right)^2.
\end{gathered}$$

The CE loss is given for margin $\lambda$ by

$$\mathcal{L}_{CE}(\hat{\bm{\mathrm{X}}}^t, \bm{\mathrm{X}}^t, \bm{\mathrm{l}}^t) = \left\{ 
    \begin{matrix} 1 - S_C (\hat{\bm{\mathrm{X}}}^t, \bm{\mathrm{X}}^t) \mid l^t = 1 \\
    \text{max}(0, S_C (\hat{\bm{\mathrm{X}}}^t, \bm{\mathrm{X}}^t) - \lambda) \mid l^t = -1
    \end{matrix}
    \right.$$

where the label, $l^t \in \bm{\mathrm{l}}^t \subset \{-1, 1\}^N$,
encourages the predicted and target vectors to be as similar as possible
for $l=1$ and penalizes the proximity of the prediction and target for
$l=-1$, and the cosine similarity, $S_C$ is

$$S_C(\hat{\bm{\mathrm{X}}}^t, \bm{\mathrm{X}}^t) := cos(\bm{\mathrm{\Theta}}^t) = \frac{\hat{\bm{\mathrm{X}}}^{t} \cdot \bm{\mathrm{X}}^t}{\| \hat{\bm{\mathrm{X}}}^t \| \| \bm{\mathrm{X}}^t \|}.$$

The final loss function is

$$\mathcal{L} = \alpha\mathcal{L}_{MSE} + \left(1-\alpha\right)\mathcal{L}_{CE 
 },$$

for hyperparameter $\alpha$ weighting each loss component.

# Data Collection

![**Visualizing GraphDLO Predictions.** Each panel shows an initial rope
configuration (opaque rope) at $t$. The GraphDLO-predicted trajectory
for $t+1:t+T+1$ is overlaid as dots, and the ground truth state at
$t+1+5$ and $t+1+10$ (for $T=10$) is overlaid as semi-transparent. These
predictions predictions demonstrate GraphDLO's ability to closely align
with real-world trajectories for three different
ropes.](figures/demo){#fig: demo width="70%"}

Over 300 hours of prehensile and non-prehensile interactions between a
gripper and each of three cotton ropes were collected using the
CloudGripper cloud robotics platform shown in
Figure [2](#fig:robots){reference-type="ref" reference="fig:robots"}.
While all the ropes share similar lengths, they differ in thickness and
stiffness as shown in
Figure [3](#fig: distribution){reference-type="ref"
reference="fig: distribution"}. To break object symmetry, one tip of
each rope was marked with red tape. The data collection process was
automated to minimize human supervision. Initial object and tip masks
were obtained using color and contour segmentation with object-specific
thresholding, followed by refinement with the the Segment Anything Model
(SAM) 2 predictor in instances where the length of the skeletonized mask
fell outside an object-specific threshold length [@ravi2024sam]. Given a
binary object segmentation mask
$\mathcal{M}^t \in \{0, 1\}^{H \times W}$, a deformable one-dimensional
object routing algorithm was used to skeletonize the mask, extract
connected chains from the skeleton, and sample $N$ evenly-distributed
nodes. Each node is represented as
$\bm{\mathrm{x}}_i^t \in \bm{\mathrm{X}}^t \subset \left[0, 1\right]^{N \times 2}$
in Cartesian coordinates and
$\left(v_i^t, u_i^t\right) \in \mathcal{I}^t\left(v_i^t, u_i^t\right) \subset \{0, \dots, H - 1\} \times \{0, \dots, W - 1\}^N$
in pixel space along the
skeleton [@keipour2022efficient; @xiang2023trackdlo]. The length of the
rope serves as a proxy for segmentation quality and was used to
constrain rope state labels during data collection. The distributions of
rope lengths in pixel coordinates for the three objects are shown in
Figure [3](#fig: distribution){reference-type="ref"
reference="fig: distribution"}.

To enable pixel-to-position transformations, planar hand-eye extrinsic
calibration was performed using a red calibration cube translated
through a dense grid of gripper positions. Each Cartesian gripper
location was mapped to the corresponding pixel centroid of the cube,
forming lookup tables for bidirectional interpolation between pixel and
world coordinates. This calibration is limited to planar mappings, so
all rope configurations were constrained to lie in the same plane.
During interaction, the gripper grasp point is selected by sampling a
node index $i_g \sim \mathcal{U}(0, N-1)$, where the pixel coordinate of
the grasp node is $\left(u_{i_g}, v_{i_g}\right)$. The grasp orientation
is computed as $\theta_g = \text{mod}(\theta_{i_g}, \pi)$, where
$\theta_{i_g}$ is the local rope orientation at node $i_g$, ensuring
valid orientations for the CloudGripper hardware. The gripper executes
the grasp by rotating to $\theta_{i_g}$ and moving to
$\left(x_{i_g}, y_{i_g}\right) = \mathcal{M}^{-1}\left(u_{i_g}, v_{i_g}\right)$
before closing. It then follows a sequence of randomly sampled
waypoints, interpolating each transition into 10 intermediate poses. At
each waypoint, rope and gripper states (in both pixel and Cartesian
space) along with synchronized images and timestamps are recorded. The
full data generation process for one episode is summarized in
Figure [4](#fig: data-generation){reference-type="ref"
reference="fig: data-generation"}.

# Demonstration and Limitations {#sec:demonstration}

The performance of the GraphDLO algorithm is demonstrated on trajectory
predictions for the thin, standard, and thick ropes. As shown in the
qualitative results shared in Figure
[5](#fig: demo){reference-type="ref" reference="fig: demo"}, GraphDLO
accurately predicts the rope trajectories sampled from the validation
data.

However, the model occasionally exhibits uncertainty in predicting DLO
shapes, particularly near where the rope contacts the fixed enclosure.
This may stem from the fact that the enclosure is not explicitly
represented in the training data or modeled within GraphDLO. Future work
could explore incorporating the enclosure as part of the action or
environmental state to assess its effect on training efficiency and
prediction accuracy. Additionally, the validity of the Markov assumption
warrants further investigation for objects that accumulate internal
energy---such as stiff deformable materials or granular media---where
system dynamics may exhibit temporal dependencies. In the quasi-static
demonstrations considered here, the Markov assumption holds reasonably
well, but it may not generalize to more dynamic manipulation tasks.

A further limitation of the current method is the absence of 3D shape
information. Future work could enhance spatial perception by
stereo-matching the top and base cameras of the CloudGripper platform,
incorporating foundation models such as Depth Anything or
FoundationStereo for depth
estimation [@yang2024depth_anything_v1; @yang2024depth_anything_v2; @wen2025foundationstereo].
This would enable the development of a 3D state estimator capable of
capturing non-planar object configurations, facilitating downstream
tasks such as knot tying or winding around a winch.

# Conclusion {#sec:conclusion}

This work introduced GraphDLO, a graph-based learning framework for
predicting the future trajectories of a DLO given its current state,
where it was grasped, and a gripper trajectory. By using this Markov
assumption, GraphDLO simplifies the prediction problem while maintaining
accuracy in quasi-static manipulation. The GraphDLO model was trained
using over 300 hours of diverse prehensile and non-prehensile planar
interactions collected autonomously with the CloudGripper platform,
featuring three ropes with varying physical properties. Current
limitations highlight opportunities for future work, including explicit
modeling of workspace constraints, investigating the limits of the
Markov assumption in more dynamic settings, and incorporating 3D shape
estimation through stereo depth models. These directions will move us
closer to generalizable, structured robot learning for real-world
deformable object manipulation tasks such as knot tying, cable routing,
and textile handling.

# Acknowledgments {#acknowledgments .unnumbered}

The authors thank João Marcos Correia Marques for feedback on the
manuscript, the teams developing the open-source software used in this
project
[@opencv_library; @harris2020numpy; @hunter2007matplotlib; @virtanen2020scipy; @paszke2019pytorch],
and the members of the Representing and Manipulating Deformable Linear
Objects project ([github.com/RMDLO](https://github.com/RMDLO)) for their
support. Holly Dinkel was supported by NASA Space Technology Graduate
Research Opportunity award 80NSSC21K1292, a P.E.O. Scholar Award, and
the Zonta International Amelia Earhart Fellowship. This work was also
supported by the Wallenberg AI, Autonomous Systems and Software Program
(WASP) funded by the Knut and Alice Wallenberg Foundation.

[^1]: ^1^Holly Dinkel, Bhumsitt Pramuanpornsatid, and Timothy Bretl are
    with the University of Illinois Urbana-Champaign, Urbana, IL, USA.
    e-mail: `{hdinkel2, bp17, tbretl}@illinois.edu`.

[^2]: ^2^Muhammad Zahid and Florian Pokorny are with the KTH Royal
    Institute of Technology, Stockholm, Sweden. e-mail:
    `{mzmi, fpokorny}@kth.se`

[^3]: ^3^Brian Coltin and Trey Smith are with the NASA Ames Research
    Center, Moffett Field, CA, USA. e-mail:
    `{brian.coltin, trey.smith}@nasa.gov`.
