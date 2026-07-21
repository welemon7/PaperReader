J. Vis. Commun. Image R.
journal homepage: www.elsevier.com/locate/jvci
 
Full Length Article
Illumination-aware softmask guided shadow removal$
Lianmeng Wei
, Sihui Luo
∗
Faculty of Electrical Engineering and Computer Science, Ningbo University, 818 Fenghua Road, Ningbo, 315000, Zhejiang, China


Shadow removal
Softmask
Image restoration
Deep learning
 
A B S T R A C T
While recent learning-based methods have boosted the performance of shadow removal, a major challenge 
persists: most leading approaches rely on manually annotated ground-truth masks as auxiliary priors. However, 
acquiring such manual annotations is costly, and model performance often degrades sharply without ground-
truth mask guidance. To tackle this problem, we propose a multi-scale illumination-aware softmask generation 
method. Specifically, we compute the luminance ratio between the shadow image and its shadow-free 
counterpart, followed by multi-scale filtering and fusion to produce a coherent softmask. This softmask is 
learned and predicted via a shallow network, which subsequently guides the restoration process. Compared 
to binary ground-truth masks, our approach yields softmask with improved coherence and more accurate 
preservation of edge gradients. Furthermore, we introduce a synergistic fusion of structural feature derived 
from self-extracted multi-scale representations using Gaussian kernels, which effectively retains structural 
information within shadowed regions. Extensive experiments confirm that our model achieves state-of-the-art 
performance. Our code is available at https://github.com/welemon7/Softmask.
1. Introduction
Shadow removal, which is dedicated to correcting illumination 
discrepancies and recovering structural texture in shadow regions, 
serves as a vital preprocessing step for a range of applications such as 
UAV remote sensing [1], autonomous driving systems [2], and visual 
servoing [3].
Recent years have seen significant progress in shadow removal 
driven by deep learning. Methods such as ShadowFormer [4], which ag-
gregates global context via channel attention to reduce computational 
complexity, and HomoFormer [5], which employs shuffle operations 
to achieve spatially uniform shadow distribution, exemplify this ad-
vancement. Despite their impressive performance, these mainstream 
methods uniformly depend on manually annotated ground-truth masks 
as auxiliary priors to guide the reconstruction. As shown in Fig. 1, when 
HomoFormer [5] is not supervised by the ground-truth (GM) prior, its 
performance degrades markedly.
Acquiring such manual annotations, however, is costly. Conse-
quently, recent research has shifted its focus toward shadow removal 
that does not require ground-truth masks. These include DeS3 [7] 
combines diffusion models with adaptive mechanisms, and OmniSR [8] 
integrates DINO features [9] with semantic cues [10] to enhance perfor-
mance. Nevertheless, such GM-free methods often struggle with com-
plex and varying illumination, while also incurring high computational 
costs.
$ This paper has been recommended for acceptance by Junsong Yuan.
∗Corresponding author.
E-mail address: luosihui@nbu.edu.cn (S. Luo).
To address the challenge of shadow removal without ground-truth 
masks, we propose an innovative softmask prediction method that 
eliminates reliance on manual annotations while providing physically 
more accurate and structure-preserving guidance for removal. Fol-
lowing physical illumination principles, our softmask is generated in 
linear color space by computing precise luminance ratios between 
shadow and shadow-free image pairs. Multi-scale illumination mod-
eling is further introduced to capture shadow characteristics across 
different scales. As shown in Fig. 2, our softmask generation method 
accurately localize shadow regions with more coherent boundaries 
than manual annotations, offering strong and reliable guidance for 
learning the softmask-prediction mapping. Additionally, multi-scale 
structural information-such as edges and textures extracted via Gaus-
sian filtering-is incorporated to offer essential structural cues for the 
removal network. Our shadow removal follows a two-stage pipeline: 
Network I is trained to predict the softmask from the shadowed in-
put; Network II then takes both the shadow image and the predicted 
softmask to reconstruct the shadow-free result.
In summary, our main contributions are two-fold: We propose a 
novel softmask generation method that produces high-quality, coherent 
softmasks from only shadow and shadow-free image pairs, offering 
more physically plausible estimates compared to conventional binary 
ground-truth masks. We design a structural extractor that captures rich 
https://doi.org/10.1016/j.jvcir.2026.104865
Received 3 February 2026; Received in revised form 26 April 2026; Accepted 7 June 2026
J. Vis. Commun. Image R. 119 (2026) 104865 
Available online 12 June 2026 
1047-3203/© 2026 Elsevier Inc. All rights are reserved, including those for text and data mining, AI training, and similar technologies. 


L. Wei and S. Luo
Fig. 1. HomoFormer [5] vs. HomoFormer (w/o GM) on ISTD+ dataset [6]. PSNR/SSIM: higher is better (positive metric), RMSE: lower is better (negative metrics 
shown with inverse normalization).
Fig. 2. The ground-truth masks (GM) and our softmask for three samples. 
Compared to the ground-truth masks, our softmask eliminates annotation 
inaccuracies and inconsistent boundaries.
multi-scale structural information, effectively enhancing the perfor-
mance of baseline models. Extensive experiments demonstrate that our 
approach significantly surpasses baseline and achieves state-of-the-art 
results on multiple benchmark datasets.
2. Related work
Shadows present significant challenges for image restoration due to 
their diverse shapes, structures, and coverage. Deep learning has driven 
remarkable progress in this area. Some approaches rely on manually 
annotated ground-truth shadow masks to guide image restoration, such 
as RASM [11], which utilizes region-aware attention to aggregate fea-
tures from both shadow and non-shadow areas, and FW-Former [12], 
features a region-aware attention mechanism for better restoration of 
fine details by focusing on information around shadow boundaries. Re-
cently, diffusion models have also been explored for shadow removal. 
ShadowDiffusion [13] leverages generative priors for natural shadow-
free results, and Diff-Shadow [14] further incorporates re-weighted 
cross-attention and global-guided sampling to maintain illumination 
consistency.
The rapid advancement of deep learning has led to a proliferation 
of data-driven shadow removal methods. The current paradigm is 
dominated by U-shaped networks, often integrated with Transformer 
modules to capture long-range dependencies, which have significantly 
boosted performance, robustness, and generalization. Representative 
works include ShadowFormer [4], which enhances context correla-
tion via shadow-interaction attention; OmniSR [8], which enriches 
semantic cues using DINO features and geometric priors; and Soft-
Shadow [15], leverages the Segment Anything Model and physical 
illumination constraints to generate softmasks for shadow removal. 
While these approaches push performance boundaries, the recent rise 
of diffusion models offers a new frontier. Methods like ShadowDiffu-
sion [13] and DeS3 [7] treat removal as an iterative denoising process, 
achieving impressive shadow-free image generation.
Subsequently, the high cost of ground-truth mask annotation and 
the severe performance drop without them have motivated GM-free 
methods, DC-ShadowNet [16] inspired this direction by using a domain 
classifier, though with limited success. Building on this, DeS3 [7] 
employed diffusion models but remains constrained by shadow lo-
cation and high parameters. Other works like ShadowRefiner [17] 
and OmniSR [8] incorporate frequency or other priors, yet still face 
performance bottlenecks.
Unlike these existing methods, we introduce a novel illumination-
aware softmask generation approach and enable the model to learn 
the mapping from the shadow image to this softmask, which is then 
integrated into the main shadow removal network to eliminate the need 
for costly manual annotations and improve the performance of methods 
that operate without ground-truth masks.
3. Methodology
3.1. Overview
Based solely on paired shadow images 𝐼𝑠 and shadow-free im-
ages 𝐼𝑔𝑡 without any manually annotated ground-truth masks, our 
framework (shown in Fig. 3) is built on a U-shaped Transformer ar-
chitecture [18] that comprises two networks: (I) SoftMask Predictor 
- a shallow network that takes the shadow image 𝐼𝑠 as input and 
outputs predicted softmask ̄𝑀𝑠, used to guide the subsequent shadow 
removal stage. The supervisory signal 𝑀𝑠 for softmask is derived from 
the multi-scale illumination-aware contrast between 𝐼𝑠 and 𝐼𝑔𝑡, as 
described in Section 3.2. (II) Shadow Removal - receives the shadow 
image 𝐼𝑠 along with the softmask ̄ 𝑀𝑠 predicted by Network I. It 
integrates a Multi-granularity Local Feature Enhancement (MLFE) mod-
ule at each encoder-decoder layer to enrich and complement texture 
representation, ultimately producing the predicted shadow-free image 
𝐼𝑑.
Journal of Visual Communication and Image Representation 119 (2026) 104865 
2 


L. Wei and S. Luo
Fig. 3. The top panel illustrates the universal framework for applying the softmask beyond the removal network. The bottom panel shows the softmask predictor, 
which is trained with softmask supervision. The right panel depicts the MLFE module, used to enhance structural representations within the removal network.
3.2. Softmask acquisition
3.2.1. Illumination map acquisition
To approximate shadow regions, we first analyze the illumina-
tion discrepancy between the shadow image 𝐼𝑠 and its shadow-free 
counterpart 𝐼𝑔𝑡. In physical imaging, light intensity relates linearly to 
scene brightness. However, standard RGB images are typically gamma-
encoded, introducing a nonlinear mapping between pixel values and 
actual radiance. Therefore, we convert both 𝐼𝑠 and 𝐼𝑔𝑡 into linear RGB 
space [19] to adhere to a physically based lighting model prior to illu-
mination comparison. Specifically, following the sRGB standard trans-
formation defined in IEC 61966-2-1 [20], each channel is linearized as:
𝐼𝐿=
⎧
⎪
⎨
⎪⎩
𝐼
12.92
𝐼≤0.04045
(
𝐼+0.055
1.055
)2.4
𝐼> 0.04045
(1)
where 𝐼∈[0, 1] denotes the gamma-encoded pixel value and 𝐼𝐿 is the 
corresponding linear radiance value.
We then compute the luminance maps 𝐿𝑠 and 𝐿𝑔𝑡 using the stan-
dard photometric weighting coefficients that reflect human spectral 
sensitivity [21]: 
𝐿= 0.299 × 𝑅+ 0.587 × 𝐺+ 0.114 × 𝐵
(2)
3.2.2. Multiscale Gaussian blur
Real-world shadows exhibit smooth illumination transitions and 
vary in size and edge softness. To model these characteristics at mul-
tiple scales, we apply Gaussian filtering [22] with different kernel 
sizes to both 𝐿𝑠 and 𝐿𝑔𝑡, obtaining smoothed luminance maps ̂𝐿𝑠 and ̂
𝐿𝑔𝑡, respectively. For a Gaussian kernel i, we compute the smoothed 
illumination as follows: ̂
𝐿𝑖= 𝑖(𝐿)
(3)
Larger kernels capture broader shadow regions with smoother
boundaries but may lose fine details, whereas smaller kernels retain 
local shadow structures but are more sensitive to noise. We then 
integrate the discrepancy illuminance maps ̂𝐿𝑖
𝑠 and ̂𝐿𝑔𝑡
𝑖 of multiple 
scales via summation followed by a sigmoid activation that serves to 
squash the aggregated responses into a bounded range and introduce a 
smooth transition, yielding the final softmask 𝑀𝑠: 
𝑀𝑠= Sigmoid
( 𝑛
∑
𝑖=1
(1 −̂
𝐿𝑖
𝑠̂
𝐿𝑖
𝑔𝑡
)
)
(4)
where 𝑛= 3 denotes the number of Gaussian kernels used, with sizes 
set to 15, 35, and 75. This multi-scale integration enhances spatial 
continuity in illumination transitions and improves robustness to noise.
The acquired 𝑀𝑠 for each training sample is then used to train the 
softmask predictor network.
3.3. Model architecture
3.3.1. Softmask predictor network
We adopt a Transformer-based U-Net architecture  [18], which 
consists of an encoder, a decoder, and a bottleneck layer, to predict 
the softmask. Given a shadow image 𝐼𝑠, an initial feature is first 
projected from it and then passed through the encoder-decoder hier-
archy. Each encoder, decoder, and bottleneck layer follows the same 
computational flow: the input feature 𝑥𝑖𝑛 is normalized via LayerNorm, 
partitioned into non-overlapping windows for window-based multi-
head self-attention 𝑊𝐴𝑡𝑡𝑛, and subsequently restored to its original 
layout. The resulting feature is normalized again and processed by a 
multi-layer perceptron (MLP) block, which consists of a linear layer, a 
deconvolutional layer, a channel-split operation followed by element-
wise multiplication for feature interaction, and a final linear layer. 
Another residual connection then yields the layer output 𝑥𝑜𝑢𝑡. This 
process can be summarized as: 
𝑥1 = 𝑥+ 𝑊𝐴𝑡𝑡𝑛(LayerNorm(𝑥𝑖𝑛))
𝑥𝑜𝑢𝑡= 𝑥1 + MLP(LayerNorm(𝑥1))
(5)
Downsampling is implemented by a 4 × 4 convolution with stride 
2, doubling the channel dimension while halving the spatial size. 
Upsample uses a 2 × 2 transposed convolution with stride 2, halving the 
channel dimension while doubling the spatial size. Skip connections are 
added between corresponding encoder and decoder layers to preserve 
fine details. After the final decoder, an output projection layer yields 
the predicted softmask.
3.3.2. Shadow removal network
The shadow removal network retains the same core Transformer-U-
Net structure as the softmask predictor, with the key modifications on 
the input/output setting and the integration of a structure-enhancing 
module.
The input is the channel-wise concatenation of the shadow image 
and the predicted softmask. After the final decoder layer, an output 
projection is applied, and its features are combined with the original 
shadow image via a residual connection to produce the shadow-free 
result.
Journal of Visual Communication and Image Representation 119 (2026) 104865 
3 


L. Wei and S. Luo
Table 1
Quantitative comparisons with SOTA methods on ISTD+ dataset [6]. The best and the second 
best results are boldfaced and underlined, respectively. GM: Rely on manually annotated 
ground-truth shadow masks. ''-'' indicates unavailable data.
Method
GM
Shadow region
All image region
PSNR↑
SSIM↑
RMSE↓
PSNR↑
SSIM↑
RMSE↓
Input Image
-
20.88
0.945
37.93
20.51
0.908
8.31
AutoExposure (CVPR'21) [23]
✓
36.21
0.975
6.50
29.45
0.840
4.20
BMNet (CVPR'22) [24]
✓
37.30
0.990
6.19
32.30
0.955
3.60
ShadowFormer (AAAI'23) [4]
✓
39.67
0.992
5.20
35.46
0.973
2.80
ShadowDiffusion (CVPR'23) [13]
✓
39.82
-
4.90
35.68
0.970
2.70
HomoFormer (CVPR'24) [5]
✓
39.49
0.993
4.73
35.35
0.975
2.64
RASM (MM'24) [11]
✓
40.73
0.993
4.41
36.16
0.976
2.53
FW-Former (JVCI'25) [12]
✓
40.11
0.993
4.71
35.84
0.975
2.58
DC-ShadowNet (ICCV'21) [16]
%
31.06
0.976
12.62
25.03
0.926
7.77
DeS3 (AAAI'24) [7]
%
36.52
0.988
6.62
31.38
0.958
3.94
OmniSR (AAAI'25) [8]
%
36.70
0.992
-
33.34
0.970
-
Ours
%
39.20
0.993
5.61
34.29
0.970
3.29
Table 2
Quantitative comparisons with SOTA methods on SRD 
dataset [25]. The best and the second best results are
boldfaced and underlined, respectively.
Method
GM
PSNR↑
RMSE↓
Input Image
-
18.19
14.05
AutoExposure(CVPR'21)  [23]
✓
26.99
5.95
BMNet (CVPR'22)  [24]
✓
31.69
4.46
ShadowFormer (AAAI'23)  [4]
✓
32.90
4.04
ShadowDiffusion (CVPR'23)  [13]
✓
34.73
3.63
HomoFormer (CVPR'24)  [5]
✓
35.37
3.33
RASM (MM'24)  [11]
✓
34.46
3.37
FW-Former (JVCI'25) [12]
✓
34.88
3.29
DC-ShadowNet (ICCV'21) [16]
%
31.53
4.65
Refusion (CVPRW'23)  [26]
%
31.60
-
DeS3 (AAAI'24)  [7]
%
34.11
3.72
OmniSR (AAAI'25)  [8]
%
32.87
4.21
Detail-Preserving (CVPR'25)  [27]
%
33.63
-
Ours
%
34.14
3.39
Table 3
Quantitative comparisons with SOTA methods on ISTD 
dataset [28]. The best and the second best results are
boldfaced and underlined, respectively.
Method
GM
PSNR↑
SSIM↑
Input Image
-
20.56
0.893
DHAN (AAAI'20)  [29]
✓
29.11
0.954
BMNet (CVPR'22)  [24]
✓
30.28
0.959
ShadowFormer (AAAI'23)  [4]
✓
32.21
0.968
ShadowDiffusion (CVPR'23)  [13]
✓
32.33
0.969
FSR-Net (MM'23)  [30]
✓
31.33
0.961
FW-Former (JVCI'25) [12]
✓
32.89
0.971
Diff-Shadow (AAAI'25) [14]
✓
33.69
0.976
DC-ShadowNet (ICCV'21) [16]
%
26.38
0.922
Refusion (CVPRW'23)  [26]
%
25.13
0.871
ShadowRefiner (CVPRW'24) [17]
%
28.75
0.916
OmniSR (AAAI'25)  [8]
%
30.45
0.964
Ours
%
30.46
0.964
Besides, we integrate a lightweight structural extractor with the 
backbone to enhance texture representation. The shadow image 𝐼𝑠 is 
smoothed by a set of depthwise convolutions with fixed multi-scale 
Gaussian kernels (kernel sizes {3, 5, 7}) to obtain low-frequency com-
ponents 𝐼𝑖
𝑙. High-frequency details 𝐼𝑖
ℎ are then derived as the residual 
between the original input 𝐼𝑠 and the smoothed versions 𝐼𝑖
𝑙: 
𝐼𝑖
𝑙= DepConv(𝐼𝑠)
𝐼𝑖
ℎ= 𝐼𝑠−𝐼𝑖
𝑙
(6)
Multi-scale high-frequency details are channel-wise concatenated, 
passed through a 1 × 1 convolution and a GELU activation [31], 
yielding the structural feature map 𝐹𝑠: 
𝐹𝑠= GELU(Conv(Concat(𝐼1
ℎ, 𝐼2
ℎ, 𝐼3
ℎ)))
(7)
The structural feature 𝐹𝑠 is integrated into each encoder-decoder 
layer through a fusion module. For a given base feature map 𝐹𝑏, 
spatial attention is first applied to enhance region-aware emphasis. 
Meanwhile, 𝐹𝑠 is resized via interpolation, projected by a 1 × 1 con-
volution, BatchNorm and activated with GELU. The two are combined 
element-wise to obtain the final feature 𝐹: 
𝐹= SAttn(𝐹𝑏) + GELU(𝐵𝑁(Conv(INT(𝐹𝑠))))
(8)
This design ensures that multi-scale structural cues are continuously 
injected throughout the network, improving texture recovery in shadow 
regions.
3.4. Loss function
We optimize the softmask predictor using the BCE loss [32]: 
(̄𝑀𝑠, 𝑀𝑠) = −[𝑀𝑠⋅log(̄𝑀𝑠) + (1 −𝑀𝑠) ⋅log(1 −̄ 𝑀𝑠)]
(9)
Besides, Charbonnier loss [33] is utilized for learning the shadow 
removal model: 
(𝐼𝑑, 𝐼𝑔𝑡) =
√
‖𝐼𝑑−𝐼𝑔𝑡‖2 + 𝜀
(10)
where ̄𝑀𝑠, 𝑀𝑠 represent the predicted softmask and the prior generated 
softmask. 𝐼𝑑, 𝐼𝑔𝑡 denote the prediction and ground-truth respectively, 
with 𝜀= 10−6 ensuring numerical stability.
4. Results
4.1. Experimental setup
4.1.1. Datasets
We evaluate our model on five challenging public datasets. For 
evaluation metrics, the datasets used are:
• SRD Dataset [25]: This dataset contains 2680 image pairs for 
training and 408 for testing, each consisting of a shadow image 
and its shadow-free counterpart.
• ISTD Dataset [28]: It comprises 1870 triplets (shadow image, 
shadow-free image, and ground-truth mask), divided into 1330 
training and 540 testing samples.
• Adjusted ISTD (ISTD+) Dataset [6]: An enhanced version of ISTD 
Dataset [28] that addresses illumination inconsistencies in the 
original data.
Journal of Visual Communication and Image Representation 119 (2026) 104865 
4 


L. Wei and S. Luo
Fig. 4. Visual comparisons of our approach with open-sourced SOTA methods.
• WRSD+ Dataset [34,35]: It Consists of 1000 training pairs and 
100 test pairs, forming a high-resolution dataset without ground-
truth masks.
For robustness evaluation, we adopt the LRSS dataset [36] and the 
USR dataset [37]. The LRSS dataset [36] comprises 46 paired shadow 
and shadow-free images, without providing ground-truth shadow
masks. In contrast, the USR dataset [37] is an unpaired shadow removal 
benchmark, containing 2445 shadow images and 1770 shadow-free 
images.
4.1.2. Evaluation metrics
To ensure fairness, we evaluated Peak Signal-to-Noise Ratio (PSNR) 
and Structural Similarity Index Measure (SSIM) on the RGB channels 
and Root Mean Square Error (RMSE) on the LAB color space to assess 
the quality. To quantify the model efficiency, we report both the num-
ber of parameters and the FLOPs. Following previous method [4,5,11], 
all images are resized to 256 × 256 when evaluating. In addition, 
given the high resolution of WRSD+ dataset [34,35], we resized all 
images to 480 × 360 (1/4 of original) for training and evaluation to 
facilitate processing and ensure a fair comparison with state-of-the-art
methods.
4.1.3. Implement details
We implement our models in PyTorch and train them for 100 and 
600 epochs, respectively, using the AdamW optimizer [38] with an 
initial learning rate of 2 × 10−4, weight decay 1 × 10−4, and batch size 
7.
Journal of Visual Communication and Image Representation 119 (2026) 104865 
5 


L. Wei and S. Luo
Fig. 5. Visual comparisons of our approach with open-sourced SOTA methods.
Table 4
Quantitative comparisons with SOTA methods on WRSD+ dataset [34,35]. The 
best and the second best results are boldfaced and underlined, respectively.
 
Method
PSNR↑
SSIM↑
RMSE↓ 
 
480 × 360
Input Image
18.87
0.825
14.76 
 
UFormer (CVPR'22) [39]
25.68
0.919
6.93 
 
ShadowFormer (AAAI'23) [4]
25.64
0.918
7.04 
 
HomoFormer (CVPR'24) [5]
25.87
0.914
7.03 
 
RASM (MM'24) [11]
25.68
0.920
6.88 
 
Ours
26.29
0.923
6.63 
4.2. Comparison with the state-of-the-art
We presents quantitative comparisons of our model with state-
of-the-art (SOTA) methods (DHAN [29], AutoExposure [23], DC-
ShadowNet [16], BMNet [24], UFormer [39], ShadowFormer [4], Shad-
owDiffusion [13], ShadowRefiner [17], FSR-Net [30], HomoFormer [5], 
RASM [11], Refusion [26], DeS3 [7], OmniSR [8], Diff-Shadow [14], 
FW-Former [12] and Detail-Preserving [27]) on SRD [25], ISTD+ [6], 
ISTD [28] and WRSD+ [34,35] datasets with three metrics. Results 
and complexity are summarized in Tables 1-5. For fair comparison, we 
primarily use results reported by the original authors. Where results 
Journal of Visual Communication and Image Representation 119 (2026) 104865 
6 


L. Wei and S. Luo
Table 5
Efficiency comparisons with state-of-the-art (SOTA) methods. The 
best and the second best results are boldfaced and underlined, 
respectively.
Method
GM
Params (M)↓
GFLOPs↓
DHAN (AAAI'20)  [29]
✓
16.4 
126.0
AutoExposure (CVPR'21)  [23]
✓
19.7 
53.0
BMNet (CVPR'22)  [24]
✓
0.4 
11.6
ShadowFormer (AAAI'23)  [4]
✓
11.4 
64.6
ShadowDiffusion (CVPR'23)  [13]
✓
55.5 
728.2
HomoFormer (CVPR'24)  [5]
✓
17.8 
35.6
RASM (MM'24)  [11]
✓
5.2
26.3
FW-Former (JVCI'25)  [12]
✓
6.3 
27.5
Refusion (CVPRW'23)  [26]
%
131.4 
63.4
DeS3 (AAAI'24)  [7]
%
113.7 
24.9
OmniSR (AAAI'25)  [8]
%
24.6
78.3
Ours
%
18.1 
36.1
are unavailable, we evaluate provided images or checkpoints using 
consistent metrics.
Our method achieves state-of-the-art for all metrics on the four 
datasets. Especially on ISTD+ [6], we achieve superior PSNR in shadow 
and all image regions, leading the second-best by 2.50 dB and 0.95 dB, 
respectively, with RMSE reductions of 1.01 and 0.65. And on WRSD+
[34,35], that lack ground-truth masks, those models heavily reliant on 
ground-truth masks [4,5] face performance bottlenecks. In contrast, 
our method can generate predicted softmasks entirely without manual 
annotation, achieving the best performance on WRSD+ dataset [34,35]. 
Meanwhile, among GM-free methods, our approach achieves the opti-
mal performance in terms of parameter count and the second-best in 
FLOPs. Compared to diffusion-based models such as Refusion [26] and 
DeS3 [7], which require massive parameters, and OmniSR [8], which 
relies on multiple additional priors, our method strikes an optimal 
balance between computational efficiency and performance. As shown 
in Figs. 4 and 5, visual comparisons with publicly available methods 
further confirm the enhanced shadow-removal capability of our model, 
which benefits from the proposed softmask and structural extractor.
4.3. Ablation study
To validate the contribution of our softmask ̄𝑀𝑠 and MLFE, we 
conduct comprehensive ablation studies on SRD dataset [25]. As sum-
marized in Table 6, we evaluate our baseline model as well as three 
distinct configurations. Experimental results clearly demonstrate that 
our softmask and structural module enhance the performance of the 
baseline model.
4.4. Robustness analysis and limitation
To demonstrate the model's generalization capability, we evaluate 
our SRD [25] pretrained model on the unseen LRSS [36] and USR 
dataset [37]. The results are visualized in Fig. 6, where our method suc-
cessfully predicts softmasks that are well aligned with shadow regions 
and achieves clean shadow removal. 
However, we have to note that under cross-domain generalization 
with significant color discrepancies, the results may exhibit minor color 
bias and slight edge discontinuities. Together with the reliance on 
paired training data, these constitute two limitations of our method.
To address these issues, future work will explore learning from 
unpaired datasets and develop more effective mechanisms to mitigate 
color inconsistency and edge artifacts, with the goal of achieving 
superior generalization and perceptual quality.
Fig. 6. Visualization on unseen LRSS [36] and USR dataset [37] with SRD [25] 
pretrained models. The first three rows show successful cases, while the bottom 
row demonstrates a failure case.
5. Conclusion
In this paper, we presented an illumination-aware softmask guided 
shadow removal network that eliminates the dependency on ground-
truth shadow masks. Our key contribution lies in softmask acquisition, 
which produces high-quality continuous shadow estimates through 
multi-scale proportional regression in linear color space. Furthermore, 
we introduced a MLFE Module that enhances the structural coher-
ence of encoder-decoder layers for better reconstruction performance. 
Combined with our transformer architecture, the proposed method 
achieves state-of-the-art performance on multiple benchmarks without 
ground-truth masks supervision.

Lianmeng Wei: Writing - original draft, Visualization, Validation, 
Software, Methodology, Investigation, Formal analysis, Data curation, 
Conceptualization. Sihui Luo: Writing - review & editing, Supervision, 
Software, Resources, Project administration, Methodology, Funding ac-
quisition, Data curation, Conceptualization.
Declaration of competing interest
The authors declare that they have no known competing finan-
cial interests or personal relationships that could have appeared to 
influence the work reported in this paper.
Acknowledgments
This work was supported by the Natural Science Foundation Projects 
of Zhejiang Province under Grant LQ22F020020.
Data availability
All the datasets used in this paper are publicly available.
Journal of Visual Communication and Image Representation 119 (2026) 104865 
7 


L. Wei and S. Luo
Table 6
Ablation study of different network modules on SRD [25] dataset. PSNR, SSIM and RMSE metrics of the Shadow, Non-shadow, 
and all regions are reported.
 ID
Baseline
MLFĒ
𝑀𝑠
Shadow
Non-shadow
All
 
PSNR↑
SSIM↑
 RMSE↓
 PSNR↑
SSIM↑
RMSE↓
PSNR↑
SSIM↑
RMSE↓ 
 i
✓
37.56
0.981
6.18
35.53
0.976
3.37
32.63
0.948
3.92 
 ii
✓
✓
38.09
0.982
5.91
36.20
0.978
3.23
33.30
0.951
3.75 
 iii
✓
✓
38.87
0.985
5.45
36.94
0.980
2.95
34.01
0.956
3.43 
 iv
✓
✓
✓
39.00
0.985
5.36
37.08
0.980
2.91
34.14
0.957
3.39 
References
[1] L.P. Osco, J.M. Junior, A.P.M. Ramos, L.A. de Castro Jorge, S.N. Fatholahi, J. 
de Andrade Silva, E.T. Matsubara, H. Pistori, W.N. Gonçalves, J. Li, A review 
on deep learning in UAV remote sensing, Int. J. Appl. Earth Obs. Geoinf. 102 
(2021) 102456.
[2] J. Levinson, J. Askeland, J. Becker, J. Dolson, D. Held, S. Kammel, J.Z. Kolter, 
D. Langer, O. Pink, V. Pratt, et al., Towards fully autonomous driving: Systems 
and algorithms, in: 2011 IEEE Intelligent Vehicles Symposium, IV, IEEE, 2011, 
pp. 163-168.
[3] S.L. Li, A. Zhang, B. Chen, H. Matusik, C. Liu, D. Rus, V. Sitzmann, Controlling 
diverse robots by inferring Jacobian fields with deep networks, Nature (2025) 
1-7.
[4] L. Guo, S. Huang, D. Liu, H. Cheng, B. Wen, ShadowFormer: Global context 
helps shadow removal, in: Proceedings of the AAAI Conference on Artificial 
Intelligence, Vol. 37, 2023, pp. 710-718.
[5] J. Xiao, X. Fu, Y. Zhu, D. Li, J. Huang, K. Zhu, Z.-J. Zha, Homoformer: 
Homogenized transformer for image shadow removal, in: Proceedings of the 
IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, 2024, 
pp. 25617-25626.
[6] H. Le, D. Samaras, Shadow removal via shadow image decomposition, in: 
Proceedings of the Proceedings of the IEEE/CVF International Conference on 
Computer Vision (ICCV)Uter Vision, 2019, pp. 8578-8587.
[7] Y. Jin, W. Ye, W. Yang, Y. Yuan, R.T. Tan, Des3: Adaptive attention-driven 
self and soft shadow removal using vit similarity, in: Proceedings of the AAAI 
Conference on Artificial Intelligence, Vol. 38, 2024, pp. 2634-2642.
[8] J. Xu, Z. Li, Y. Zheng, C. Huang, R. Gu, W. Xu, G. Xu, Omnisr: Shadow removal 
under direct and indirect lighting, in: Proceedings of the AAAI Conference on 
Artificial Intelligence, Vol. 39, 2025, pp. 8887-8895.
[9] H. Zhang, F. Li, S. Liu, L. Zhang, H. Su, J. Zhu, L.M. Ni, H.-Y. Shum, Dino: Detr 
with improved denoising anchor boxes for end-to-end object detection, 2022, 
arXiv preprint arXiv:2203.03605.
[10] L. Yang, B. Kang, Z. Huang, Z. Zhao, X. Xu, J. Feng, H. Zhao, Depth anything 
v2, Adv. Neural Inf. Process. Syst. 37 (2024) 21875-21911.
[11] H. Liu, M. Li, X. Guo, Regional attention for shadow removal, in: Proceedings of 
the 32nd ACM International Conference on Multimedia, 2024, pp. 5949-5957.
[12] Z. Xiujin, C. Chee-Onn, C.J. Huang, Regional decay attention for image shadow 
removal, J. Vis. Commun. Image Represent. (2025) 104694.
[13] L. Guo, C. Wang, W. Yang, S. Huang, Y. Wang, H. Pfister, B. Wen, Shadowd-
iffusion: When degradation prior meets diffusion model for shadow removal, 
in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern 
Recognition, CVPR, 2023, pp. 14049-14058.
[14] J. Luo, R. Li, C. Jiang, X. Zhang, M. Han, T. Jiang, H. Fan, S. Liu, Diff-shadow: 
Global-guided diffusion model for shadow removal, in: Proceedings of the AAAI 
Conference on Artificial Intelligence, Vol. 39, 2025, pp. 5856-5864.
[15] X. Wang, L. Guo, X. Wang, S. Huang, B. Wen, SoftShadow: Leveraging soft 
masks for Penumbra-aware shadow removal, in: Proceedings of the IEEE/CVF 
Conference on Computer Vision and Pattern Recognition, CVPR, 2025, pp. 
23217-23226.
[16] Y. Jin, A. Sharma, R.T. Tan, Dc-shadownet: Single-image hard and soft shadow 
removal using unsupervised domain-classifier guided network, in: Proceedings of 
the Proceedings of the IEEE/CVF International Conference on Computer Vision 
(ICCV)Uter Vision, 2021, pp. 5027-5036.
[17] W. Dong, H. Zhou, Y. Tian, J. Sun, X. Liu, G. Zhai, J. Chen, ShadowRefiner: 
Towards mask-free shadow removal via fast fourier transformer, in: Proceedings 
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 
Workshop, CVPRW, 2024, pp. 6208-6217.
[18] H. Cao, Y. Wang, J. Chen, D. Jiang, X. Zhang, Q. Tian, M. Wang, Swin-unet: Unet-
like pure transformer for medical image segmentation, in: European Conference 
on Computer Vision, ECCV, Springer, 2022, pp. 205-218.
[19] S. Süsstrunk, R. Buckley, S. Swen, Standard RGB color spaces, in: Color and 
Imaging Conference, Vol. 7, Society of Imaging Science and Technology, 1999, 
pp. 127-134.
[20] I.E. Commission, et al., IEC 61966-2-1: Multimedia systems and equipment-
Colour measurement and Management-Part 2-1: Colour Management-Default 
RGB Colour Space-sRGB, International Electrotechnical Commission, Geneva, 
Switzerland, 1999.
[21] R. BT, et al., Studio encoding parameters of digital television for standard 
4: 3 and wide-screen 16: 9 aspect ratios, in: International radio consultative 
committee international telecommunication union, CCIR Rep, Switzerland, 2011.
[22] K. Ito, K. Xiong, Gaussian filters for nonlinear filtering problems, IEEE Trans. 
Autom. Control 45 (5) (2002) 910-927.
[23] L. Fu, C. Zhou, Q. Guo, F. Juefei-Xu, H. Yu, W. Feng, Y. Liu, S. Wang, 
Auto-exposure fusion for single-image shadow removal, in: Proceedings of the 
IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR, 2021, 
pp. 10571-10580.
[24] Y. Zhu, J. Huang, X. Fu, F. Zhao, Q. Sun, Z.-J. Zha, Bijective mapping network 
for shadow removal, in: Proceedings of the IEEE/CVF Conference on Computer 
Vision and Pattern Recognition, CVPR, 2022, pp. 5627-5636.
[25] L. Qu, J. Tian, S. He, Y. Tang, R.W. Lau, Deshadownet: A multi-context 
embedding deep network for shadow removal, in: Proceedings of the IEEE 
Conference on Computer Vision and Pattern Recognition, CVPR, 2017, pp. 
4067-4075.
[26] Z. Luo, F.K. Gustafsson, Z. Zhao, J. Sjölund, T.B. Schön, Refusion: Enabling large-
size realistic image restoration with latent-space diffusion models, in: Proceedings 
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 
Workshop, CVPRW, 2023, pp. 1680-1691.
[27] J. Xu, Y. Zheng, Z. Li, C. Wang, R. Gu, W. Xu, G. Xu, Detail-preserving latent 
diffusion for stable shadow removal, in: Proceedings of the IEEE/CVF Conference 
on Computer Vision and Pattern Recognition, CVPR, 2025, pp. 7592-7602.
[28] J. Wang, X. Li, J. Yang, Stacked conditional generative adversarial networks for 
jointly learning shadow detection and shadow removal, in: Proceedings of the 
IEEE Conference on Computer Vision and Pattern Recognition, CVPR, 2018, pp. 
1788-1797.
[29] X. Cun, C.-M. Pun, C. Shi, Towards ghost-free shadow removal via dual 
hierarchical aggregation network and shadow matting gan, in: Proceedings of 
the AAAI Conference on Artificial Intelligence, Vol. 34, 2020, pp. 10680-10687.
[30] J. Yu, P. He, Z. Peng, Fsr-net: Deep fourier network for shadow removal, in: 
Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 
2335-2343.
[31] M. Lee, Gelu activation function in deep learning: a comprehensive mathematical 
analysis and performance, 2023, arXiv preprint arXiv:2305.12073.
[32] U. Ruby, V. Yendapalli, et al., Binary cross entropy with deep learning technique 
for image classification, Int. J. Adv. Trends Comput. Sci. Eng. 9 (10) (2020).
[33] P. Charbonnier, L. Blanc-Feraud, G. Aubert, M. Barlaud, Two deterministic half-
quadratic regularization algorithms for computed imaging, in: Proceedings of 1st 
International Conference on Image Processing, Vol. 2, IEEE, 1994, pp. 168-172.
[34] F.-A. Vasluianu, T. Seizinger, R. Timofte, Wsrd: A novel benchmark for high 
resolution image shadow removal, in: Proceedings of the IEEE/CVF Conference 
on Computer Vision and Pattern Recognition Workshop, CVPRW, 2023, pp. 
1826-1835.
[35] F.-A. Vasluianu, T. Seizinger, Z. Zhou, Z. Wu, C. Chen, R. Timofte, W. Dong, 
H. Zhou, Y. Tian, J. Chen, et al., NTIRE 2024 image shadow removal challenge 
report, in: Proceedings of the IEEE/CVF Conference on Computer Vision and 
Pattern Recognition Workshop, CVPRW, 2024, pp. 6547-6570.
[36] M. Gryka, M. Terry, G.J. Brostow, Learning to remove soft shadows, ACM Trans. 
Graph. 34 (5) (2015) 1-15.
[37] X. Hu, Y. Jiang, C.-W. Fu, P.-A. Heng, Mask-shadowgan: Learning to remove 
shadows from unpaired data, in: Proceedings of the Proceedings of the IEEE/CVF 
International Conference on Computer Vision (ICCV)Uter Vision, 2019, pp. 
2472-2481.
[38] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, Int. Conf. Learn. 
Represent. (ICLR) (2019).
[39] Z. Wang, X. Cun, J. Bao, W. Zhou, J. Liu, H. Li, Uformer: A general u-shaped 
transformer for image restoration, in: Proceedings of the IEEE/CVF Conference 
on Computer Vision and Pattern Recognition, CVPR, 2022, pp. 17683-17693.
Journal of Visual Communication and Image Representation 119 (2026) 104865 
8