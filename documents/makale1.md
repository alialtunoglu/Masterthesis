Machine Learning with Applications 18 (2024) 100605
Contents lists available at ScienceDirect
Machine Learning with Applications
journal homepage: www.elsevier.com/locate/mlwa
A survey on knowledge distillation: Recent advancements
Amir Moslemia,b,*, Anna Briskinaa, Zubeka Danga, Jason Lia
aSchool of Software Design & Data Science, Seneca Polytechnic, Toronto, Ontario, Canada
bDepartment of Physics, Toronto Metropolitan University, Toronto, Ontario, Canada
A R T I C L E I N F O A B S T R A C T
Keywords: Deep learning has achieved notable success across academia, medicine, and industry. Its ability to identify
Deep learning complex patterns in large-scale data and to manage millions of parameters has made it highly advantageous.
Knowledge distillation However, deploying deep learning models presents a significant challenge due to their high computational de-
Model compression
mands. Knowledge distillation (KD) has emerged as a key technique for model compression and efficient
Self-distillation
knowledge transfer, enabling the deployment of deep learning models on resource-limited devices without
Adversarial distillation
compromising performance. This survey examines recent advancements in KD, highlighting key innovations in
architectures, training paradigms, and application domains. We categorize contemporary KD methods into
traditional approaches, such as response-based, feature-based, and relation-based knowledge distillation, and
novel advanced paradigms, including self-distillation, cross-modal distillation, and adversarial distillation stra-
tegies. Additionally, we discuss emerging challenges, particularly in the context of distillation under limited data
scenarios, privacy-preserving KD, and the interplay with other model compression techniques like quantization.
Our survey also explores applications across computer vision, natural language processing, and multimodal tasks,
where KD has driven performance improvements and enhanced model compression. This review aims to provide
researchers and practitioners with a comprehensive understanding of the state-of-the-art in knowledge distilla-
tion, bridging foundational concepts with the latest methodologies and practical implications.
1. Introduction extensively studied, developed, and refined, finding wide-ranging ap-
plications across various domains of artificial intelligence, particularly
Knowledge distillation (KD), a technique for model compression and in scenarios where model efficiency is crucial.
knowledge transfer, has gained significant attention in the field of ma- Knowledge distillation offers solutions to a range of industry chal-
chine learning since its formal introduction by Hinton et al. (2015). This lenges. Notable applications include reducing the memory requirements
approach builds upon the earlier work of Caruana et al. (2006), who of models, maintaining data privacy during training, and decreasing
demonstrated that the collective knowledge of an ensemble of models energy consumption. For example, reducing the memory requirements
could be condensed into a single, smaller model. This groundbreaking of a model without compromising performance is particularly beneficial
idea laid the foundation for addressing the challenge of deploying deep for embedded systems in self-driving cars (Li et al., 2022c). In situations
models on resource-constrained devices such as mobile phones and where privacy is a concern, data-free knowledge distillation addresses
embedded systems, which often have limited memory and processing these issues by generating data, thereby eliminating the need to collect
power. Hinton et al. (2015) expanded on this concept, showing that and store sensitive information (Wang et al., 2024). Additionally,
information from a large, complex model (the teacher) could be effec- knowledge distillation can help lower the energy consumption of a
tively transferred to a smaller, more compact model (the student) model by transferring the performance of a larger model to a smaller
through a process they termed "knowledge distillation" (shown as one, with minimal performance loss (Gowda et al., 2024).
Fig. 1). This technique has since become a cornerstone in the ongoing Knowledge distillation can be classified into several schemes and
effort to create more efficient and deployable machine learning models. algorithms, each focusing on different strategies for transferring
In the years following its introduction, knowledge distillation has been knowledge from the teacher model to the student model. Response-
* Corresponding author.
E-mail addresses: amir.moslemi@ryerson.ca(A. Moslemi), anna.briskina@senecapolytechnic.ca(A. Briskina), zubeka-dane.dang@senecapolytechnic.ca(Z. Dang),
jason.li@senecapolytechnic.ca(J. Li).
https://doi.org/10.1016/j.mlwa.2024.100605
Received 24 October 2024; Received in revised form 8 November 2024; Accepted 9 November 2024
Available online 10 November 2024
2666-8270/© 2024 The Authors. Published by Elsevier Ltd. This is an open access article under the CC BY-NC-ND license ( http://creativecommons.org/licenses/by-
nc-nd/4.0/) .

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
approaches and their applications across various domains, illustrating
the diverse methodologies, network architectures, and datasets used in
offline, online, and self-distillation approaches.
Various knowledge distillation techniques employ specialized
mechanisms and architectural components to effectively capture and
transfer complex relationships and contextual information from teacher
models to student models. For instance, cross-modal distillation in-
volves transferring knowledge between models trained on different
modalities (e.g., images to text), enabling the student to understand
concepts from diverse data types. In graph neural networks (GNNs), KD
Fig. 1. Overview of the Teacher–Student Knowledge Distillation Framework. is employed to transfer the structural knowledge of graph data. In
graph-based distillation, the student model learns not only from the
based distillation is the traditional form of KD, where knowledge is node features but also from the intricate relationships between nodes,
transferred by training the student model to mimic the output logits of such as edges and connectivity patterns (Dong et al., 2021; Yang et al.,
the teacher model, often using soft targets (Song et al., 2023). This 2022). By capturing the pairwise interactions and the overall graph to-
method has been widely utilized due to its simplicity and effectiveness in pology, the student model can effectively mimic the teacher model’s
improving model performance across various tasks. In contrast, fea- performance in tasks like node classification, link prediction, and graph
ture-based distillation does not focus solely on the outputs but rather classification. This approach is essential for graph-structured data,
transfers intermediate features from the teacher to the student. This where relational information often holds the key to understanding
method aims to align the feature representations and activation maps of complex structures and patterns. Building on feature-based distillation,
the two models, thereby capturing richer knowledge from the teacher attention-based distillation guides the student model to focus on
network (Ji et al., 2021a). Feature-based distillation has gained popu- specific regions or aspects of the input data deemed important by the
larity for its ability to convey more nuanced information from the teacher. This method leverages attention mechanisms to refine the
teacher to the student. Relation-based distillation transfers relational learning process, making it particularly useful for tasks that require
knowledge by teaching the student to understand interdependencies or interpretability and attention to detail (Ji et al., 2021a).
similarities between instances learned by the teacher. It often finds its Certain knowledge distillation methods are designed to function
application in tasks where understanding the relationships between data independently of the original training data, enabling distillation in
points is crucial, making it particularly suitable for graph-based, visual, scenarios where data access is limited or restricted. For example, data-
and language models. Table 1provides an overview of the various types free distillation conducts distillation without access to the original
of knowledge distillation and their applications across different do- training data. This distillation technique is especially useful when access
mains, showcasing the distinct methods, knowledge types, and network to the original training data is restricted due to privacy or security
architectures employed in response-based, feature-based, and concerns (Zhu et al., 2021; Chawla et al., 2021). It utilizes synthetic data
relation-based distillation techniques. or generative models to create pseudo-data, allowing the student to
Offline distillation involves training the teacher model first and learn from the teacher without relying on the original dataset. Alter-
then transferring its knowledge to the student in a separate process. It is natively, adversarial distillation introduces an adversarial training
commonly used when a strong pre-trained teacher is available, allowing framework into the distillation process, where the student model is
efficient model compression (Srinivasagan et al., 2023; Yin et al., 2022). trained to not only mimic the teacher but also to compete against a
Online distillation, in contrast, simultaneously trains both teacher and discriminator model that differentiates between the teacher and student
student models, facilitating real-time knowledge transfer as both models outputs (Goodfellow et al., 2014). This adversarial learning process
evolve together. This approach is suitable for applications requiring improves the robustness and generalization of the student model,
continual learning (Li et al., 2022a; Chen et al., 2020). Another making it suitable for tasks where security and robustness are critical.
approach is self-distillation, which differs from traditional KD by Recently, Generative Adversarial Networks (GANs) (Goodfellow et al.,
involving a single model that acts as both the teacher and the student. In 2014) have become increasingly popular in deep learning and are now
this scheme, the model refines its own knowledge through recursive widely applied in adversarial distillation (Ye & Bors, 2021; Zhai et al.,
learning processes, enhancing performance without needing a larger 2021). In this context, GANs generate challenging examples that help
teacher network (Zhang et al., 2019; Zhang et al., 2021). This technique the student model learn to generalize better, effectively transferring
simplifies the training process while still achieving performance gains. more nuanced knowledge from the teacher. By leveraging the adversa-
Table 2 shows an overview of the different knowledge distillation rial framework, the student model can acquire knowledge that enhances
Table 1
Overview of knowledge distillation types and their applications.
Authors Knowledge schemes Methods Knowledge types Teacher networks Student networks Dataset
Song et al. Response-Based Distillation; EffDstl Soft Targets ResNet34 ResNet18 ImageNet
(2023)
Ahmad et al. Response-Based Distillation; Multi- MTCM-KD Soft Labels CDSFL T1CE modality BraTS-2021
(2024) teacher cross-modal distillation;
Kim et al. (2023) Response-Based Distillation; RCKD Soft Labels MoNuSeg2018 CSAT TCGA
Ji et al. (2021a) Feature-Based Distillation; AFD Feature Links; ResNet ResNet CIFAR-100;
Attention-Based Distillation Activation Map tinyImageNet; ImageNet
Ji et al. (2021b) Feature-Based Distillation; Self- FRSKD Feature Maps; Soft ResNet ResNet CIFAR-100;
Distillation Labels tinyImageNet; ImageNet
Sepahvand et al. Feature-Based Distillation Distillation Feature Maps ResNet-50; VGG- LeNet-5; AlexNet CIFAR-10; Pokemon
(2022) Module 19
Dong et al. Relation-Based Distillation ERDIL Relation Graph ResNet ResNet CIFAR100;
(2021) miniImageNet; CUB200
Yang et al. Relation-Based Distillation CIRKD Pixel-to-Pixel; DeepLabV3; DeepLabV3; PSPNet; Cityscapes; CamVid;
(2022) Pixel-to-Region ResNet-101 ResNet18; MobileNetV2 Pascal VOC
2

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
Table 2
Overview of knowledge distillation approaches and their applications.
Authors Knowledge schemes Methods Knowledge types Teacher networks Student networks Dataset
Srinivasagan et al. Offline Distillation ITS Soft Labels ViTSTR ViTSTR MJ; ST
(2023)
Yin et al. (2022) Offline Distillation CVRKD- High-Quality and Low- FR-teacher NAR-student Kaddid10K; LIVE; TID2013; KonIQ-10K
IQA Quality Distribution
Schmid et al. Offline Distillation PaSST Soft Labels Transformer MobileNetV3-Large AudioSet
(2022)
Li et al. (2022a) Online Distillation OKDHP Pixel-wise; Feature HG; FAU HG; FAU MPII; COCO
Maps
Chen et al. (2020) Online Distillation OKDDip Feature Maps; Soft ResNet; VGG; ResNet; VGG; CIFAR; ImageNet
Labels DensNet; WRN DensNet; WRN
Li et al. (2020) Online Distillation FFM Feature Fusion Module ResNet; Xception; ResNet; Xception; CIFAR; CINIC-10
ShuffleNet ShuffleNet
Kim et al. (2021) Self-Distillation PS-KD Soft Labels ResNet; DenseNet; ResNet; DenseNet; CIFAR-100
PyramidNet PyramidNet
Ge et al. (2021) Self-Distillation BAKE Soft Labels RestNet; MobileNet; RestNet; MobileNet; ImageNet-1K; CIFAR-100;
EfficientNet EfficientNet TinyImageNet; CUB-200-2011; Stanford
Dogs; MIT67
Yue et al. (2022) Self-Distillation SSAD Soft Labels SSJD; PCN; SSL SSJD; PCN; SSL Indian Pines; University of Pavia;
Houston
Ji et al. (2021b) Feature-Based FRSKD Feature Maps; Soft ResNet ResNet CIFAR-100; tinyImageNet; ImageNet
Distillation; Self- Labels
Distillation
its ability to handle complex, real-world scenarios, particularly in do- 2. Types of knowledge distillation
mains like image synthesis, anomaly detection, and robust classification
(Higuchi et al., 2022; Lee et al., 2022; He et al., 2022). This section explores the primary types of knowledge distillation,
Several distillation techniques focus on optimizing the KD process to including response-based, feature-based, and relation-based distillation.
enhance efficiency and performance. For instance, quantized distilla- These methods vary in the way they extract and transfer knowledge
tion aims to reduce the memory and computational footprint of the from the teacher model, offering diverse strategies to improve student
student model by employing techniques like quantization during the model performance while maintaining efficiency.
distillation process (Kim et al., 2019). This involves compressing the
model’s parameters into lower precision representations without
2.1. Response-based knowledge distillation
significantly sacrificing performance. Additionally, NAS-based distil-
lation integrates Neural Architecture Search (NAS) techniques with KD
Response-based knowledge distillation is a technique where a stu-
to automatically find the most optimized student network architecture
dent model learns to mimic a teacher model’s soft output probabilities,
(Lee et al, 2023). This approach strikes a balance between performance
or "soft targets." These soft targets provide a nuanced view of data,
and computational efficiency and is increasingly popular for designing
encoding crucial class relationships for generalization. This approach
student networks tailored to specific tasks and constraints.
helps the student model understand the teacher’s decision-making
Additionally, ensemble-based knowledge distillation approaches
process, improving performance and avoiding overfitting (Gou et al.,
combine insights from multiple teacher models, aiming to improve the
2021).
accuracy and robustness of the resulting student model. For example,
The loss function of response-based KD focuses on matching the
lifelong distillation focuses on scenarios where models need to
output of the teacher and student models. This is done by minimizing the
continuously learn from new data while retaining previously acquired
Kullback-Leibler (KL) divergence between the softened output proba-
knowledge, employing distillation to mitigate catastrophic forgetting
bilities of the teacher and the student networks (Hilton et al., 2015). The
(Ye & Bors, 2021; Zhai & Mori, 2021). This approach is particularly
softened output of teacher and student is formulated as:
relevant in dynamic environments where models need to adapt over ( )
time, such as real-time monitoring systems and evolving data streams. ( ) exp zt
L
o
a
ft
s
e
t
n
ly
f
,
r o
m
m
u l
d
t
i
i
f
-
f
t
e
e
r
a
e
c
n
h
t
e
m
r
o
d
d
i
a
s
l
t
i
i
t
l
i
l
e
a
s
t
,
i
t
o
o
n
p r
e
o
m
v
p
id
lo
e
y
a
s
c
m
om
ul
p
t
r
ip
e
l
h
e
e n
te
si
a
v
c
e
h
k
er
n o
m
w
o
le
d
d
e
g
ls
e
, pt =σ z
T
t =
∑ N i=1 exp
T(
z T i t
) (1.1.1)
base for training the student model (Ahmad et al., 2024). By combining
diverse forms of knowledge, this approach improves the student model’s ( )
g d e is n t T e il r h l a a i l s t i i z o p a n a t p io h e n a r s c p a r e p o v a v o b i l d v il e e i s d t i e a a s . n co d m b p e r e e n h e a n p s p iv li e e d o , v e e r x v p i l e o w ri n o g f h th o e w d k i n v o er w s l e e d a g p e - ps =σ ( z T s ) = ∑ N i e = x 1 e p xp z T s ( z T i s ) (1.1.2)
proaches that have emerged in this rapidly advancing field and
emphasizing the importance of KD in enabling advanced artificial in- Where:
telligence capabilities on resource-constrained devices. The methods Nis the number of classes.
presented in this paper demonstrate the versatility and applicability of zt and zs represent the logits (pre-softmax outputs) of the teacher and
knowledge distillation in a range of real-world contexts. Table 3offers a student networks, respectively.
detailed summary of KD methods employed across various domains. T is the temperature parameter used to soften the logits and normally
The structure of this paper is illustrated in Fig. 2. This figure cate- set to 1 or higher to soften the probability.
gorizes various types of knowledge distillation (KD) techniques based on σ(x)is the softmax function.
their approaches, the types of knowledge they leverage, and their The loss function of Response-Based KD is combined of KL diver-
respective applications (Figs. 3–16). gence distillation loss and the Cross-Entropy (CE) task loss. The KL
divergence loss minimizes the difference between the soft logits of the
teacher and the student, while the CE loss minimizes the error between
3

A. Moslemi et al.                                                                                                                                                                               M   a  c h  i n e    L  e a  r n  i n  g   w   i t h   A   p p  l i c a  t i o  n s 18 (2024) 100605
Table 3
Comprehensive overview of specialized knowledge distillation techniques.
Authors Knowledge schemes Methods Knowledge types Teacher networks Student networks Dataset
Ahmad et al.  Response-Based Distillation;  MTCM-KD Soft Labels CDSFL T1CE modality BraTS-2021
(2024) Multi-teacher cross-modal
distillation;
Ji et al.  Feature-Based Distillation;  AFD Feature Links; Activation  ResNet ResNet CIFAR-100; tinyImageNet;
| (2021a) Attention-Based Distillation | Map |     |     | ImageNet |
| ------------------------------------ | --- | --- | --- | -------- |
Lee et al.  Adversarial Distillation GCNs Feature Matrix; Similarity  ResNet; MobileNet;  ResNet; MobileNet;  CIFAR
| (2022) | Matrix: | Wide ResNet | Wide ResNet |     |
| ------ | ------- | ----------- | ----------- | --- |
Higuchi et al.  Adversarial Distillation adv-CNN Soft Labels ResNet ResNet CIFAR-10
(2022)
He et al.  Adversarial Distillation GraphAKD Discriminators GCNII; GAMLP GCN; Cluster-GCN Cora; CiteSeer; PubMed; Flickr;
| (2022) |     |     |     | Arxiv; Reddit; Yelp; Products |
| ------ | --- | --- | --- | ----------------------------- |
Wu et al.  Multi-teacher Distillation MT-BERT Soft Labels BERT; RoBERTa;  UniLM SST2; RTE; MIND
| (2021) |     | UniLM |     |     |
| ------ | --- | ----- | --- | --- |
Pham et al.  Multi-teacher Distillation CMT-KD Feature Maps AlexNet; ResNet18 AlexNet; ResNet18 CIFAR-100; ImageNet
(2022)
Liu et al.  Multi-teacher Distillation AMTML-KD Soft Labels; Intermediate  ResNet; VGG;  Stu1; Stu2; Stu3 CIFAR; TinyImageNet
| (2020) | Features | DensNet |     |     |
| ------ | -------- | ------- | --- | --- |
Xue et al.  Cross-Modal Distillation MFH; MVD Modality Features ResNet ResNet Gaussian; AVMNIST; RAVDESS;
| (2022) |     |     |     | VGGSound; NYU Depth V2;  |
| ------ | --- | --- | --- | ------------------------ |
MM-IMDB
Sarkar and  Cross-Modal Distillation XKD Attention Maps; Modality  ViT ViT UCF101; HMDB51; Kinetics400;
| Etemad  | Features |     |     | Kinetics-Sound; AudioSet;  |
| ------- | -------- | --- | --- | -------------------------- |
| (2022)  |          |     |     | ESC50; FSD50K              |
Xia et al.  Cross-Modal Distillation MNF; CSC Modality Features ResNet-50 ResNet-50 UCF51; ActivityNet
(2023)
Yang et al.  Graph-Based Distillation LSP Feature Maps GCN GCN PPI; ModelNet40
(2020)
Liu et al.  Graph-Based Distillation HIRE Soft Labels; Semantic  GCN; GAT; RGCN;  GCN; GAT; RGCN;  ACM; IMDB; DBLP
(2022a) Relations; Intermediate  HAN; HGT; HGConv HAN; HGT; HGConv
Features
Wang et al.  Graph-Based Distillation CKD Regional Knowledge;  HIN HIN Pubmed; DBLP; ACM; Freebas
| (2022a) | Global Knowledge |     |     |     |
| ------- | ---------------- | --- | --- | --- |
Ji et al.  Attention-Based Distillation AFD Feature Maps ResNet; Wide  ResNet; Wide  CIFAR-100; tinyImageNet;
| (2021a) |     | ResNet | ResNet | ImageNet |
| ------- | --- | ------ | ------ | -------- |
Passban et al.  Attention-Based Distillation ALP-KD Soft Labels BERT BERT CoLA; MNLI; MRPC; QNLI;
| (2020) |     |     |     | QQP; RTE; SST-2; STS-B |
| ------ | --- | --- | --- | ---------------------- |
Wu et al.  Attention-Based Distillation Universal-KD Intermediate Features BERT BERT GLUE
(2021)
Zhu et al.  Data-Free Distillation FEDGEN Feature Maps FEDGEN FEDGEN MNIST; EMNIST; CELEBA
(2021)
Chawla et al.  Data-Free Distillation DIODE Soft Labels Yolo-V3 Yolo-V3 MSCOCO
(2021)
Fang et al.  Data-Free Distillation FastDFKD Feature Maps ResNet; VGG; WRN ResNet; VGG; WRN CIFAR; NYUv2; ImageNet
(2022)
Kim et al.  Quantized Distillation QKD Feature Maps ResNet; MobileNet ResNet;  CIFAR; ImageNet
| (2019) |     |     | MobilenetV3 |     |
| ------ | --- | --- | ----------- | --- |
Zhao and  Quantized Distillation SQAKD Feature Maps ResNet; VGG MobileNet;  CIFAR-10; CIFAR-100;
| Zhao   |     |     | ShuffleNet;  | TinyImageNet |
| ------ | --- | --- | ------------ | ------------ |
| (2024) |     |     | SqueezeNet   |              |
Boo et al.  Quantized Distillation SPEQ Intermediate Features ResNet; VGG;  ResNet; VGG;  CIFAR10; CIFAR100; ImageNet
| (2021) |     | MobileNet | MobileNet |     |
| ------ | --- | --------- | --------- | --- |
Ye and Bors  Lifelong Distillation;  LT-GANs Discriminator LT-GANs LT-GANs CIFAR; ImageNet; MNIST;
| (2021) Adversarial Distillation |     |     |     | SVHN; Fashion; Omniglot |
| ------------------------------- | --- | --- | --- | ----------------------- |
Zhai et al.  Lifelong Distillation Hyper-  Discriminator Hyper-LifelongGAN Hyper-LifelongGAN Image
(2021) LifelongGAN
Lee et al.  NAS-Based Distillation DaSS Meta-Knowledge; Search  ResNet ResNet TinyImageNe
| (2023) | Strategy |     |     |     |
| ------ | -------- | --- | --- | --- |
Trivedi et al.  NAS-Based Distillation KD-NAS Search Strategy XLM-Roberta KD-NAS CC100
(2023)
Liu et al.  Cross-Architecture  ViT; CNNs Feature Maps; PCA ViT; CNNs ViT; CNNs CIFAR; ImageNet
(2022b) Distillation
Hao et al.  Cross-architecture  OFA-KD Intermediate Features ResNet Swin CIFAR; ImageNet
(2024) distillation
Dong et al.  NAS-Based Distillation DisWOT Feature Semantics ResNet ResNet CIFAR; ImageNet
(2023)
Fang et al.  Cross-Modal Distillation; VL  DistillVLM Soft Labels; Intermediate  VL DistillVLM COCO Captioning; VQA
| (2021a) Distillation | Features |     |     |     |
| -------------------- | -------- | --- | --- | --- |
Fang et al.  Self-Distillation SEED Feature Maps EfficientNet;  EfficientNet;  ImageNet
| (2021b) |     | MobileNet | MobileNet |     |
| ------- | --- | --------- | --------- | --- |
Fang and  Cross-Modal Distillation; VL  VL Soft Labels; Intermediate  VL DistillVLM COCO Captioning; VQA
| Yang  Distillation | Features |     |     |     |
| ------------------ | -------- | --- | --- | --- |
(2023)
4

A. Moslemi et al.                                                                                                                                                                               M   a  c h  i n e    L  e a  r n  i n  g   w   i t h   A   p p  l i c a  t i o  n s 18 (2024) 100605
Fig. 2. The schematic structure of the paper and the relationship between its sections. The body of this survey mainly contains the fundamentals of knowledge
distillation approaches, advanced techniques, and applications of knowledge distillation. Subsections of each main section are listed in this figure.
Fig. 5. A schematic framework of relation-based knowledge distillation.
Fig. 3. A schematic framework of response-based knowledge distillation.
Fig. 4. A schematic framework of feature-based knowledge distillation.
the student’s soft logits and the ground truth labels y.
The KL divergence loss is formulated as:  Fig. 6. A schematic framework of offline knowledge distillation.
| ∑N            | pi(T) |          |
| ------------- | ----- | -------- |
| =T2 pi (T)log | t     |          |
| LKL           | (T)   | (1.1.3)  |
t pi
| i   | s   |     |
| --- | --- | --- |
Where:
piand piis the teacher and the student model’s predicted probability
t  s
5

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
Fig. 7. A schematic framework of online knowledge distillation. Fig. 11. A schematic framework of attention-based knowledge distillation.
Fig. 8. A schematic framework of self-distillation.
Fig. 12. A schematic framework of data-free knowledge distillation.
for class i, respectively.
The CE task loss is formulated as:
∑N (cid:0) )
LCE =CE(y,σ(zs ))=(cid:0) yilog pi
s
(1.1.4)
i=1
Where yi is the truth label for class .
By combining equations (1.1.3)and (1.1.4), the overall loss function
for Response-Based KD becomes:
Fig. 9. A schematic framework of cross-modal knowledge distillation. LResponseKD =αLCE +(1(cid:0) α)LKL (1.1.5)
Where α is a hyperparameter that balances the contributions of the
KL distillation loss and the CE task loss.
Song et al. (2023) analyzed knowledge transfer from weak and
strong teachers, concluding that strong teachers generally lead to more
effective distillation. However, they also demonstrated the value of
weak teachers through their Efficient Distillation (EffDstl) framework,
which minimizes reliance on computationally heavy models. Ahmad
et al. (2024)introduced MTCM-KD, a multi-teacher cross-modal distil-
lation approach for unimodal segmentation. This technique integrates
response-based KD from multiple teachers trained on different modal-
ities, achieving improved segmentation accuracy through a cooperative
deep supervision fusion learning framework. Kim et al. (2023)proposed
Response-Based Cross-Task Knowledge Distillation (RCKD), applying
response-based KD across different tasks. This method uses soft targets
from a teacher model trained on one task to guide a student model on a
related but distinct task, effectively capturing shared patterns and
improving performance in scenarios with limited data.
In summary, response-based KD is widely adopted for its simplicity
Fig. 10. A schematic framework of graph-based knowledge distillation. and direct method of transferring knowledge, making it effective across
various applications. It uses soft target outputs from the teacher model,
which can help avoid overfitting in the student model. However, this
approach can face performance limitations when there is a large ca-
pacity gap between the teacher and student models, which can make it
difficult for the student to fully replicate the teacher’s output patterns,
6

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
Fig. 13. Types of primary adversarial distillation techniques. (a) The generator in the GAN framework creates training data to enhance the effectiveness of
knowledge distillation, with the teacher model serving as the discriminator. (b) The GAN discriminator ensures that the student model, which also acts as a generator,
closely imitates the teacher model. (c) Both teacher and student models operate as a generator, while the discriminator supports online knowledge distillation by
refining the process.
leading to suboptimal learning outcomes (Mirzadeh et al., 2020).
2.2. Feature-based knowledge distillation
Feature-based knowledge distillation transfers internal representa-
tions from a teacher model to a student model, enabling the student to
capture intricate structures and relationships encoded in the teacher’s
feature maps (Yang et al., 2023a). This approach provides richer
knowledge transfer than simply mimicking output probabilities.
The loss function of feature-based KD focuses on aligning the inter-
mediate feature representations between the teacher and student net-
works, and normally can be formulated as:
Fig. 14. A schematic framework of quantized knowledge distillation.
LFeature =
N
1
∑N
‖F
T
i (cid:0) F
S
i ‖2
2
(1.2.1)
i=1
Where:
LFeature is the feature-based KD loss function.
N is the number of feature layers or map points considered for
distillation.
Fi and Fi represent the feature maps of the teacher and student
T S
networks, respectively.
‖⋅‖2
2
denotes the squared Euclidean (L2) norm between the teacher’s
and student’s feature maps.
The total KD loss is a combination of the feature-based KD loss and
Fig. 15. A schematic framework of lifelong knowledge distillation.
the task loss (typically CE loss), expressed as:
LKD =αLCE +(1(cid:0) α)LFeature (1.2.2)
Where α is a hyperparameter that balances the contributions of the
feature-based KD and the CE task loss.
Ji et al. (2021a) proposed an attention-based feature alignment
mechanism in "Show, Attend and Distill: Knowledge Distillation via
Attention-Based Feature Matching." This method uses an attention
mechanism to align intermediate feature representations, ensuring the
student focuses on critical parts of the feature space. In another paper, Ji
et al. (2021b) introduced Feature Refinement via Self-Knowledge
Distillation (FRSKD), a self-teaching approach where a model refines
its own features through iterative self-knowledge distillation. This
method incorporates top-down and bottom-up paths in an auxiliary
self-teacher network, generating refined feature maps and soft labels as
pseudo-labels. Sepahvand et al. (2022) presented a technique for
knowledge distillation in resource-constrained environments, decom-
posing the teacher’s deep feature representations into manageable
components. This approach is particularly suitable for intelligent mobile
Fig. 16. A schematic framework of multi-teacher knowledge distillation. applications.
Overall, feature-based KD captures detailed internal representations
from the teacher model, leading to a richer transfer of knowledge and
improved performance in complex tasks. However, despite its effec-
tiveness, feature-based KD faces challenges, such as difficulty in aligning
feature maps between teacher and student models with different
7

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
architectures or capacities (Yang et al., 2023a); potential transfer of approach has demonstrated superior performance compared to existing
redundant or irrelevant information (Heo et al., 2019); and sensitivity to KD techniques on datasets like Cityscapes and Pascal VOC, highlighting
the selection of layers for knowledge distillation (Yim et al., 2017). the importance of preserving complex relational information in image
segmentation tasks.
In essence, relation-based KD effectively captures structural and
2.3. Relation-based knowledge distillation
relational information within data, making it highly suited for graph-
based models and tasks that require understanding inter-data relation-
Relation-based knowledge distillation is an advanced technique that
ships. However, this technique can be computationally intensive and
preserves structural relationships and dependencies within data learned
may require careful design of relational loss functions to avoid losing
by a teacher model (Park et al. 2019). Unlike earlier KD methods that
critical structural information during transfer.
focus on output probabilities or feature representations, relation-based
A practical example provided by Zhao et al. (2023) demonstrates
KD captures and transfers relational information between different
how response-based, feature-based, and relation-based KD techniques
data points or features, such as maintaining pairwise similarities in the
can be effectively utilized in addressing industry challenges. The study
teacher’s feature space (Tung and Mori, 2019), angular relationships
uses a Semi-Supervised Knowledge Distillation (SSKD) framework to
between data points (Park et al. 2019), or the correlation between in-
improve the performance of deep learning models for vision-based robot
stances (Peng et al., 2019).
guidance in smart manufacturing. The framework addresses the chal-
The relation-based knowledge distillation loss function focuses on
lenge of obtaining sufficient labeled data and the need for efficient
preserving the relational structure of the data learned by the teacher to
models in real-world factory settings. It uses KD methods to transfer
transfer the relational knowledge between pairs or groups of data points
knowledge from a larger, pre-trained teacher model (YOLOv5m) to a
from the teacher network to the student network. After establishing a
smaller, more efficient student model (YOLOv5s). The study employs a
relational measure between data points (e.g., pairwise distance or cosine
combination of response-based, feature-based, and relation-based KD
similarity), the correlation between the teacher’s and student’s feature
techniques to train the student model. This process involves using the
maps is calculated.
teacher model’s outputs, intermediate feature maps, and relational in-
The pairwise distance between two different data points can be
formation between detected objects to guide the student model’s
computed as:
learning. By using KD, the framework reduces inference time from 185
RT (i,j)= ‖F T i (cid:0) F T j‖ 2 (1.3.1) ms to 45 ms while maintaining high accuracy, achieving recall and
precision values exceeding 99.5% and 92.6%, respectively. This results
Additionally, the cosine similarity to measure the similarity between
in a significant improvement in efficiency and generalizability across
feature representations is defined as:
different working environments, ultimately achieving a 200%
Fi⋅Fj improvement in labor efficiency.
RT (i,j)=
‖Fi‖
T
⋅‖
T
Fj‖
(1.3.2)
T T 3. Knowledge distillation approaches
Here Fi and Fj are the feature representations of data points i and j in
T T
the teacher network Similarly, Fi and Fj represent the corresponding Knowledge distillation can be implemented in several ways, each
S S
tailored to different training scenarios and model configurations. The
feature representations of data points i and j in the student network.
primary approaches include offline distillation, online distillation, and
The relation-based knowledge distillation loss can be formulated as:
self-distillation. This section will explore these approaches in detail,
1
∑N ∑N
highlighting their unique characteristics, benefits, and new applications
LRelation =
N2
‖RT (i,j)(cid:0) RS (i,j)‖2
2
(1.3.3)
across various domains.
i=1 j=1
Where: 3.1. Offline distillation
N is the number of data points.
RT (i,j)and RS (i,j)are the distances or the similarities for the teacher Offline knowledge distillation is a technique where knowledge is
and student networks for data points i and j, respectively. transferred from a pre-trained teacher model to a smaller student model
‖⋅‖2
2
denotes the squared Euclidean (L2) norm between the teacher’s during a separate training phase. As described by Gou et al. (2021), this
and student’s relational matrices. method allows for the compression of large, complex models into more
The total KD loss is a combination of the relation-based KD loss and efficient versions suitable for deployment on resource-constrained de-
the task loss (typical CE loss), expressed as: vices while maintaining comparable performance.
The offline knowledge distillation process typically involves two
LKD =αLCE +(1(cid:0) α)LRelation (1.3.4)
main steps. First, a teacher model is pretrained on the dataset. Next, the
Where α is a hyperparameter that balances the contributions of the outputs from this teacher model are used as soft labels to guide the
relation-based KD and the CE task loss. training of a student model. In this stage, the student model learns by
Recent applications have demonstrated the versatility and effec- minimizing the loss between the true labels and the teacher’s soft labels
tiveness of relation-based KD in various domains. Dong et al. (2021) (Gou et al., 2024).
extended the approach to few-shot class-incremental learning (FSCIL) Despite its efficiency, offline KD faces several challenges. The static
with their Exemplar Relation Distillation Incremental Learning (ERDIL) nature of the teacher model can limit the student’s adaptability to new
framework. ERDIL leverages an Exemplar Relation Graph (ERG) to data patterns, and biases in the teacher may be transferred to the student
capture relational knowledge between classes and transfers this (Zhang et al., 2018). The method’s reliance on large, computationally
knowledge through an Exemplar Relation Loss function. This innovative expensive teacher models can be impractical in resource-limited envi-
approach has shown impressive results, outperforming state-of-the-art ronments. Additionally, the student’s performance is often tied to the
methods on benchmark datasets such as CIFAR100 and Mini Image- teacher’s quality, potentially inheriting its limitations (Lan et al., 2018).
Net. In the field of semantic segmentation, Yang et al. (2022)proposed Offline KD may also struggle to transfer fine-grained knowledge from
Cross-Image Relational KD (CIRKD), a method that transfers structured the teacher’s intermediate layers (Chen et al., 2017).
pixel-to-pixel and pixel-to-region relations across entire images. CIRKD Recent studies demonstrate the versatility and effectiveness of offline
employs a memory bank to store rich embeddings, allowing the student KD across various domains. Srinivasagan et al. (2023) applied offline
model to capture global pixel dependencies across different images. This distillation to compress an image-to-speech system for low-resource
8

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
devices, achieving significant reductions in model size and inference allowing continuous feedback among multiple student branches, though
time while maintaining audio quality. In the field of image quality its simultaneous training approach can be computationally intensive
assessment, Yin et al. (2022)used offline KD to transfer knowledge from and requires careful management to maintain representation diversity.
a full-reference teacher model to a non-aligned reference student model,
enhancing performance while maintaining efficiency. Schmid et al. 3.3. Self-Distillation
(2022) employed offline KD to transfer knowledge from a
high-performing Transformer to a more efficient Convolutional Neural Self-distillation is a variant of knowledge distillation where a model
Network (CNN) for large-scale audio tagging, achieving state-of-the-art acts as both the teacher and the student, using its own knowledge to
results on the AudioSet dataset. guide its learning process. Unlike traditional distillation, where a large
Ultimately, offline KD enables efficient compression of large models teacher model transfers knowledge to a smaller student model, self-
for deployment on resource-limited devices, though it can limit the distillation encourages the model to refine its own predictions itera-
student’s adaptability to new data and may inherit biases from the static tively during training (Zhang et al., 2019). While self-distillation can
teacher model. lead to improved model performance and generalization, it has several
limitations. One of the primary challenges is the risk of reinforcing
3.2. Online distillation incorrect or suboptimal predictions, as the model relies heavily on its
own outputs without external supervision. This can lead to overfitting,
Online knowledge distillation is an innovative approach that in- especially if the initial predictions are flawed (Chen & Chu, 2023).
volves the simultaneous training of multiple model branches in a mutual Additionally, self-distillation requires sophisticated training schedules
learning environment. Unlike offline distillation, which requires a pre- and careful tuning of hyperparameters to ensure that the model refines
trained teacher model, online KD allows for continuous feedback and its knowledge in a meaningful way, which can add complexity to the
knowledge sharing between peer models during the learning process. training process (Pham et al., 2022a).
This method aims to enhance overall model performance by improving The typical process of self-distillation involves dividing the model
diversity and preventing homogenization among student branches into segments, such as each residual block in a ResNet model. Each
(Chen et al., 2020). segment’s feature maps are then connected to a fully connected layer
The standard process of online knowledge distillation involves and a classifier. These branched out segments are further connected to
training multiple student models concurrently. The outputs of these each other to enable the transfer of complex knowledge from deeper
models are aggregated and compared with the true labels as well as with blocks to shallower ones (Zhang et al., 2019). In other words, the
each other. This approach allows the student models to learn simulta- deepest block acts as a teacher model for the shallower segments. It
neously and strengthens the robustness of the learning process (Chen transfers knowledge by minimizing the loss between its branch and
et al., 2020). those of the shallower layers, as well as the ground truth. Once training
Nevertheless, online KD faces certain challenges, such as the is complete, the branches are discarded.
increased computational complexity, as simultaneous training of mul- Recent advancements in self-distillation techniques include the work
tiple models can be resource-intensive, particularly for large-scale of Zhang et al. (2021), who introduced a method of self-distillation that
models or datasets (Anil et al., 2018). Managing diversity among stu- progressively refines the network’s internal representations. In this
dent branches remains challenging, as they may still converge toward approach, intermediate layers act as "teachers" for deeper layers,
similar representations without careful design (Chen et al., 2020). allowing the network to become more efficient and compact without
Additionally, the evolving nature of the teacher model during training external supervision. Their method successfully reduces model size
can lead to suboptimal knowledge transfer if it is not adequately robust while maintaining or even improving performance, making it particu-
early in the process (Lan et al., 2018). larly useful for deployment in resource-constrained environments. This
Recent research has demonstrated the versatility and effectiveness of approach has demonstrated strong results in model compression and
online KD in various domains. Li et al. (2022a) proposed a novel efficiency, further supporting the potential of self-distillation in prac-
approach for pixel-level human pose estimation using a Feature Aggre- tical applications.
gation Unit (FAU). This method generates and aggregates heatmaps Kim et al. (2021)introduced progressive self-knowledge distillation
from diverse student branches, resulting in a more robust and accurate (PS-KD), which continuously updates predictions during training to find
pose estimation model. The diversity introduced by different student flatter minima in the loss landscape. While improving generalization
architectures enhances overall accuracy by providing varied perspec- and robustness, PS-KD requires precise control of training dynamics and
tives on the task. Chen et al. (2020)introduced the OKDDip framework, careful hyperparameter tuning. Ge et al. (2021) proposed Batch
which employs a two-stage process to reduce the risk of homogenization Knowledge Ensembling (BAKE), which aggregates predictions from
in student models. This method uses auxiliary student branches and a different mini batches during training. This approach creates more
group leader to aggregate learned knowledge, incorporating an atten- diverse learning targets, improving generalization and mitigating the
tion mechanism to assign weights based on branch performance. Li et al. risk of reinforcing incorrect predictions. BAKE demonstrated significant
(2020) further improved upon this framework by introducing a feature improvements in ImageNet classification accuracy. Yue et al. (2022)
fusion module and a classifier diversification loss function, enhancing combined self-supervised learning with adaptive distillation for hyper-
the diversity and robustness of the knowledge-sharing process. spectral image classification. This method uses self-supervised pre--
Reflecting the real-world utility of this approach, Lin et al. (2023)use training to extract features from unlabeled data, followed by adaptive
online KD, specifically deep mutual learning (DML), to improve the distillation that dynamically adjusts the model’s self-learning process.
accuracy and stability of carbon emission forecasting in the electric This approach effectively handles complex data structures and improves
power industry. The study employs two student models — Long classification accuracy in hyperspectral imaging.
Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) — that The main superiority of self distillation over teacher-student
learn from each other during training. This cooperative learning process approach is that no extra teacher is required. Conversely, teacher-
helps to reduce overfitting and enhances the models’ ability to capture student approach is working based on training an overparameterized
complex patterns within the time series data. By improving forecasting teacher model at first to teach shallow model (student). Designing and
accuracy, the study aims to help power industry decision-makers adjust training the teacher model are the main challenges for traditional KD
power generation policies, ultimately leading to the achievement of techniques. Additionally, training teachers (deep networks) takes long
carbon reduction targets and the goals of carbon peaking and neutrality. time, since teacher model is an over-parameterized network. Zhang et al.
Overall, online KD enhances model diversity and robustness by (2019) experimentally showed that self distillation technique
9

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
outperformed deep networks such as VGG19, ResNet18 and ResNet50 4.1. Complex relationships and context-aware distillation
on CIFAR dataset. The reason why shallow network with self distillation
could outperform all deep supervision is that an extra bottleneck was This section explores advanced KD methods leverage specific
added to identify classifier-specific features. mechanisms or architectural components to enhance knowledge trans-
Dual Teachers for Self-Knowledge Distillation (DTSKD) was pro- fer. Cross-modal distillation transfers knowledge between different data
posed to enhance the performance of self distillation approach (Li et al. modalities (e.g., from images to text). Graph-based distillation employs
2024a). In DTSKD, student is trained by two teachers based on self su- graph structures to capture and convey relational information. Finally,
pervision staretgy from two inherently different domains including the attention-based distillation utilizes attention mechanisms to align the
past learning history and the current network structure. In specific, focus of the student model with that of the teacher. By exploiting these
historical teacher and structural teacher are two teachers of DTSKD to mechanisms, the student model can more effectively replicate the
train student. Historical teacher provides knowledge from the previous teacher’s performance.
epoch and structural teacher distils the knowledge form the current
iteration. Yang et al. (2023b) proposed Universal Self-Knowledge 4.1.1. Cross-modal distillation
Distillation (USKD). In USKD, KD loss is decomposed to a Normalized Cross-modal knowledge distillation involves transferring knowledge
KD (NKD) loss and soft labels are constructed for both target class and from a teacher network trained on one modality, such as images, to a
non-target classes. USKD constructs soft labels for both target and student network trained on a different modality, such as audio. This
non-target classes without having a teacher by smoothing the target approach allows the student model to leverage rich information from
logit of the student as the soft target label. Lee et al. (2023b)proposed another modality, often improving performance in tasks where data
self- distillation drop-out (SD-Dropout) to reduce the number of train- from one modality is scarce or less informative (Gupta et al., 2016).
able parameters. In SD-Dropout, distributions of multiple models are However, cross-modal KD introduces unique challenges, such as align-
distilled using a dropout sampling. SD-Dropout is simple yet effective ing disparate feature spaces and dealing with modality-specific noise or
approach to train a shallow model based on self supervise learning. irrelevant features that can hinder the effectiveness of the knowledge
Inconsistency between deep and shallow classifiers in self-KD during transfer (Wang et al., 2023; Huo et al., 2024).
distilling knowledge is one of the concerns. In this regard, Liang et al. A common cross-modal distillation process incorporates several
(2024) proposed Neighbor Self-Knowledge Distillation (NSKD) to main stages. First, a high-capacity teacher model is trained on the source
circumvent mismatch problem between deep and shallow classifiers modality with abundant labeled data, learning rich feature representa-
using auxiliary classifiers which are added to the shallow parts of the tions. Second, a student model designed for the target modality is
network. These auxiliary classifiers provide distillations of multiple initialized, often facing limited labeled data (Gupta et al., 2016). Third,
neighboring which leads to reduce the mismatch. mechanisms are established to align the feature spaces of the teacher
To demonstrate a practical application of self-distillation, the study and student models, such as shared embedding spaces or adaptation
by Li et al. (2024b)proposes a deep knowledge distillation model for layers that bridge the modality gap. Fourth, the cross-modal distillation
traffic prediction that leverages self-distillation and mutual learning loss function is designed to transfer knowledge between two different
techniques to improve the accuracy of traffic flow forecasting. The modalities, such as from a teacher model trained on one modality (e.g.,
model employs two graph neural networks with encoder-decoder vision) to a student model working on another modality (e.g., audio or
structures. Self-distillation is employed within each network, text) (Gupta et al., 2016; Huo et al., 2024). This loss function typically
enhancing feature sensitivity by transferring knowledge from the deeper combines a task-specific loss (e.g., CE loss) with a distillation loss (e.g.,
structure to the shallower structure. Mutual learning enables the two KL Divergence) that helps align the student’s predictions or feature
networks to learn collaboratively, improving feature learning and representations with those of the teacher. Finally, the student model is
generalization by minimizing the divergence between their predictions. trained using both the distillation loss and any available supervised loss
Experiments on real-world datasets demonstrate the model’s effective- on the target modality, enabling it to leverage the teacher’s knowledge
ness, particularly for long-term predictions (beyond 30 minutes). The despite the modality differences.
deep knowledge distillation model achieved lower Mean Absolute Error Xue et al. (2022) introduced the Modality Focusing Hypothesis
(MAE), Mean Absolute Percentage Error (MAPE), and Root Mean Square (MFH) and Modality Venn Diagram (MVD) to understand and improve
Error (RMSE) compared to the baseline DCRNN model, with improve- cross-modal distillation. They proposed adjusting the teacher network to
ments of 0.19, 0.7%, and 0.49 respectively on the METR-LA dataset. On focus on modality-general features, significantly improving the student
the PEMS-BAY dataset, the model achieved even better results, reducing model’s performance by preserving relevant information during
MAE by 0.18, MAPE by 0.4%, and RMSE by 0.41. These results indicate cross-modal transfer. Sarkar and Etemad (2022) developed XKD, a
the effectiveness of the hybrid KD approach in capturing spatiotemporal self-supervised framework for learning transferable features across
dependencies and enhancing the feature extraction capabilities of graph video modalities. Their method combines Masked Data Modeling with
neural networks for improved traffic flow prediction. Knowledge Distillation and Domain Alignment, using attention maps
In summary, self-distillation allows a model to refine its predictions and maximum mean discrepancy loss to align features between audio
by learning from its own knowledge, which can lead to enhanced per- and visual streams. This approach improves the generalization and
formance without needing a larger teacher model. However, this tech- transferability of learned representations in various video-related tasks.
nique may inadvertently reinforce its own inaccurate predictions and Xia et al. (2023) addressed challenges in cross-modal knowledge
requires precise training schedules and fine-tuning, which can add transfer for unconstrained videos. They proposed the Modality Noise
complexity to the process. Filter (MNF) to remove irrelevant modality-specific noise and the
Contrastive Semantic Calibration (CSC) module to align visual and audio
4. Advanced Techniques in Knowledge Distillation features based on semantic relevance. These innovations ensure more
effective knowledge transfer in noisy, unconstrained video
Advanced techniques in knowledge distillation expand on traditional environments.
methods to tackle more complex tasks. These methods can be grouped To demonstrate this method in action, the study by Yang and Xu
into four main categories: Complex Relationships and Context-Aware (2021) proposes a cross-modality knowledge distillation (CMKD)
Distillation, Data-Free and Synthetic Data Distillation, Model method for multi-modal aerial view object classification, aiming to
Compression and Architecture Optimization, and Ensemble Distillation improve the performance of object classification models using both
Methods. This section provides an overview of these techniques and synthetic aperture radar (SAR) and electro-optical (EO) images. The
their various applications across different domains. study uses two different network structures, CMKD-s and CMKD-m, to
10

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
transfer knowledge between SAR and EO modalities. CMKD-s employs performance of heterogeneous graph neural networks (HGNNs) across
online KD to transfer information between the two sensors, enhancing multiple datasets. Wang et al. (2022a)proposed Collaborative Knowl-
the robustness of the aerial view object classification model. CMKD-m edge Distillation (CKD) for embedding nodes in heterogeneous infor-
further improves upon this by introducing a semi-supervised enhanced mation networks (HINs). CKD uses a three-stage approach: Semantic
training approach, enabling mutual knowledge transfer between the Context Subgraph Sampling, Heterogeneous Knowledge Modeling, and
models. Both CMKD-s and CMKD-m were evaluated on the NTIRE2021 Collaborative Knowledge Distillation. This method effectively handles
SAR-EO challenge dataset, achieving higher accuracy compared to a large datasets and complex HINs, achieving superior performance in
baseline method without knowledge transfer. The study demonstrates node classification and link prediction tasks.
that leveraging KD techniques across different imaging modalities can Overall, graph-based methods excel in tasks that require an under-
significantly improve the accuracy and robustness of object classifica- standing of structural dependencies but, similar to relation-based KD,
tion models, particularly in challenging scenarios where a single sensor this technique may increase computational demands due to the
may not capture sufficient information. complexity of graph data.
To recapitulate, cross-modal KD enriches the student model’s
versatility across modalities, though alignment of feature spaces can be 4.1.3. Attention-Based Distillation
complex and may require handling modality-specific noise. Attention-based distillation leverages attention mechanisms to
enhance the process of transferring knowledge from a teacher model to a
4.1.2. Graph-based distillation student model by focusing on the most relevant features or representa-
Graph-based knowledge distillation focuses on transferring knowl- tions. Unlike traditional methods that manually select feature align-
edge from teacher models to student models specifically designed for ments between teacher and student models, attention-based distillation
graph-structured data. The graph structure allows for the preservation of automatically identifies and emphasizes important regions of the
high-order dependencies and structural information that might be lost in model’s outputs or intermediate layers (Zagoruyko & Komodakis,
conventional distillation techniques (Zhang & Peng, 2018). 2016). This approach improves the efficiency and effectiveness of
A typical graph-based KD architecture consists of a teacher and a knowledge transfer by enabling the student model to focus on the most
student graph neural network (GNN). The teacher model is responsible informative aspects of the teacher’s learned representations.
for capturing detailed structural patterns and relational information Attention-based distillation employs activation-based attention
from the graph data. The student model aims to replicate the teacher’s transfer or gradient-based attention transfer methods (Zagoruyko &
performance by learning from its representations. According to Liu et al. Komodakis, 2016). In activation-based attention transfer, the attention
(2023), this is achieved by aligning the embeddings and outputs of the maps are computed using the activations of a teacher network. The goal
student GNN with those of the teacher through specialized distillation is to train the student network not only to make accurate predictions but
loss functions that consider the graph topology and node relationships. also to produce attention maps that closely resemble those of the
This architecture enables the student model to effectively inherit the teacher. This helps the student network mimic the teacher’s focus on
teacher’s ability to understand complex graph structures while being specific spatial regions of the input. While in gradient-based attention
more efficient in computation. transfer, attention is encoded by the sensitivity of the model’s output
The graph-based distillation loss (Liu et al., 2023) encourages the prediction with respect to the input spatial locations. This approach
student network to learn the relational structure (graph) between data evaluates how changes in specific input locations (e.g., pixels) affect the
points as captured by the teacher. A common approach is to minimize output, implying that the network is "paying attention" to those
the squared difference (Frobenius norm) between the adjacency locations.
matrices of the teacher and student graphs: The activation-based attention transfer loss can be formulated as:
LGraph =
N
1
2
∑N ∑N
‖AT (i,j)(cid:0) AS (i,j)‖2
2
(3.1.2.1) Lactivation =
N
1
∑N
‖f
(cid:0)
Ai
T
)
(cid:0) f
(cid:0)
Ai
S
)
‖2
2
(3.1.3.1)
i=1 j=1 i=1
Where: Where:
N is the number of data points. N is the number of spatial locations in the attention map.
AT (i,j)and AS (i,j)represent the strength of the relationship between Ai
T
and Ai
S
are the attention values at location i for the teacher and
the i -th and j -th data points in the teacher’s and student’s graphs, student networks, respectively.
respectively. f(x)is the attention mapping function.
‖⋅‖2
2
denotes the squared Euclidean (L2) norm between the entries of ‖⋅‖2
2
denotes the squared Euclidean (L2) norm between the teacher’s
the teacher’s and student’s adjacency matrices. and student’s attention maps.
The total KD loss is a combination of the graph-based KD loss and the The gradient-based attention transfer loss can be formulated as:
task loss (typical CE loss), expressed as:
1
∑N
LKD =αLCE +(1(cid:0) α)LGraph (3.1.2.2) Lgradient = N
i=1
‖gT (xi )(cid:0) gS (xi )‖2 2 (3.1.3.2)
Where α is a hyperparameter that balances the contributions of the
Where:
graph-based KD and the CE task loss. gT (xi )and gS (xi )the gradient-based attention values at input location
Yang et al. (2020)introduced the Local Structure Preserving (LSP)
xi for the teacher and student networks, respectively.
module for Graph Convolutional Networks (GCNs). This module trans-
Guo et al. (2023)instituted a class attention transfer method where
fers topological knowledge from teacher to student GCNs by preserving
they proposed a novel attention-based distillation framework that fo-
local graph structures, ensuring the student retains the input graph’s
cuses on class-specific features. In their architecture, attention maps are
topological relationships. The method adapts to dynamic graphs and
generated for each class, and the student model is trained to mimic these
shows strong performance in node classification and 3D object recog-
class-wise attention maps from the teacher model. This class-wise
nition tasks. Liu et al. (2022a) developed HIRE, a framework for
attention alignment allows the student to capture more discriminative
distilling knowledge from heterogeneous graphs. HIRE combines
features relevant to each class, leading to improved performance in
Node-level Knowledge Distillation (NKD) and Relation-level Knowledge
classification tasks. By concentrating on the class-specific attention, the
Distillation (RKD) to transfer soft labels and capture high-order semantic
student model benefits from a more targeted knowledge transfer, which
relations between different node types. This approach enhances the
11

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
enhances its ability to generalize and recognize intricate patterns asso- noise inputs so that their activations match those recorded from the
ciated with each category. teacher model (Lopes et al., 2017). The student model is then trained on
Ji et al. (2021a) introduced Attention-based Feature Distillation these synthetic samples using a combination of loss functions:
(AFD), which uses an attention mechanism to identify similarities be- cross-entropy loss measures how well the student predicts target labels;
tween teacher and student networks. AFD automatically controls the activation-based distillation loss aligns internal representations between
intensity of the distillation process based on feature similarity, elimi- the student and teacher networks; and information entropy loss mini-
nating the need for manually selected links and ensuring the student mizes uncertainty in the student’s predictions by pushing the softmax
focuses on the most relevant teacher features. Passban et al. (2020) outputs toward one-hot vectors (Lopes et al., 2017). This framework
proposed ALP-KD, an attention-based layer projection technique allows the student to learn effectively from the teacher without access-
addressing the skip and search problems in intermediate layer distilla- ing the original training data.
tion. ALP-KD uses attention to project information from all teacher The data-free distillation the loss function combines three compo-
layers to the student, weighted by relevance, ensuring the student learns nents: cross-entropy loss, activation-based distillation loss, and infor-
from all teacher layers even when there’s a mismatch in layer depth. Wu mation entropy loss. While the cross-entropy loss measures how well the
et al. (2021b) developed Universal-KD, an attention-based out- student model can predict the target labels, and the activation-based
put-grounded intermediate layer distillation method. It uses attention to distillation loss helps align the internal representations (activations)
assign weights to pseudo classifiers attached to intermediate layers, between the student and teacher networks, the information entropy loss
providing interpretable output space probabilities. This approach ad- is used to minimize uncertainty in its predictions by pushing the softmax
dresses the capacity gap between different-sized models and supports outputs toward one-hot vectors. The information entropy loss is given
cross-architecture distillation. by:
To showcase how this method functions in practical scenarios, a
s
e
t
n
u
h
d
a
y
n c
b
e
y
th
Lo´
e
p
p
e
e
z
r
-
f
C
o
i
r
f
m
ue
a
n
n
t
c
e
e
s
o
e
f
t
a
a l
s
.
m
(
a
2
l
0
le
2
r
3 )
stu
u
d
ti
e
li
n
z
t
e s
n e
a
t
t
w
te
o
n
r
t
k
io
i
n
n
- b
sc
a
e
se
n
d
e r
K
e
D
co
t
g
o
-
Linformation =(cid:0)
N
1 ∑
i=
N
1
pi
S
log (cid:0) pi
S
) (3.2.1.1)
nition by transferring knowledge from a larger teacher network. This is
By combining equations (1.1.4), (3.1.3.1), and (3.2.1.1), the total
achieved by matching the activation maps, which represent attention to
loss for data-free distillation can be expressed as:
relevant image regions, between the teacher and student networks.
Instead of relying on pixel-wise comparisons using the ℓ2 norm, this LKD = αLCE +βLactivation +γLinformation (3.2.1.2)
study introduces a novel DCT-based metric to compare the activation
Where α, β, γare hyperparameters that balance the importance of
maps in the frequency domain. This method aims to better capture
the cross-entropy loss, activation-based distillation loss, and information
global image cues and spatial relationships that are crucial for scene
entropy loss, respectively.
recognition, enabling the student network to learn more descriptive
Zhu et al. (2021)introduced FEDGEN for heterogeneous federated
features and achieve higher performance. By minimizing the differences
learning. This method uses a lightweight generator to aggregate user
between the transformed activation maps, the student network learns to
information without external data, improving model generalization and
focus on the same relevant image areas as the teacher network,
reducing communication rounds in decentralized environments where
improving its ability to recognize complex scenes.
data privacy and heterogeneity are critical. Chawla et al. (2021)
In summary, attention-based KD enhances transfer efficiency and
developed DIODE (DeepInversion for Object Detection) for data-free
interpretability but requires careful tuning to ensure attention is focused
distillation in object detection tasks. DIODE synthesizes high-quality
on the most informative features.
images by inverting a pre-trained object detection network, employing
differentiable augmentations and a novel bounding box sampling
4.2. Data-Free and Synthetic Data Distillation
strategy. This approach outperforms proxy datasets in distillation effi-
cacy when real data is inaccessible. Fang et al. (2022) proposed
This section focuses on distillation methods that operate without
FastDFKD, accelerating data-free distillation by reusing common fea-
access to the original training data. Data-Free Distillation and Adver-
tures from training data. A meta-synthesizer learns and reuses shared
sarial Distillation generate synthetic data, often using techniques like features across multiple data points, achieving 10 ×to 100 ×speedup in
GANs or model inversion, to enable the student model to learn from the
data synthesis while maintaining competitive performance on various
teacher. By creating artificial data that approximates the original data
tasks. This method is particularly beneficial for large-scale tasks where
distribution, these methods facilitate effective knowledge transfer while
traditional DFKD methods are computationally prohibitive.
addressing privacy or data availability concerns.
To illustrate a practical application of data-free distillation, Xiang
et al. (2024) proposed a novel technique called Data-Free Knowledge
4.2.1. Data-Free Distillation
Distillation for Diffusion Models (DKDM) that significantly accelerates
Data-free knowledge distillation (DFKD) is a technique that enables
diffusion models without requiring access to the original training data.
the transfer of knowledge from a teacher model to a student model
Diffusion models excel at generating realistic images, videos, and audio
without requiring access to the original training data (Chen et al.,
but suffer from slow inference speeds. DKDM addresses this limitation
2019b). This approach is especially valuable in scenarios where the data
by distilling the knowledge from a large, pre-trained "teacher" diffusion
is proprietary, sensitive, or unavailable due to privacy concerns (Lopes,
model into a faster "student" model with a smaller architecture. The key
2017). Data-free distillation techniques typically rely on synthetic data
innovation lies in DKDM’s ability to achieve this distillation without
generation, where the student model learns from examples created by
using the source data. The authors achieve this by synthesizing
the teacher model or a generator, rather than directly from the original
denoising data from the teacher model and employing a dynamic iter-
dataset. However, the absence of real data introduces unique challenges,
ative distillation method to prevent the data generation process from
such as generating high-quality, informative synthetic samples that
becoming the main bottleneck. Experiments demonstrate that DKDM
effectively capture the underlying data distribution and ensuring effi-
successfully compresses diffusion models while preserving comparable
cient training without the use of ground-truth data (Chen et al., 2019b).
performance, enabling up to 2x faster models. The paper concludes that
A typical data-free distillation process involves generating synthetic
DKDM is a significant advancement in the field, allowing for efficient
data using information extracted from the teacher model to train the
model compression and facilitating wider adoption of diffusion models
student model. In this approach, ’meta-data’ is transferred from the
across diverse applications.
teacher’s batch normalization layers to reconstruct the data distribution.
In summary, the process of data-free knowledge distillation involves
Specifically, synthetic samples are generated by optimizing random
12

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
utilizing activation statistics to bypass the need for original training (2022) proposed a framework that focuses on both final outputs and
data. These statistics capture the specific neurons activated during the intermediate representations in CNNs. This method combines adversa-
training of the teacher model (Lopes et al., 2017). By leveraging these rial training for robustness with knowledge distillation, aligning inter-
activation statistics and labels from the teacher model, it is possible to mediate feature representations between teacher and student models to
recreate a replica of the original training data, which can then be used to enhance resilience against adversarial attacks. He et al. (2022)devel-
train the student model. oped a technique for compressing Graph Neural Networks (GNNs) using
adversarial knowledge distillation. Their student-teacher framework
4.2.2. Adversarial distillation challenges the student GNN to match the teacher’s outputs while pre-
Adversarial knowledge distillation (AKD) is a technique that com- serving essential graph structural information, resulting in efficient,
bines the principles of adversarial training and knowledge distillation to smaller GNNs that maintain performance on benchmark datasets.
enhance the performance and robustness of student models. In AKD, the For a concrete example of this approach in action, Bai et al. (2022)
student model is trained not only to mimic the teacher model’s outputs use adversarial knowledge distillation to improve the performance of
but also to resist adversarial attacks, ensuring that the student model BioBERT, a pre-trained language model, on a biomedical factoid
retains essential knowledge while improving its generalization capa- question-answering task. The goal is to enhance the model’s ability to
bilities (Goodfellow et al., 2014). identify the exact answer to a question within a given passage of text.
AKD employs three primary strategies to enhance the knowledge This is achieved by using a teacher-student framework. The teacher
transfer process between teacher and student models. First, it utilizes an model is also based on BioBERT but includes an additional module
adversarial generator to create synthetic data, which either serves as the designed to capture question-answer interaction knowledge. This
main training dataset or augments existing data. Second, AKD in- knowledge is then distilled to the student model, which is a basic Bio-
corporates one or more discriminators that distinguish between outputs BERT model. To further enhance the student model’s robustness and
from the student and teacher models, using either logits or features, and prevent overfitting, adversarial training is incorporated into the distil-
leveraging unlabeled data to facilitate knowledge transfer. Finally, AKD lation process. This involves constructing perturbed training examples
implements a joint optimization approach where both the teacher and by adding noise to the original training data. By forcing the student
student models are simultaneously refined in each training iteration. model to mimic the teacher model’s predictions on both original and
These methods collectively harness adversarial techniques to boost the perturbed examples, the student model can learn the knowledge of
efficiency of knowledge distillation, ultimately leading to improved question-answer interaction and improve its performance on the
performance in the student model while maintaining a compact archi- question-answering task.
tecture (Gou et al., 2021). In essence, adversarial knowledge distillation improves generaliza-
The AKD loss function typically consists of two components: the tion and security but requires careful balancing of adversarial and
discriminator loss and the adversarial loss. These losses are optimized distillation losses, which can be challenging to achieve without
separately, with the discriminator loss training the discriminator to impacting accuracy.
differentiate between the teacher and student outputs, and the adver-
sarial loss training the student to fool the discriminator by mimicking
the teacher’s outputs. 4.3. Model compression and architecture optimization
Since the discriminator is trained to maximize its ability to differ-
entiate between the teacher’s output and the student’s output, the dis- This section encompasses distillation techniques aimed at producing
criminator’s loss can be formulated as: efficient student models with reduced size and computational re-
quirements. Quantized distillation combines quantization with distilla-
LDiscriminator = (cid:0) Ex∼Pdata [log(D(T(x)))](cid:0) Ex∼Pdata [log(1(cid:0) D(S(x)))] tion to create smaller, faster models suitable for deployment on
(3.2.2.1) resource-constrained devices. NAS-based distillation integrates Neural
Architecture Search (NAS) to automatically find the optimal student
Where:
model architecture, balancing performance and efficiency during the
Ex are the expectation value of the input data points.
distillation process.
Pdata refers to the data distribution from which the training data or
inputs are sampled.
x∼Pdata means that xis a random variable sampled from the prob- 4.3.1. Quantized Distillation
Quantized knowledge distillation (QKD) focuses on improving the
ability distribution Pdata, which is typically the distribution.
D(T(x)) is the probability of the discriminator’s output for the performance of quantized neural networks (QNNs) by leveraging
teacher’s output. knowledge distillation techniques (Kim et al., 2019). QNNs, which
D(S(x))is the probability of the discriminator’s output for the stu- reduce the precision of weights and activations to lower-bit represen-
dent’s output. tations, are designed for deployment on resource-constrained devices
like mobile phones and embedded systems. Although quantization
While the discriminator tries to maximize its loss, student network is
significantly reduces memory usage and increases computational effi-
trained to minimize its adversarial loss by generating outputs that can
ciency, it often leads to performance degradation compared to
"fool" the discriminator into classifying them as if they were produced by
full-precision models. Quantized distillation addresses this problem by
the teacher. The adversarial loss function can be formulated as:
using knowledge from a full-precision teacher model to guide the
LAdversarial = (cid:0) Ex∼Pdata [log(D(S(x)))] (3.2.2.2) training of a quantized student model, helping the student overcome the
accuracy loss that typically accompanies quantization (Kim et al., 2019).
One of the main challenges of AKD lies in balancing the adversarial
The total loss function for QKD combines the CE task loss, the KL
and distillation losses, as improper tuning can lead to either reduced
divergence distillation loss, and the quantization-aware loss. The
accuracy or vulnerability to adversarial examples (Wang et al., 2018).
quantization-aware loss minimizes the difference between the quantized
To enhance the transfer of complex knowledge structures between
and full-precision student activations or weights by penalizing the
teacher and student models, Lee et al. (2022)introduced a method using
Graph Convolutional Networks (GCNs) to preserve similarity relation-
quantization error between the quantized student weights Wq and the
ships between data samples during distillation. By combining traditional
full-precision student weights Wf:
d
an
is
d
ti l
m
la
a
t
i
i
n
on
ta i
l
n
os
s
s
r
w
el
i
a
t
t
h
io
a
n
d
a
v
l
e
i
r
n
s
t
a
e
r
g
ia
ri
l
t y
lo
i
s
n
s,
t
t
h
h
e
is
s
a
tu
p
d
p
e
r
n
oa
t
c
m
h
o
i
d
m
e
p
l.
r o
H
v
i
e
g
s
u c
a
h
cc
i
u
e
r
t
a
a
c
l
y
.
LQuantized = ‖Wf (cid:0) Wq ‖2
2
(3.3.1.1)
Where Wf and Wq represent the full-precision weights and the
13

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
quantized weights, respectively. optimal architecture is identified, balancing performance metrics like
The total loss for quantized distillation can be expressed as: accuracy, model size, and computational efficiency, it is trained from
scratch using conventional methods and the distilled knowledge from
LKD = αLCE +βLKL +γLQuantized (3.3.1.2)
the teacher.
Where α, β, γ are hyperparameters that balance the importance of the However, traditional NAS methods are often computationally
CE task loss, the KL divergence distillation loss, and the quantization- expensive for optimizing the neural architecture for the target task and
aware loss, respectively. can struggle to generalize small datasets (Elsken et al., 2020). Moreover,
A typical quantized KD architecture involves a full-precision teacher a significant drawback is that most NAS approaches concentrate mainly
model and a quantized student model, where the teacher provides soft on optimizing accuracy, model size, or computational efficiency
targets and activation maps to guide the student in learning to replicate (FLOPs), while paying insufficient attention to the robustness of the
the teacher’s outputs despite the reduced precision of its own weights searched architectures against adversarial attacks. This lack of focus on
and activations. Kim et al. (2019)proposed QKD, a three-phase method adversarial robustness is crucial because it affects the security and
for improving quantized neural networks. It involves "self-studying" for resilience of machine learning systems in real-world applications (Nath
good initialization, "co-studying" to make the teacher more et al., 2024)
quantization-friendly, and "tutoring" to transfer knowledge efficiently. The total loss function for NAS-based distillation combines the CE
The architecture incorporates quantization operations into the forward task loss, the KL divergence distillation loss, and the architecture search
and backward passes of the student model, allowing it to learn quanti- loss (Liu et al., 2019b). The architecture search loss optimizes the ar-
zation effects during training. The loss function combines the standard chitecture of the student network, which normally is formulated as:
classification loss with a distillation loss that measures the discrepancy
between the teacher’s and student’s outputs. By integrating quantiza-
LArchitecture = Eα∼A [Lval (w(α),α)] (3.3.2.1)
tion processes into the knowledge distillation framework, the student Where:
model can achieve higher accuracy compared to traditional quantization Eα∼A is the expectation over architectures α sampled from the ar-
methods, effectively narrowing the performance gap between quantized chitecture search space A.
and full-precision models. This comprehensive approach significantly w(α)is the weight of the student network given an architecture α,
improves the performance of quantized models, making them compa- which is optimized with respect to the task-specific loss on the training
rable to full-precision models while maintaining reduced computational set.
demands. Lval is the validation loss, used to guide architecture optimization
Zhao and Zhao (2024)introduced a self-supervised quantization-a- based on how well the architecture performs on the validation set.
ware knowledge distillation (SQAKD) method. This approach combines The total loss for NAS-based distillation can be expressed as:
self-supervised learning with SQAKD, using proxy tasks and unlabeled
data to train models. It improves the efficiency and accuracy of quan-
LKD = αLCE +βLKL +γLArchitecture (3.3.2.2)
tized models while reducing reliance on labeled datasets, making it Where α, β, γ are hyperparameters that balance the importance of the
valuable when labeled data is scarce. Boo et al. (2021)developed the CE task loss, the KL divergence distillation loss, and the architecture
Stochastic Precision Ensemble (SPEQ) technique. SPE assigns random search loss, respectively.
precision levels during training, creating an ensemble of models with Lee et al. (2023) introduced Distillation-aware Student Search
varying precisions. Knowledge from this ensemble is distilled into a (DaSS), a method that predicts student architecture performance
single quantized model, enabling it to learn from diverse precision without extensive training on target tasks. DaSS incorporates a
scenarios. This self-knowledge distillation approach mitigates accuracy distillation-aware task encoding that adapts to the teacher model’s ac-
loss in quantized models, resulting in robust and efficient models for curacy, enabling efficient and scalable architecture search. By
edge devices. leveraging meta-prediction and gradient-based adaptation, DaSS gen-
Ultimately, QKD provides an efficient model for low-resource envi- eralizes across different datasets and teacher models, reducing compu-
ronments, though precision reduction can introduce noise, impacting tational costs while improving the quality of discovered student models.
performance if not managed carefully. Trivedi et al. (2023)proposed KD-NAS, a system combining NAS with
knowledge distillation to optimize student architectures for multilingual
4.3.2. NAS-Based Distillation language models. KD-NAS uses a NAS controller to predict rewards
Neural Architecture Search (NAS) is a powerful method for auto- based on distillation loss and inference latency, selecting top candidate
matically discovering optimal neural network architectures tailored to architectures for knowledge transfer. It also introduces a multi-layer
across a large variety of artificial intelligence tasks (Liu et al., 2019a). hidden state distillation method, allowing students to learn from mul-
When combined with knowledge distillation, NAS-based distillation tiple teacher layers without requiring pre-training. KD-NAS demon-
aims to find the most effective student architectures for learning from a strates significant improvements in speed and efficiency while
pre-trained teacher model. The goal is to improve student model per- maintaining strong performance in multilingual tasks.
formance while maintaining efficiency, making NAS-based distillation In essence, NAS for knowledge distillation involves searching within
particularly useful for creating smaller, faster models suited for a predefined space of student architectures (Li et al., 2020). During this
deployment on resource-constrained devices. process, a pre-trained teacher model transfers its knowledge to a
A typical NAS-based distillation process involves an iterative search candidate student model. After the search is completed, the student
that explores a vast space of potential student models, guided by model that best meets the desired objectives, such as performance and
knowledge distillation from a robust teacher model (Nath et al., 2024). computational efficiency, is selected.
Initially, a weight-sharing super-network is constructed, encompassing Ultimately, NAS-based distillation is powerful for designing efficient
all candidate architectures within a predefined search space. This over- models but can be computationally expensive due to the architecture
parameterized network includes distinct paths for each possible archi- search process and may overlook robustness to adversarial attacks,
tecture, with shared weights optimized through gradient descent during which is essential for secure applications.
training. Throughout the search phase, each candidate architecture (or
path) is evaluated based on its performance when distilled from the 4.4. Ensemble Distillation Methods
teacher model. The student models not only learn from the teacher’s soft
labels but may also align their intermediate feature representations with This section provides an overview of several ensemble distillation
those of the teacher, enhancing both accuracy and robustness. Once the methods, including lifelong distillation and multi-teacher distillation.
14

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
Lifelong distillation applies knowledge distillation within a continual demonstrates effectiveness in image generation tasks, balancing new
learning framework, enabling the student model to incrementally learn learning with knowledge retention. Zhai et al. (2021) developed
new tasks over time without forgetting previous knowledge. Multi- Hyper-LifelongGAN, a scalable architecture for lifelong learning in
teacher distillation involves distilling knowledge from multiple image-conditioned generation tasks. It decomposes filters into dynamic
teacher models into a single student model, aggregating expertise from base filters and a shared weight matrix, using task-specific coefficients to
various sources to improve generalization and performance. adjust for each task. This approach leverages knowledge distillation to
retain previous task knowledge, addressing catastrophic forgetting and
4.4.1. Lifelong Distillation scalability issues efficiently.
Lifelong distillation focuses on enabling machine learning models to As a practical application, the study by Li et al. (2021)on continual
learn continuously from a sequence of tasks while retaining knowledge multi-task image restoration uses lifelong knowledge distillation on both
from previously learned tasks (Wang et al., 2022b). This approach ad- the generators and discriminators in its CycleGAN framework. The goal
dresses the challenge of "catastrophic forgetting," where a model forgets is to alleviate the catastrophic forgetting problem, where a model forgets
previously learned information when trained on new tasks. Lifelong previously learned tasks when learning new ones. The distillation pro-
knowledge distillation allows models to retain essential knowledge cess involves constraining the current task’s generator and discriminator
through teacher-student frameworks, where the teacher model guides outputs to be similar to those of the previous task’s models. This ensures
the student to retain old knowledge while learning new tasks (Wang that knowledge acquired from past tasks is retained while learning a new
et al., 2022b). This method is particularly useful for applications that task. Specifically, the generators for the current task are constrained to
require continuous learning and adaptation, such as autonomous sys- produce outputs similar to the previous task’s generators, ensuring the
tems, generative models, and image processing tasks. transfer of learned restoration techniques. Similarly, the discriminators
Lifelong distillation typically involves a dynamic teacher-student for the current task are trained to have consistent outputs with those of
framework, in which the teacher model retains knowledge from previ- the previous task, promoting the preservation of learned image quality
ous tasks, while the student model focuses on learning new tasks while evaluation. This approach enables the model to continually learn and
preserving previously acquired knowledge. In this architecture, the adapt to new image restoration tasks, such as low-light enhancement,
teacher model can be an earlier snapshot of the student, or an ensemble deblurring, and denoising, without losing proficiency in previously ac-
of models trained on previous tasks. The student model learns from new quired skills.
data and receives guidance from the teacher through knowledge distil- To recapitulate, lifelong KD enables continuous learning by allowing
lation techniques that minimize discrepancies between their outputs models to acquire new information while retaining previous knowledge.
(Hong et al., 2020). This setup ensures that the student model not only However, a challenge is that prior knowledge may not always be
acquires new knowledge but also retains proficiency in previously applicable to future tasks due to differences in task domains. Addition-
learned tasks. However, a drawback of lifelong KD is that the "previous ally, balancing new and old knowledge without excessive retention loss
knowledge" might not be easily applicable or reusable for future tasks is crucial for optimal performance across tasks, as excessive focus on
due to differences in task domains (Hong et al., 2020). Therefore, preserving past knowledge can hinder the learning of new tasks.
carefully evaluating which previous knowledge can be reused for new
target domains is essential. 4.4.2. Multi-teacher Distillation
The total loss function for Lifelong distillation combines the CE task Multi-teacher knowledge distillation involves leveraging the
loss, the KL divergence distillation loss, and the knowledge retain loss. knowledge of multiple teacher models to guide the learning of a single
The knowledge retention loss prevents catastrophic forgetting of pre- student model (You et al., 2017). This approach can enhance the per-
vious tasks by regularizing the model’s parameters, such as Elastic formance of the student by combining diverse knowledge sources, but it
Weight Consolidation (EWC)(Kirkpatrick et al., 2017), which adds a also introduces challenges related to managing the variability among the
penalty term to restrict significant changes in important weights: teachers and ensuring effective knowledge aggregation. One major
∑ advantage of multi-teacher distillation is that it can improve general-
LRetain = λ Ω i (θ i (cid:0) θ i(cid:0)1 )2 (3.4.1.1) ization by exposing the student to a wider range of information, but
i aligning the teachers’ outputs and managing their contributions can be
Where: computationally expensive and complex (You et al., 2017).
θ i are the current model parameters. A typical multi-teacher distillation framework utilizes softened out-
θ i(cid:0)1 are the parameters learned from previous task. puts, intermediate feature-based distillation, or a combination of both to
Ω i are importance weights (indicating how crucial each parameter is guide the student model. In the approach proposed by Pham et al.
for the previous task). (2022b), multiple quantized teacher models with varying bit-widths
λ is a hyperparameter that controls the strength of this regulariza- collaborate to transfer knowledge. The contributions of each teacher
tion. are balanced according to its individual capacity. Each teacher’s inter-
The total loss for lifelong distillation can be expressed as: mediate layer outputs are fused into a shared feature map, weighted by
LKD = αLCE +βLKL +γLRetain (3.4.1.2) an importance factor that reflects the teacher’s influence on the collec-
tive knowledge. This collaborative learning approach encourages
Where α, β, γ are hyperparameters that balance the importance of the teachers with varying bit-widths to contribute optimally to the shared
CE task loss, the KL divergence distillation loss, and the knowledge knowledge, forming a robust representation that the student can mimic.
retain loss, respectively. The student learns from both the ensemble logits of the teachers (via
Soltoggio et al. (2024) introduced the Shared Experience Lifelong softened outputs and cross-entropy loss) and the shared feature maps at
Learning (ShELL) framework for distributed AI systems. ShELL enables selected layers using distance-based losses, such as attention loss or
AI agents to learn and share knowledge continuously at the network FitNet hint loss, to align the student’s intermediate representations with
edge, integrating knowledge-sharing mechanisms across agents. This those of the teachers.
approach allows for collaboration in decentralized environments and With single-teacher distillation, the student network learns from the
augments advanced AI models with lifelong learning capabilities. Ye and softened output of the teacher network. When there are multiple
Bors (2021)proposed Lifelong Twin GANs (LT-GANs), a dual-GAN sys- teachers, the student network learns from the average of the softened
tem for retaining knowledge across tasks. LT-GANs use a outputs of the multiple teacher networks. The loss function for multi-
Teacher-Assistant mechanism where roles alternate after learning new teacher knowledge distillation can be formulated as:
tasks, ensuring previously learned tasks are not forgotten. This method
15

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
1 ∑M knowledge transfer with adaptive target enhancement, significantly
LMTKD =αLCE +(1(cid:0) α)⋅ M LKL (3.4.2.1) improving student model performance, especially with dissimilar
i=1
teacher-student architectures. Dong et al. (2023) present DisWOT, a
Where: zero-training method for student architecture search in KD. DisWOT
LCE is the cross-entropy loss, or the standard task loss, which mini- uses similarity metrics and evolutionary algorithms to predict optimal
mizes the error between the student’s soft logits and the ground truth student architectures without training, reducing computational costs
labels. while achieving competitive performance.
LKL represents Kullback-Leibler divergence, or the distillation loss, In summary, recent advancements in knowledge distillation tech-
minimizes the difference between the soft logits of the i-th teacher and niques have broadened the scope of model compatibility, enabling
the student. effective knowledge transfer between heterogeneous architectures.
M is the number of teachers. These developments indicate a significant progression in the applica-
α is a balancing hyperparameter that controls the trade-off between bility of knowledge distillation, making it a viable approach for diverse
the cross-entropy loss and Kullback-Leibler divergence. deployment scenarios.
Wu et al. (2021a) proposed a framework for distilling knowledge
from multiple pre-trained language models (MT-BERT) by aligning their 6. Applications
hidden states. They introduced multi-teacher hidden loss and
multi-teacher distillation loss, which align intermediate representations Knowledge distillation techniques have found extensive applications
and weigh teacher contributions based on confidence, effectively across multiple domains, enabling the deployment of efficient and
handling inconsistencies in pre-trained language model feature spaces. compact models in areas where computational and memory resources
Liu et al. (2020) presented an adaptive framework that integrates are limited. From computer vision and natural language processing to
knowledge from multiple teacher networks by assigning weights based the medical field and autonomous systems, KD plays a critical role in
on teacher confidence. They introduced latent representations and a achieving high-performance models suitable for real-world constraints.
multi-group hint strategy to capture hidden representations and facili- This section explores the use of KD in various applications, discussing
tate intermediate-level knowledge transfer, ensuring more effective how different techniques address specific needs in each field.
learning from reliable teachers. To aid in understanding the trade-offs among KD techniques, Table 4
To illustrate a real-world application, the study by Tang and Liu provides a comparative overview based on key metrics such as accuracy
(2024)uses a multi-teacher KD approach to detect financial fraud. The boost, resource efficiency, computational complexity, and ideal use
goal is to improve detection accuracy, inference speed, and generaliza- cases. This table serves as a quick reference for selecting the appropriate
tion ability when dealing with imbalanced datasets and industry-specific KD method depending on the target application’s requirements and
challenges. The process starts by training multiple teacher models, each constraints, helping to weigh the strengths and limitations of each
specialized in detecting fraud within a specific industry. These models approach.
are then used to transfer their knowledge to a single, more compact
student model. This distillation process leverages both hard targets (true 6.1. KD in vision and language
labels) and soft targets (probability distributions from the teachers) to
guide the student’s learning. By incorporating insights from multiple Knowledge distillation has been applied across various domains,
expert models, the student model becomes capable of detecting fraud including vision and language (VL), where large models are compressed
across different industries while maintaining a smaller size for faster into smaller, efficient ones while maintaining high performance. VL
inference. This distributed approach addresses the limitations of tradi- models integrate visual and textual data for tasks like image captioning
tional methods that struggle with industry-specific variations and large, and visual question answering but are often computationally intensive,
complex models. making KD crucial for their deployment in resource-limited settings.
In summary, multi-teacher distillation improves model robustness by Fang et al. (2021a)propose DistillVLM to address misalignment in
integrating diverse knowledge but can be computationally intensive due visual tokens between teacher and student models. By aligning visual
to the need to manage multiple teacher contributions. The process of and linguistic features, they achieve significant compression without
multi-teacher distillation typically involves pretraining multiple teacher sacrificing accuracy, making the model suitable for resource-constrained
models and aggregating their outputs during the training of the student tasks. In another work, Fang et al. (2021b) introduce SEED, a
model (Amirkhani et al., 2021). These aggregated outputs serve as soft self-supervised distillation method that improves visual representations
labels, which are used alongside the true labels to train the student by guiding the student model during pre-training. SEED enhances per-
model. formance without labeled data, proving effective in various tasks with
limited data. Fang and Yang (2023) discuss cross-modal KD in VL,
5. Teacher–student architecture variations highlighting techniques like shared embeddings and contrastive
learning to improve tasks such as visual question answering and
Knowledge distillation involves transferring knowledge from a large, language-driven image generation. Liu et al. (2022c) apply
complex teacher model to a smaller, more efficient student model cross-domain KD for deepfake detection, focusing on localized in-
(Hilton et al., 2015). This technique traditionally assumes that the consistencies in spatial and frequency domains. Their approach im-
teacher and student share similar architectures. However, in real-world proves detection accuracy across various deepfake styles. Yun et al.
applications, models with different architectures may need to be (2021)defend KD for task incremental learning in 3D object detection,
deployed due to varying computational constraints or specific applica- using KD with a Bayesian approach to prevent catastrophic forgetting,
tion requirements. ensuring high accuracy across multiple detection tasks.
Liu et al. (2022b)propose a method for cross-architecture knowledge In the work on enhancing Alexa’s natural language understanding
distillation (KD) using two projectors, partially cross-attention (PCA) (NLU) system, FitzGerald et al. (2022)at Amazon explore KD to train
and group-wise linear (GL), to align teacher-student feature represen- multi-billion-parameter encoders for language tasks. The Alexa Teacher
tations. This improves student model performance, even with simpler Model demonstrates the potential of KD in efficiently handling
architectures, outperforming traditional KD across multiple bench- large-scale, real-world NLU applications by pretraining and distilling
marks. Hao et al. (2024)introduce the One-for-All (OFA-KD) framework high-capacity teacher models into smaller, deployable models. This
to transfer knowledge between heterogeneous architectures by projec- approach enables Alexa to provide fast and accurate responses in
ting intermediate features into a unified latent space. OFA-KD enhances resource-constrained voice-activated devices.
16

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
Table 4
Comparative analysis of knowledge distillation techniques based on different metrics.
Technique Accuracy Resource Computational Ideal Use Cases Key Trade-Offs
Boost Efficiency Complexity
Response-Based KD Moderate High Low General-purpose, NLP tasks Limited by capacity gap between teacher
and student
Feature-Based KD High Moderate Moderate Vision, complex visual tasks May struggle with feature alignment across
architectures
Relation-Based KD High Low High Graph-based applications High computational cost for graph-
structured data
Offline Distillation Moderate- High Moderate Pre-trained models, resource-limited devices Inflexible to new data patterns
High
Online Distillation High Moderate High Real-time learning, continual updates High computational cost due to
simultaneous training
Self-Distillation Moderate High Low Mobile deployment Risk of reinforcing incorrect predictions
Cross-Modal Moderate Moderate High Multi-modal tasks (e.g., image-text, audio- Requires advanced alignment for disparate
Distillation video) feature spaces
Graph-Based High Low High Node classification, social network analysis Requires complex loss functions for graph
Distillation integrity
Attention-Based Moderate- Moderate High Vision, NLP with interpretability needs High computational demands for attention
Distillation High mechanism
Data-Free KD Moderate Moderate High Privacy-sensitive environments Quality of synthetic data impacts
effectiveness
Adversarial KD Moderate- Low High Security-sensitive applications Balancing adversarial and distillation loss
High
Quantized KD Low- Very High Moderate IoT, edge devices Accuracy may decrease with reduced bit
Moderate precision
NAS-Based High High Very High Edge devices, compact model deployment Initial search is computationally demanding
Distillation
Lifelong Distillation Moderate Moderate High Dynamic applications (e.g., language Must carefully select reusable prior
learning, autonomous systems) knowledge for new domains
Multi-Teacher High Moderate High Multi-modal, diverse information tasks Alignment of multiple teachers can be
Distillation resource-intensive
McDonald et al. (2024) document the use of KD by Mistral AI to accurate organ-specific image processing in diagnostics.
reduce hallucination rates in LLMs, a common challenge in generative Zou et al. (2021)introduce Coco DistillNet for pathological gastric
models. By distilling knowledge from teacher to student models, they cancer segmentation. This approach improves segmentation accuracy by
improve the accuracy and reliability of AI-generated content across transferring knowledge from a teacher model to a lightweight student
benchmarks like the MMLU. This distillation approach results in smaller, model through cross-layer correlation distillation. It ensures better vi-
more efficient LLMs that are better suited for deployment in environ- sual feature understanding, making it suitable for clinical settings with
ments requiring controlled output reliability. limited computational resources. Li et al. (2022b) combine
In 3D computer vision, Chan et al. (2022)leverage KD to enhance the self-supervised learning with self-knowledge distillation for COVID-19
performance of geometry-aware GANs for applications requiring real- detection from chest X-rays. This method helps overcome the scarcity
istic 3D image synthesis. Their work on Efficient Geometry-Aware 3D of labeled data by allowing the student model to refine its representa-
GANs focuses on refining spatial and feature representation, reducing tions, achieving high detection accuracy with minimal labeled data.
the computational cost of rendering realistic 3D visuals. KD facilitates Gordienko et al. (2023)propose an ensemble KD framework for edge
model compression, allowing high-quality 3D generation to be deployed intelligence in medical applications. Multiple teacher models distill
on resource-limited devices, making it suitable for applications like knowledge into a compact student model, enhancing its performance
AR/VR. and generalization. This framework is particularly useful for real-time,
low-latency tasks on edge devices like portable diagnostic tools.
6.2. KD in medical applications
6.3. KD in Other Applications
Knowledge distillation has become crucial in the medical field,
where efficient and deployable models are essential for tasks like disease Knowledge distillation extends beyond traditional fields like com-
detection, medical image segmentation, and patient monitoring. KD puter vision and NLP, proving useful in areas such as autonomous sys-
allows for the creation of lightweight models that maintain the perfor- tems, neural architecture search (NAS), and industrial anomaly
mance of larger ones, making it ideal for resource-constrained medical detection. KD helps reduce model complexity, enhance performance,
environments. and enable deployment in resource-limited environments.
In medical imaging, Generative Adversarial Networks (GANs) play a Kargin and Petrenko (2023) introduce KD for autonomous un-
transformative role in handling various tasks, including segmentation, manned systems like UAVs and AGVs. Their method improves
reconstruction, detection, classification, augmentation, registration, and decision-making by distilling knowledge about environmental factors
image synthesis (Islam et al., 2024). With the capacity to generate and event streams, allowing lightweight models to function efficiently in
high-quality synthetic images even with limited datasets, GANs have dynamic environments with real-time constraints. Trofimov et al.
notably improved diagnostic precision and image quality enhancement. (2023)combine multi-fidelity NAS with KD to accelerate the search for
According to Alamir and Alghamdi (2023), GANs contribute to a new optimal architectures. By leveraging teacher models for guidance, KD
level of diagnostic capability by producing realistic medical images that reduces computational costs and improves the efficiency of NAS,
aid in training and validating other diagnostic models. GANs are espe- discovering high-performing architectures faster. Rakhmonov et al.
cially impactful in the segmentation of critical organs; Makhlouf et al. (2023) propose an Extensive Knowledge Distillation (EKD) model for
(2023)highlight that brain, chest, breast, and lung images are among real-time anomaly detection in industrial settings. EKD transfers
the most frequently segmented areas, reflecting the high demand for knowledge from complex teacher models to lightweight students,
17

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
enabling accurate, real-time fault detection, enhancing safety and Chawla, A., Yin, H., Molchanov, P., & Alvarez, J. (2021). Data-free knowledge distillation
operational efficiency. for object detection | IEEE conference publication | IEEE Xplore. Data-Free Knowledge
Distillation for Object Detection. https://ieeexplore.ieee.org/document/9423376.
Mora et al. (2022) explore the integration of KD into federated Chen, D., Mei, J. P., Wang, C., Feng, Y., & Chen, C. (2020). Online knowledge distillation
learning (FL) systems. In federated learning, data privacy and commu- with diverse peers. arXiv.org. https://arxiv.org/abs/1912.00350.
nication efficiency are key challenges. The authors present KD as a so- Chen, G., Choi, W., Yu, X., Han, T., & Chandraker, M. (2017). Learning efficient object
detection models with knowledge distillation. Advances in Neural Information
lution that allows local models to distill their knowledge into a central
Processing Systems, 30.
student model without sharing raw data. This not only improves the Chen, W. C., & Chu, W. T. (2023). Sssd: Self-supervised self distillation. In Proceedings of
generalization of the global model but also reduces communication the IEEE/CVF Winter Conference on Applications of Computer Vision (pp. 2770–2777).
Dong, P., Li, L., & Wei, Z. (2023). Diswot: Student architecture search for distillation
overhead in FL. Their framework enhances model performance in
without training. In Proceedings of the IEEE/CVF Conference on Computer Vision and
federated settings, particularly in environments with non-IID data and Pattern Recognition (pp. 11898–11908).
limited communication resources, making it a valuable technique for Dong, S., Hong, X., Tao, X., Chang, X., Wei, X., & Gong, Y. (2021). Few-shot class-
incremental learning via relation knowledge distillation. In Proceedings of the AAAI
distributed machine learning tasks.
Conference on Artificial Intelligence. https://ojs.aaai.org/index.php/AAAI/article/vie
w/16213.
7. Conclusion Elsken, T., Staffler, B., Metzen, J. H., & Hutter, F. (2020). Meta-learning of neural
architectures for few-shot learning. In Proceedings of the IEEE/CVF conference on
computer vision and pattern recognition (pp. 12365–12375).
Knowledge distillation has revolutionized model compression and Fang, G., Mo, K., Wang, X., Song, J., Bei, S., Zhang, H., & Song, M. (2022). Up to 100 x
knowledge transfer in machine learning since its introduction by Hinton faster data-free knowledge distillation. arXiv.org. https://arxiv.org/abs/2112.06253.
Fang, Z., Wang, J., Hu, X., Wang, L., Yang, Y., & Liu, Z. (2021a). Compressing visual-
et al. (2015). This paper has provided a comprehensive overview of its
linguistic model via knowledge distillation. In Proceedings of the IEEE/CVF
evolution, exploring various types of knowledge transfer (response-- International Conference on Computer Vision (pp. 1428–1438).
based, feature-based, and relation-based), distillation schemes (offline, Fang, Z., Wang, J., Wang, L., Zhang, L., Yang, Y., & Liu, Z. (2021b). Seed: Self-supervised
online, and self-distillation), and advanced algorithms. The advance- distillation for visual representation. arXiv preprint. arXiv:2101.04731.
Fang, Z., & Yang, Y. (2023). Knowledge distillation across vision and language.
ments in knowledge distillation have significantly improved the Advancements in Knowledge Distillation: Towards New Horizons of Intelligent Systems
deployment of efficient models on resource-constrained devices while (pp. 65–94). Cham: Springer International Publishing.
maintaining high performance, enabling sophisticated AI capabilities in FitzGerald, J., Ananthakrishnan, S., Arkoudas, K., Bernardi, D., Bhagia, A., Delli Bovi, C.,
… Natarajan, P. (2022). Alexa teacher model: Pretraining and distilling multi-
diverse real-world scenarios. As the field continues to evolve, future
billion-parameter encoders for natural language understanding systems. In
research may focus on improving existing approaches and exploring Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data
novel applications. Knowledge distillation remains crucial in bridging Mining (pp. 2893–2902).
Ge, Y., Zhang, X., Choi, C. L., Cheung, K. C., Zhao, P., Zhu, F., … Li, H. (2021). Self-
the gap between complex state-of-the-art AI models and practical
distillation with batch knowledge ensembling improves imagenet classification.
deployment constraints, promising continued innovation in the field of arXiv preprint. arXiv:2104.13298.
artificial intelligence. Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., …
Bengio, Y. (2014). Generative adversarial nets. Advances in Neural Information
Processing Systems, 27.
Declaration of competing interest Gordienko, Y., Shulha, M., Kochura, Y., Rokovyi, O., Alienin, O., Taran, V., & Stirenko, S.
(2023). Ensemble knowledge distillation for edge intelligence in medical
applications. Advancements in Knowledge Distillation: Towards New Horizons of
There is not conflict of interest. Intelligent Systems (pp. 135–168). Cham: Springer International Publishing.
Gou, J., Yu, B., Maybank, S. J., & Tao, D. (2021). Knowledge distillation: A survey.
International Journal of Computer Vision, 129(6), 1789–1819.
Funding
Gou, J., Chen, Y., Yu, B., Liu, J., Du, L., Wan, S., & Yi, Z. (2024). Reciprocal teacher-
student learning via forward and feedback knowledge distillation. IEEE Transactions
This work was supported by the Applied Research and Technology on Multimedia, 26, 7901–7916. https://doi.org/10.1109/TMM.2024.3372833. IEEE
Transactions on Multimedia.
Partnership grants (ARTPs) funded by the Natural Sciences and Engi-
Gowda, S. N., Hao, X., Li, G., Gowda, S. N., Jin, X., & Sevilla-Lara, L. (2024). Watt for
neering Research Council of Canada (NSERC) under Grant CCB21-2021- what: Rethinking deep learning’s energy-performance relationship (No. arXiv:
00266 and by BFI Energy Group Inc. 2310.06522). arXiv. http://arxiv.org/abs/2310.06522.
Guo, Ziyao, Yan, Haonan, Li, Hui, & Lin, Xiaodong (2023). Class attention transfer based
knowledge distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision
Data availability and Pattern Recognition (pp. 11868–11877).
Gupta, S., Hoffman, J., & Malik, J. (2016). Cross modal distillation for supervision
transfer. In Proceedings of the IEEE conference on computer vision and pattern recognition
No data was used for the research described in the article.
(pp. 2827–2836).
Hao, Z., Guo, J., Han, K., Tang, Y., Hu, H., Wang, Y., & Xu, C. (2024). One-for-all: Bridge
References the gap between heterogeneous architectures in knowledge distillation. Advances in
Neural Information Processing Systems, 36.
He, H., Wang, J., Zhang, Z., & Wu, F. (2022). Compressing deep graph neural networks
Ahmad, S., Ullah, Z., & Gwak, J. (2024). Multi-Teacher Cross-Modal Distillation with
via adversarial knowledge distillation. In Proceedings of the 28th ACM SIGKDD
Cooperative Deep Supervision Fusion Learning for Unimodal Segmentation, 297. Conference on Knowledge Discovery and Data Mining (pp. 534–544).
Knowledge-Based Systems, Article 111854.
Heo, B., Kim, J., Yun, S., Park, H., Kwak, N., & Choi, J. Y. (2019). A comprehensive
Amirkhani, A., Khosravian, A., Masih-Tehrani, M., & Kashiani, H. (2021). Robust
overhaul of feature distillation. In Proceedings of the IEEE/CVF international conference
s 1 e 1 m 90 an 4 t 9 i – c 1 s 1 e 9 g 0 m 6 e 6 n . t h at t i t o p n s: / w / i d t o h i . m or u g l / ti 1 -t 0 e . a 1 c 1 h 0 e 9 r / k A n C o C w E l S e S d . g 2 e 0 2 d 1 is .3 ti 1 ll 0 a 7 ti 8 o 4 n 1 . IEEE Access, 9, on computer vision (pp. 1921–1930).
Higuchi, H., Suzuki, S., & Shouno, H. (2022). Adversarial training with knowledge
Anil, R., Pereyra, G., Passos, A., Ormandi, R., Dahl, G. E., & Hinton, G. E. (2018). Large
distillation considering intermediate representations in CNNs. In International
scale distributed neural network training through online distillation. arXiv preprint. Conference on Neural Information Processing (pp. 683–691). Singapore: Springer
arXiv:1804.03235.
Nature Singapore.
Bai, J., Yin, C., Zhang, J., Wang, Y., Dong, Y., Rong, W., & Xiong, Z. (2022). Adversarial
Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network.
knowledge distillation based biomedical factoid question answering. IEEE/ACM
arXiv preprint. arXiv:1503.02531.
Transactions on Computational Biology and Bioinformatics, 1. https://doi.org/
Hong, X., Guan, S. U., Man, K. L., & Wong, P. W. (2020). Lifelong machine learning
10.1109/tcbb.2022.3161032
architecture for classification. Symmetry, 12(5), 852.
Boo, Y., Shin, S., Choi, J., & Sung, W. (2021). Stochastic precision ensemble: Self-
Huo, F., Xu, W., Guo, J., Wang, H., & Guo, S. (2024). C2KD: Bridging the modality gap
knowledge distillation for quantized deep neural networks. In , 35. Proceedings of the
Caru A a A na A , I R C ., o B nf u e c r i e l n u c ǎ e , o C n ., A & r N tifi ic c u ia le l s I c n u te - l M lig iz en il c , e A ( . p ( p 2 . 0 0 6 6 7 ) 9 . 4 M –6 o 8 d 0 e 2 l ) c . ompression. In Proceedings Islam f C o , o r S m . c , p r A u o t s , e s A r - m . V , o i A s d i z o a i n z l , k a M n n o . d w T P . l , a e N t d te g a r e b n i d l R , i s H e t c i . o l R l g a n . t , i i t J o i i o n m n . , I ( n J p . p P R . r . o 1 , c 6 K e 0 e a 0 d b i 6 i n r – g , 1 s M 6 o 0 . f M 1 th 5 . e , ) . M IE r E id E h /C a, V M F . C F o . n , f … er e S n h c i e n o , n J
of the 12th ACM SIGKDD international conference on Knowledge discovery and data
(2024). Generative adversarial networks (GANs) in medical imaging: advancements,
mining (pp. 535–541).
Chan, E. R., Lin, C. Z., Chan, M. A., Nagano, K., Pan, B., De Mello, S., … Wetzstein, G. applications and challenges. IEEE Access.
Ji, M., Heo, B., & Park, S. (2021a). Show, Attend and Distill: Knowledge distillation via
(2022). Efficient geometry-aware 3d generative adversarial networks. In Proceedings
attention-based feature matching. Cornell University. arXiv.
of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp.
16123–16133).
18

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
Ji, M., Shin, S., Hwang, S., Park, G., & Moon, I. C. (2021b). Refine myself by teaching Makhlouf, A., Maayah, M., Abughanam, N., et al. (2023). The use of generative
myself: Feature refinement via self-knowledge distillation. In Proceedings of the IEEE/ adversarial networks in medical image augmentation. Neural Comput & Applic, 35,
CVF conference on computer vision and pattern recognition (pp. 10664–10673). 24055–24068. https://doi.org/10.1007/s00521-023-09100-z
Kargin, A., & Petrenko, T. (2023). Knowledge distillation for autonomous intelligent Mirzadeh, S. I., Farajtabar, M., Li, A., Levine, N., Matsukawa, A., & Ghasemzadeh, H.
unmanned system. Advancements in Knowledge Distillation: Towards New Horizons of (2020, April). Improved knowledge distillation via teacher assistant. In Proceedings of
Intelligent Systems (pp. 193–230). Cham: Springer International Publishing. the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 5191-5198).
Kim, H., Kwak, T. Y., Chang, H., Kim, S. W., & Kim, I. (2023). RCKD: response-based https://doi.org/10.1609/aaai.v34i04.5963.
cross-task knowledge distillation for pathological image analysis. Bioengineering Mora, A., Tenison, I., Bellavista, P., & Rish, I. (2022). Knowledge distillation for
(Basel), 10(11), 1279-. https://doi.org/10.3390/bioengineering10111279 federated learning: a practical guide. arXiv preprint. arXiv:2211.04742.
Kim, J., Bhalgat, Y., Lee, J., Patel, C., & Kwak, N. (2019). Qkd: Quantization-aware Park, W., Kim, D., Lu, Y., & Cho, M. (2019). Relational knowledge distillation. In
knowledge distillation. arXiv preprint. arXiv:1911.12491. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition
Kim, K., Ji, B., Yoon, D., & Hwang, S. (2021). Self-knowledge distillation with (CVPR) (pp. 3967–3976).
progressive refinement of targets. In 2021 IEEE/CVF International Conference on Passban, P., Wu, Y., Rezagholizadeh, M., & Liu, Q. (2020). ALP-KD: Attention-Based Layer
Computer Vision (ICCV) (pp. 6547–6556). https://doi.org/10.1109/ Projection for Knowledge Distillation. Cornell University. https://doi.org/10.48550/
ICCV48922.2021.00650 arxiv.2012.14022. arXiv.
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., … Peng, B., Jin, X., Liu, J., Li, D., Wu, Y., Liu, Y., Zhou, S., & Zhang, Z. (2019). Correlation
Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. congruence for knowledge distillation. In Proceedings of the IEEE/CVF International
Proceedings of the National Academy of Sciences, 114(13), 3521–3526. Conference on Computer Vision (ICCV) (pp. 5007–5016).
Lan, X., Zhu, X., & Gong, S. (2018). Knowledge distillation by on-the-fly native ensemble. Pham, C., Hoang, T., & Do, T. (2022b). Collaborative multi-teacher knowledge
Advances in Neural Information Processing Systems, 31. distillation for learning low bit-width deep neural networks. arXiv.org. https://arxiv.
Lee, H., An, S., Kim, M., & Hwang, S. J. (2023). Meta-prediction model for distillation- org/abs/2210.16103.
aware NAS on unseen datasets. arXiv.org. https://arxiv.org/abs/2305.16948. Pham, M., Cho, M., Joshi, A., & Hegde, C. (2022a). Revisiting self-distillation. arXiv
Lee, H., Park, Y., Seo, H., & Kang, M. (2023b). Self-knowledge distillation via dropout. preprint. arXiv:2206.08491.
Computer Vision and Image Understanding, 233, Article 103720. Rakhmonov, A. A. U., Subramanian, B., Olimov, B., & Kim, J. (2023). Extensive
Lee, S., Kim, S., Kim, S. S., & Seo, K. (2022). Similarity-based adversarial knowledge knowledge distillation model: An end-to-end effective anomaly detection model for
distillation using graph convolutional neural network. Electronics Letters, 58(16), real-time industrial applications. IEEE Access.
606–608. Sarkar, P., & Etemad, A. (2022). XKD: cross-modal knowledge distillation with domain
Li, C., Peng, J., Yuan, L., Wang, G., Liang, X., Lin, L., & Chang, X. (2020). Block-wisely alignment for video representation learning. Cornell University. https://doi.org/
supervised neural architecture search with knowledge distillation. In 2020 IEEE/CVF 10.48550/arxiv.2211.13929. arXiv.
Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 1986–1995). Schmid, F., Koutini, K., & Widmer, G. (2022). Efficient large-scale audio tagging via
https://doi.org/10.1109/CVPR42600.2020.00206 transformer-to-CNN knowledge distillation. Cornell University. https://doi.org/
Li, Y., Nie, X., Diao, W., & Zheng, S. (2021). Lifelong CycleGAN for continual multi-task 10.48550/arxiv.2211.04772. arXiv.
image restoration. Pattern Recognition Letters, 153, 183–189. https://doi.org/ Sepahvand, M., Abdali-Mohammadi, F., & Taherkordi, A. (2022). Teacher–student
10.1016/j.patrec.2021.12.010 knowledge distillation based on decomposed deep feature representation for
Li, Z., Ye, J., Song, M., Huang, Y., & Pan, Z. (2022a). Online knowledge distillation for intelligent mobile applications. Expert Systems with Applications, 202, Article 117474.
efficient pose estimation. arXiv.org. https://arxiv.org/abs/2108.02092. Soltoggio, A., Ben-Iwhiwhu, E., Braverman, V. et al. A collective AI via lifelong learning
Li, G., Togo, R., Ogawa, T., & Haseyama, M. (2022b). Self-knowledge distillation based and sharing at the edge. Nature Machine Intelligence 6, 251–264 (2024).
self-supervised learning for covid-19 detection from chest x-ray images. In ICASSP Song, L., Gong, X., Zhou, H., Chen, J., Zhang, Q., Doermann, D., & Yuan, J. (2023).
2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing Exploring the knowledge transferred by response-based teacher-student distillation.
(ICASSP) (pp. 1371–1375). IEEE. In Proceedings of the 31st ACM International Conference on Multimedia (pp.
Li, W., Wang, J., Ren, T., Li, F., Zhang, J., & Wu, Z. (2022c). Learning accurate, speedy, 2704–2713).
lightweight CNNs via instance-specific multi-teacher knowledge distillation for Srinivasagan, G., Deisher, M., & Georges, M. (2023). Compression of End-to-End Non-
distracted driver posture identification. IEEE Transactions on Intelligent Transportation Autoregressive Image-to-Speech System for Low-Resourced Devices. Cornell University.
Systems, 23(10), 17922–17935. https://doi.org/10.1109/TITS.2022.3161986. IEEE https://doi.org/10.48550/arxiv.2312.00174. arXiv.
Transactions on Intelligent Transportation Systems. Tang, Y., & Liu, Z. (2024). A Distributed knowledge distillation framework for financial
Li, Z., Li, X., Yang, L., Song, R., Yang, J., & Pan, Z. (2024a). Dual teachers for self- fraud detection based on transformer. IEEE Access, 12, 62899–62911. https://doi.
knowledge distillation. Pattern Recognition, 151, Article 110422. org/10.1109/ACCESS.2024.3387841
Li, Y., Li, P., Yan, D., Liu, Y., & Liu, Z. (2024b). Deep knowledge distillation: A self- Trivedi, A., Udagawa, T., Merler, M., Panda, R., El-Kurdi, Y., & Bhattacharjee, B. (2023).
mutual learning framework for traffic prediction. Expert Systems With Applications, Neural architecture search for effective teacher-student knowledge transfer in
252, Article 124138. https://doi.org/10.1016/j.eswa.2024.124138 0language models. arXiv.org. https://arxiv.org/abs/2303.09639.
Liang, P., Zhang, W., Wang, J., & Guo, Y. (2024). Neighbor self-knowledge distillation. Trofimov, I., Klyuchnikov, N., Salnikov, M., Filippov, A., & Burnaev, E. (2023). Multi-
Information Sciences, 654, Article 119859. fidelity neural architecture search with knowledge distillation. IEEE Access, 11,
Lin, R., Lv, X., Hu, H., Ling, L., Yu, Z., & Zhang, D. (2023). Dual-stage ensemble approach 59217–59225.
using online knowledge distillation for forecasting carbon emissions in the electric Tung, F., & Mori, G. (2019). Similarity-preserving knowledge distillation. In Proceedings
power industry. Data Science and Management, 6(4), 227–238. https://doi.org/ of the IEEE/CVF International Conference on Computer Vision (ICCV) (pp. 1365–1374).
10.1016/j.dsm.2023.09.001 Wang, C., Zhou, S., Yu, K., Chen, D., Li, B., Feng, Y., & Chen, C. (2022a). Collaborative
Liu, C., Chen, L. C., Schroff, F., Adam, H., Hua, W., Yuille, A. L., & Fei-Fei, L. (2019a). knowledge distillation for heterogeneous information network embedding. In
Auto-deeplab: Hierarchical neural architecture search for semantic image Proceedings of the ACM Web Conference 2022. https://doi.org/10.1145/
segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern 3485447.3512209
recognition (pp. 82–92). Wang, W., Liu, F., Liao, W., & Xiao, L. (2023). Cross-modal graph knowledge
Liu, J., Zheng, T., & Hao, Q. (2022a). HIRE: Distilling High-Order Relational Knowledge representation and distillation learning for land cover classification. IEEE
from Heterogeneous Graph Neural Networks. Cornell University. https://doi.org/ Transactions on Geoscience and Remote Sensing, 61, 1. https://doi.org/10.1109/
10.48550/arxiv.2207.11887. arXiv. TGRS.2023.3307604
Liu, J., Zheng, T., Zhang, G., & Hao, Q. (2023). Graph-based knowledge distillation: A Wang, X., Zhang, R., Sun, Y., & Qi, J. (2018). Kdgan: Knowledge distillation with
survey and experimental evaluation. arXiv preprint. arXiv:2302.14643. generative adversarial networks. Advances in neural information processing systems,
Liu, H., Simonyan, K., & Yang, Y. (2019b). Darts: Differentiable architecture search. arXiv 31.
preprint. arXiv:1806.09055. Wang, Y. H., Lin, C. Y., Thaipisutikul, T., & Shih, T. K. (2022b). Single-head lifelong
Liu, Y., Zhang, W., & Wang, J. (2020). Adaptive multi-teacher multi-level knowledge learning based on distilling knowledge. IEEE Access, 10, 35469–35478.
distillation. Neurocomputing, 415, 106–113. https://doi.org/10.1016/j. Wang, N., Deng, Y., Feng, W., Yin, J., & Ng, S. K. (2024). Data-free federated class
neucom.2020.07.048 incremental learning with diffusion-based generative memory (no. arXiv:
Liu, Y., Cao, J., Li, B., Hu, W., Ding, J., & Li, L. (2022b). Cross-architecture knowledge 2405.17457). arXiv. https://doi.org/10.48550/arXiv.2405.17457
distillation. In Proceedings of the Asian conference on computer vision (pp. 3396–3411). Wu, C., Wu, F., & Huang, Y. (2021a). One teacher is enough? pre-trained language model
Liu, Z., Wang, H., & Wang, S. (2022c). Cross-domain local characteristic enhanced distillation from multiple teachers. arXiv.org. https://arxiv.org/abs/2106.01023.
deepfake video detection. In Proceedings of the Asian Conference on Computer Vision Wu, Y., Rezagholizadeh, M., Ghaddar, A., Haidar, M. A., & Ghodsi, A. (2021b). Universal-
(pp. 3412–3429). KD: attention-based output-grounded intermediate layer knowledge distillation. In
Lopes, R. G., Fenu, S., & Starner, T. (2017). Data-free knowledge distillation for deep Proceedings of the 2021 Conference on Empirical Methods in Natural Language
neural networks. arXiv preprint. arXiv:1710.07535. Processing. https://doi.org/10.18653/v1/2021.emnlp-main.603
Lo´pez-Cifuentes, A., Escudero-Vin˜olo, M., Besco´s, J., & Miguel, J. C. S. (2023). Attention- Xia, W., Li, X., Deng, A., Xiong, H., Dou, D., & Hu, D. (2023). Robust Cross-Modal
based knowledge distillation in scene recognition: The impact of a DCT-Driven loss. knowledge distillation for unconstrained videos. Cornell University. https://doi.org/
IEEE Transactions on Circuits and Systems for Video Technology, 33(9), 4769–4783. 10.48550/arxiv.2304.07775. arXiv.
https://doi.org/10.1109/tcsvt.2023.3250031 Xiang, Q., Zhang, M., Shang, Y., Wu, J., Yan, Y., & Nie, L. (2024). DKDM: Data-Free
McDonald, D., Papadopoulos, R., & Benningfield, L. (2024). Reducing llm hallucination Knowledge Distillation for Diffusion Models with Any Architecture. arXiv.org. htt
using knowledge distillation: A case study with mistral large and mmlu benchmark. ps://arxiv.org/abs/2409.03550.
Authorea Preprints. Xue, Z., Gao, Z., Ren, S., & Zhao, H. (2022). The modality focusing hypothesis: towards
understanding crossmodal knowledge distillation. arXiv (Cornell University). htt
ps://doi.org/10.48550/arxiv.2206.06487.
19

A. Moslemi et al. M a c h i n e L e a r n i n g w i t h A p p l i c a t i o n s 18 (2024) 100605
Yang, C., Zhou, H., An, Z., Jiang, X., Xu, Y., & Zhang, Q. (2022). Cross-image relational Yun, P., Liu, Y., & Liu, M. (2021). In defense of knowledge distillation for task
knowledge distillation for semantic segmentation. arXiv.org. https://arxiv. incremental learning and its application in 3D object detection. IEEE Robotics and
org/abs/2204.06986. Automation Letters, 6(2), 2012–2019.
Yang, C., Yu, X., An, Z., & Xu, Y. (2023a). Categories of response-based, feature-based, Zagoruyko, S., & Komodakis, N. (2016). Paying more attention to attention: Improving
and relation-based knowledge distillation. Advancements in Knowledge Distillation: the performance of convolutional neural networks via attention transfer. arXiv
Towards New Horizons of Intelligent Systems (pp. 1–32). Cham: Springer International preprint. arXiv:1612.03928.
Publishing. Zhai, M., Chen, L., & Mori, G. (2021). Hyper-lifelonggan: Scalable lifelong learning for
Yang, Z., Zeng, A., Li, Z., Zhang, T., Yuan, C., & Li, Y. (2023b). From knowledge image conditioned generation. In Proceedings of the IEEE/CVF Conference on Computer
distillation to self-knowledge distillation: A unified approach with normalized loss Vision and Pattern Recognition (pp. 2246–2255).
and customized soft labels. In Proceedings of the IEEE/CVF International Conference on Zhang, C., & Peng, Y. (2018). Better and faster: knowledge transfer from multiple self-
Computer Vision (pp. 17185–17194). supervised learning tasks via graph distillation for video classification. arXiv preprint.
Yang, L., & Xu, K. (2021). Cross modality knowledge distillation for multi-modal aerial arXiv:1804.10069.
view object classification. In 2021 IEEE/CVF Conference on Computer Vision and Zhang, L., Song, J., Gao, A., Chen, J., Bao, C., & Ma, K. (2019). Be your own teacher:
Pattern Recognition Workshops (CVPRW) (pp. 382–387). https://doi.org/10.1109/ Improve the performance of convolutional neural networks via self distillation. In
cvprw53098.2021.00048 Proceedings of the IEEE/CVF international conference on computer vision (pp.
Yang, Y., Qiu, J., Song, M., Tao, D., & Wang, X. (2020). Distilling Knowledge from Graph 3713–3722).
Convolutional Networks. Cornell University. https://doi.org/10.48550/ Zhang, L., Bao, C., & Ma, K. (2021). Self-distillation: Towards efficient and compact
arxiv.2003.10477. arXiv. neural networks. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(8),
Yim, J., Joo, D., Bae, J., & Kim, J. (2017). A gift from knowledge distillation: Fast 4388–4403.
optimization, network minimization and transfer learning. In Proceedings of the IEEE Zhang, Y., Xiang, T., Hospedales, T. M., & Lu, H. (2018). Deep mutual learning. In
Conference on Computer Vision and Pattern Recognition (pp. 4133–4141). Proceedings of the IEEE conference on computer vision and pattern recognition (pp.
Yin, G., Wang, W., Yuan, Z., Han, C., Ji, W., Sun, S., & Wang, C. (2022). Content-Variant 4320–4328).
Reference Image Quality Assessment Via Knowledge Distillation. Cornell University. Zhao, Z., Lyu, J., Chu, Y., Liu, K., Cao, D., Wu, C., Qin, L., & Qin, S. (2023). Toward
https://doi.org/10.48550/arxiv.2202.13123. arXiv. generalizable robot vision guidance in real-world operational manufacturing
Ye, F., & Bors, A. G. (2021). Lifelong twin generative adversarial networks. In 2021 IEEE factories: A semi-supervised knowledge distillation approach. Robotics and Computer-
International Conference on Image Processing (ICIP) (pp. 1289–1293). IEEE. https:// Integrated Manufacturing, 86, Article 102639. https://doi.org/10.1016/j.
doi.org/10.1109/ICIP42928.2021.9506116. rcim.2023.102639
You, S., Xu, C., Xu, C., & Tao, D. (2017). Learning from multiple teacher networks. In Zhao, K., & Zhao, M. (2024). Self-supervised quantization-aware knowledge distillation.
Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery arXiv preprint. arXiv:2403.11106.
and Data Mining (pp. 1285–1294). https://doi.org/10.1145/3097983.3098135 Zhu, Z., Hong, J., & Zhou, J. (2021). Data-free knowledge distillation for heterogeneous
Yue, J., Fang, L., Rahmani, H., & Ghamisi, P. (2022). Self-supervised learning with federated learning. arXiv.org. https://arxiv.org/abs/2105.10056.
adaptive distillation for hyperspectral image classification. IEEE Transactions on Zou, W., Qi, X., Wu, Z., Wang, Z., Sun, M., & Shan, C. (2021). Coco distillnet: a cross-
Geoscience and Remote Sensing, 60, 1–13. https://doi.org/10.1109/ layer correlation distillation network for pathological gastric cancer segmentation.
TGRS.2021.3057768 In 2021 IEEE International Conference on Bioinformatics and Biomedicine (BIBM) (pp.
1227–1234). IEEE.
20