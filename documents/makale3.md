NeuralComputingandApplications(2022)34:14287–14296
https://doi.org/10.1007/s00521-021-06882-y
(0123456789().,-vol(V0)123456789().,-volV)
| S.I. : EMERGING | TRENDS IN       | AI & ML  |         |             |     |     |     |     |
| --------------- | --------------- | -------- | ------- | ----------- | --- | --- | --- | --- |
| Knowledge       | distillation    | in plant | disease | recognition |     |     |     |     |
| Ali Ghofrani1   | • Rahil Mahdian | Toroghi1 |         |             |     |     |     |     |
Received:17April2021/Accepted:19December2021/Publishedonline:10March2022
(cid:2)TheAuthor(s),underexclusivelicencetoSpringer-VerlagLondonLtd.,partofSpringerNature2022
Abstract
Recognizing the plantdisease and pests inits golden time is a highly criticalproblem tobe addressed, since the herbalist
can apply treatments within this period and save the agricultural product. In this paper, a deep learning approach to
recognize the disease from the leaves of the plants has been pursued. A client-server system is proposed in which the
server-sidemodelcanleveragehugedeepCNNarchitecturestoclassifythediseases,whereastheclient-sidemodelistobe
chosenamongsmalldeepCNNarchitectureswithlownumberofparametersinordertobeeasilydeployedontheend-user
mobile devices with poor processing powers. Here, a novel knowledge distillation technique has been leveraged that
improves the accuracy levelof the small client-side model significantly. This technique distills the perception knowledge
of a large model classifier and transfers this knowledge to the small model in order to perform a similar prediction
capability. By applying this idea on Plantvillage dataset, we could achieve 97:58% accuracy on a small MobileNet
architecture which is very close to the accuracy of a large Xception model on the server with 99:73% accuracy. Through
applying this teacher-student idea, we could improve the classification rate of the state-of-the-art tiny model by 2:12%.
Keywords Plant disease recognition (cid:2) Deep convolutional neural network (cid:2) Knowledge distillation (cid:2) Tiny mobileNet
1 Introduction
|     |     |     |     | In our     | previous paper          | [1], we proposed | two      | different |
| --- | --- | --- | --- | ---------- | ----------------------- | ---------------- | -------- | --------- |
|     |     |     |     | approaches | for the above-mentioned |                  | problem. | One       |
Ever-increasing population of the world along with daily approach was based on implementing a client-server sys-
lossofaseriousportionofarablelandswillshortlyleadto tem, in which a heavy architecture was developed on the
increasingfooddemandsallovertheworld.Moreover,due server based on the EfficientNet-B0 [2] architecture with
to a high amount of wastes out of farm productions, as a minimum processing requirements on the client side. This
result of pests and plant diseases, a more efficient strategy system could achieve the maximum reported accuracy on
of farming and food production will soon become a Plantvillage dataset in comparison with the other existing
necessity. methods, theretofore. Second approach was a standalone
Artificial intelligence has been played a promising role system based on a light version of MobileNet architecture
thatwassimplifiedanddesignedusingmerely%25offilter
| in many | different aspects | of human | life and various |     |     |     |     |     |
| ------- | ----------------- | -------- | ---------------- | --- | --- | --- | --- | --- |
industries.Inthisregard,herewetrytoaddressthecritical weights with about 200k parameters, yet its accuracy was
problem of plant disease recognition in an efficient way satisfactory and could be implemented as an standalone
usingacomputervisionsolutionwhichcouldbepractically application on normally-used handheld mobile devices[3].
implemented in farming industry and be utilized by the One critical challenge of practical plant disease recog-
| farmers. |     |     |     | nitionproblemistosavethegoldentime,duringwhichthe |                      |               |               |           |
| -------- | --- | --- | --- | ------------------------------------------------- | -------------------- | ------------- | ------------- | --------- |
|          |     |     |     | plant disease                                     | has to be recognized | and           | a proper      | herbalist |
|          |     |     |     | has to be                                         | found in order       | to figure out | the infection | type      |
& RahilMahdianToroghi (e.g., pests or other reasons) and take an appropriate
mahdian@iribu.ac.ir
measuretocurethediseasewithinthistime.Anyproposed
AliGhofrani systemwhichtriestoaddressthemainproblemshouldalso
alighofrani@iribu.ac.ir
|     |     |     |     | have this | capability to be  | qualified in | realistic scenarios. |           |
| --- | --- | --- | --- | --------- | ----------------- | ------------ | -------------------- | --------- |
|     |     |     |     | In line   | with our previous | work [1],    | here we              | intend to |
1 MediaEngineeringFaculty,UniversityofIslamicRepublic
ofIranBroadcasting(IRIBU),Tehran,Iran improve ourmodels inorder toachieve a higher accuracy,
123

14288 NeuralComputingandApplications(2022)34:14287–14296
while at the same time less parameters be imposed to the augmentation that guarantees higher accuracy without an
finalsystem.Abig-pictureofourdesiredrealisticscenario adverse impact on class separability is another novel
and the incorporated designed systems is visualized in proposition in this paper, as well.
Fig. 1. The rest of this paper is organized as follows. First,
Efforts being performed to simplify the ultimate model relatedworksareexplained.Then,theproposedideaalong
(for test-time executions) and to improve the accuracy, with the details of models are explained. The intuitions
makes our proposed model suitable for practical imple- behind using the teacher-student strategy has been pointed
mentations. In [1], we developed a light system on the out and the entire proposed system has been presented, in
client-side and a large one on the server-side. However, in the sequel. Then, experimental results and comparisons
order to make the light system deployable on the end-user have been shown. The conclusion and references are the
mobile devices (i.e. farmers) in a real scenario, we were terminators of this paper.
madetosimplifytheMobileNetarchitectureandthiscould
sacrifice accuracy for simplicity. Dealing with this short-
coming is the major novelty of this paper. Hence, the idea 2 Related works
istoincorporatethenewconceptofknowledgedistillation
in such a way that the achievable perception power by the In order to tackle the plant’s disease detection problem,
larger model (as a teacher with a large number of param- there are approaches that leverage image processing tech-
eters) is to be transferred to a student model with a small niques.Withintheseconventionalmethods,therearethose
number of parameters. This teacher-student strategy (i.e. that extract features out of the input plant image, such as
knowledge distillation [4]) has gained considerable LBP (Local Binary Pattern) and HBBP (Brightness Bi-
improvementsinsimilarexperiences [5,6],andenablesus Histogram Equalization) [7]. Other methods extract fea-
totransfertheperceptionpowerofalargermodel,whichis tures using HoG (Histogram of Gradients) followed by
hinderedtobedeployedinourtaskduetoitslargevolume, SVM classifiers [8, 9]. One recent successful method, in
to a simpler implementable model. This way our hypoth- this regard, uses Otsu classifier [10, 11].
esis is to gain a few percentages of accuracy compared to Theend-to-endmethodshavebeenachievedaveryhigh
our previous work, while preserving the simplicity of the accuracy level in most computer vision applications.
client-side system. Moreover, by ignoring the feature engineering step from
A second novelty, which is deemed to be subsidiary to the processing pipeline they have been succeeded in
that of main one, is a tricky data augmentation technique rapidlyproducingofreliableproductsinsuchapplications.
that allows us to gain a level of accuracy for the final However, they mostly suffer from data starvation. Recent
model. In contrast to generic data augmentation methods availability of large datasets revolved the approaches of
which are prevalently used in object recognition tasks solving the problems from conventional methods toward
including adding noise to the inputs, modifying the inten- end-to-end deep learning ones due to several experiences
sity and contrast of images, and applying geometrical of achieving higher accuracy levels. Among deep learning
changessuch asrotation, shearing,andtranslationthatcan models, CNNs have gained a high reputation in extracting
improve the accuracy and robustness of the model and reliable features for downstream tasks in computer vision
prevent it from overfitting, many of these techniques are applications [12].
invalid in our problem. The reason is due to the fact that By advancement of computer vision and deep learning
there is an intrinsic high correlation among the classes of techniques, new methods have been developed based on
diseasesandtheirvisualdiscriminativefeatures.Therefore, deepCNN,forbothdiseasediagnosisandhealthy/infected
augmentation should not fool the class discrimination plant classifications [13, 14]. Kurup et al. [15] incorpo-
property by the visual effects. An intelligent way of rated the capsule network idea to classify plant diseases
Fig.1 Application’sBigpicture
(Testphase):Smallclient-side
modelvs.ahugepowerful
modelintheserver-sideasthe
classifier,bothusingdeep
CNNs
123

NeuralComputingandApplications(2022)34:14287–14296 14289
and species. In a similar attempt, Verma et al. employed a of MLPs with a down-sized series of
capsule network for disease classification in plants [16]. 2048[ [768[ [256[ [39 neurons, for the Effi-
Both of these papers have achieved considerable results, cientNet basic model a down-sized series of
howeverusingdifferentdatasetspreventsthepossibilityof 1280[ [576[ [256[ [39 neurons, and for the
comparingthetwomethods.Averynicecomparativestudy Tiny MobileNet a down-sized series of 256[ [39
of the plant disease classification methods has been neurons.
recently published in [17, 18], and we encourage the In addition to the output changes, these three architec-
readers to follow these paper to know more about the tures use different input resolutions. The Xception model
literature. uses 3(cid:3)299(cid:3)299 input resolution, whereas EfficientNet
uses 3(cid:3)224(cid:3)224 and Tiny MobileNet 3(cid:3)128(cid:3)128
input resolutions. The details of these three employed
3 The proposed idea architectures are depicted in Figs. 2, 3 and 4. Training of
these models are based on the categorical cross-entropy
In our previous paper [1], it has been shown that a rela- loss function, as already used in [1].
tively simple architecture based on EfficientNet allows us Acomparisonofdifferentdeep-CNNarchitectureonthe
to achieve a higher accuracy compared to the methods, Plantvillage classification task has been reported in [17]
theretofore. In addition, we proposed a very simple model and [15], and further shown in Table 1.
using only 280K parameters that could achieve a reason-
ableaccuracyof95:5%.Inthispaper,weintendtoimprove 3.2 Employingtheideaofknowledgedistillation
theaccuracylevelwhilepreservingthevolumeandnumber
ofparametersforthatsimplemodel.Basedon [4],wetrain Since the main chosen architecture (e.g., Xception or
ahugemodelasateacheranddistillitsothatitsperception EfficientNet) is large and not suitable to be directly
power is transferred to a simple model exactly similar to deployed on the client side, which is an end-user mobile
whatweused inourprevious workfortheclientside,asa device, the idea is to employ it as a teacher model in a
student. Thus, we envisage a higher accuracy level com- teacher-student(akaknowledgedistillation)strategy.Tobe
pared to the vanilla client-side network without increasing more precise, a small deep-CNN architecture with a low
the complexity. number of parameters being easily deployable on normal
mobile devices of the farmers is playing the role of a stu-
3.1 Training setup dent. On the other hand, the knowledge of perception for
the classification task that has been learned by the large
Therearetwotrainingphasestobepursued.Onestepisto model (as a teacher) is planned to be transferred to that
train a huge CNN architecture which is supposed to smallmodel(e.g.,aTinyMobileNet [21,22])thatcouldbe
become powerful for plant disease classification task. This easily deployed on the client side.
architecture could be chosen among large deep CNN In [1], we simplified the vanilla MobileNet-V2 archi-
models such as Xception [19] or EfficientNet [2], which tecture to have only 25% of the parameters, in order to
have been pre-trained on imageNet dataset [20] and makeitfeasibletobedeployedontheclientside.Here,we
employed using transfer learning. In this paper, we choose keep the chosen architecture as in our previous wok,
the stronger model of Xception as the large model to be howeverwedistilltheperceptionknowledgeoftheteacher
trained as the classifier. This architecture is slightly mod- modelontothisTinyMobileNetarchitecturetoimproveits
ified to be customized for our task and that will be accuracy levelcomparedtothatofourpreviouswork.The
explained in the sequel. distillation process is depicted in Fig. 5.
Architectures being pretrained on imageNet data use
224(cid:3)224 input resolution with a final Fully-Connected 3.2.1 Knowledge distillation loss function
(FC) layer of 1000 neurons for the classes. In order to use
these architectures,we need tomodify the inputresolution The idea of teacher-student learning was primarily intro-
as well as the final FC-layer for our custom classification duced as the vanilla knowledge distillation, in which the
task.Hence,duringtrainingthemodelswehavetoremove logits of a large deep model are used as a teacher knowl-
the last FC-layer and connect the feature embedding layer edge to be transferred [4]. A comprehensive study on
(i.e.,alayer before FC-layer)toaGlobal Average Pooling knowledgedistillationmethodshasbeenpublishedin[23].
layer (GAP), followed by a stack of MLPs. These MLP Vanilla knowledgedistillation is called theresponse-based
layers are customized based on the chosen basic architec- knowledgedistillation, inwhich the neural response ofthe
ture, and the classification task. For example, if the basic last output layer of the teacher is important for the student
architectureisXception,thenafterGAPwechooseastack that directly mimics the predictions of his teacher.
123

| 14290                       |                                  |     |     |     | NeuralComputingandApplications(2022)34:14287–14296 |     |     |
| --------------------------- | -------------------------------- | --- | --- | --- | -------------------------------------------------- | --- | --- |
| Fig.2 TeacherXception-based | modelpretrainedonthePlantvillage |     |     |     |                                                    |     |     |
dataset
| Assuming | that the last FC-layer | of a neural | network |     |     |     |     |
| -------- | ---------------------- | ----------- | ------- | --- | --- | --- | --- |
ith
| model has | a logit (i.e., penultimate | output value) | z i for |     |     |     |     |
| --------- | -------------------------- | ------------- | ------- | --- | --- | --- | --- |
class(herei2f1;...;39gandj2½1;39(cid:4)sincethereare39
|     |     |     |     | Fig. 3 Teacher | EfficientNet-based | model pretrained | on the Plantvil- |
| --- | --- | --- | --- | -------------- | ------------------ | ---------------- | ---------------- |
classesofdiseasesinthedataset),asofttargetisdefinedas
lagedata
123

NeuralComputingandApplications(2022)34:14287–14296 14291
entropy between the ground truth labels and the soft logits
of the student model, as
X
L ¼Hðytrue;pðzst;T ¼1ÞÞ¼(cid:5) ytruelogðpðzst;T ¼1ÞÞ
SL i i
i
ð3Þ
where ytrue isthe groundtruthlabel,pðzst;TÞisthestudent
i
modellogitofithclass,andTisthetemperature.Moreover,
the term of knowledge distillation loss, L , which is
KD
defined as the cross entropy between the soft target of the
teacher model and that of the student model is defined as
L ¼Hðpðzteacher;TÞ;pðzst;TÞÞ
KD
¼(cid:5) X pðzteacher;TÞlogðpðzst;TÞÞ ð4Þ
i i
i
where pðzteacher;TÞ is the soft target of teacher model with
i
temperature T and pðzst;TÞ is the soft target of the student
i
model, both for ith class. In addition, the k in (2) is the
balancing weight between the two loss terms and in our
experiments is considered as a constant value of k¼0:07.
The temperature value is also set to T ¼3:25. The loss
function in (2) is minimized using stochastic gradient
descent with momentum.
4 Experimental studies and evaluations
4.1 Data preparation
Dataset which has been used in our experiments is
PlantVillage [24]. This dataset is publicly available and
Fig.4 StudentTinyMobileNet-basedmodelpretrainedonthedataset
contains 54, 306 images of different leaves of healthy as
well as sick plants. There are 39 classes with 14 different
expðz=TÞ
pðz i ;TÞ¼ P expð i z=TÞ ð1Þ plant species and 26 different plant’s diseases. There are
j j 21917 training samples before the augmentation being
where T is called the temperature factor that controls the
applied,fromwhich80%areusedfortrainingand20%for
validation. This dataset also contains a separate test data
importance of each soft target. As explained in [4], these
with 32388 number of samples. Some examples of plant
soft targets contain the dark knowledge information from
theteacher.Thetotalstudentmodel’sdistillationloss,L , diseases are shown in Fig. 6, taken from [24, 25].
ST
During training of the Xception model the input size is
is then written as
larger, hence the batch-size has been chosen to be 32.
L
ST
¼kL
SL
þð1(cid:5)kÞL
KD
ð2Þ
However, for the EfficientNet and Tiny MobileNet with
smaller input size, the batch-size could be chosen as 64.
where L is the student model loss defined as the cross
SL
Table1 Trainingvs.validation
Model #Ofparameters #Ofepochs Trainaccuracy Valid.accuracy
accuracylevelsandthenumber
ofparametersfordifferentdeep
LeafNet 324K 59 85.90 79.61
CNNmodelsbeingtrainedon
VGG-16 138M 59 83.39 81.89
Plantvillagedata[15,17]
ResNet50 23.6M 55 98.73 94.23
Xception 23M 34 99.90 97.98
Capsnet 9.5M 100 Notreported 95.29
123

| 14292 |     |     |     |     |     |     | NeuralComputingandApplications(2022)34:14287–14296 |     |     |     |
| ----- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- |
Fig.5 Visualgraphofthe
proposedidea.Xception
networkasateacheris
pretrainedonthedatasetandits
knowledgeisdistilledonthe
TinyMobileNet(aka.Lite-
MobileNet).Thestudent
networkisthendeployedonthe
clientside(i.e.,thefarmer’s
mobiledevices)asalight
executableapplication
Fig.6 Samplesofplantdiseases
fromthePlantVillagedataset
4.1.1 Intelligent data augmentation value for the contrast is about 10%. Figure 7 denotes dif-
|     |     |     |     |     |     | ferent visual | effects based | on various | augmentation | meth- |
| --- | --- | --- | --- | --- | --- | ------------- | ------------- | ---------- | ------------ | ----- |
Theonlydiscriminatingfeaturestoclassifythediseasesof ods, and the intuition of how they can impact the
plants with infected leaves are the visual effects that are classification task. As it is obvious from the figure, for-
imposedtothem.Therefore,theapplieddataaugmentation bidden augmentations have devastated the discriminative
should not create overlaps on these visual features that visual features of diseases.
might fool the classifier. In order to gain from data aug- The effect of data augmentation on the training and
mentation it should be performed in a tricky way, other- testing accuracy of the models is investigated and pre-
wise no benefit would be obtained and the model will not sented in Table 2. This effect is clearly observable by
converge. In [1], we could achieve a 99% of accuracy looking at the validation and test accuracy levels for the
using a relatively strong model (i.e. EfficientNet) with a basic models, in the last two columns of the table.
| reasonable | data            | augmentation. | All              | these augmentations |               |              |              |           |     |     |
| ---------- | --------------- | ------------- | ---------------- | ------------------- | ------------- | ------------ | ------------ | --------- | --- | --- |
| have been  | performed       | based         | on the explained |                     | intuition and |              |              |           |     |     |
|            |                 |               |                  |                     |               | 4.2 Hardware | and software | platforms |     |     |
| practical  | considerations, | as            | follows:         | (1) Augmentations   |               |              |              |           |     |     |
basedonimagegeometricalmodifications,suchasrotation
and perspective,are permissible. Here, we applied random The system platform has been an Intel Core i7-7700 with
rotations along with isometric translations up to 1/10 of 32 GB of RAM, an Nvidia GTX 1080-Ti GPU and the
lengthandwidthofimagesandnon-isometrictranslations, framework of Tensorflow 2.3 on Cuda 10.2.
| such as perspective, |         | up to 30%.   | (2) Adding | noise     | or patch |             |               |     |     |     |
| -------------------- | ------- | ------------ | ---------- | --------- | -------- | ----------- | ------------- | --- | --- | --- |
|                      |         |              |            |           |          | 4.3 Results | and Analytics |     |     |     |
| removal              | are not | allowed. (3) | Changes    | of colors | across   |             |               |     |     |     |
channelsarenotpermissible,sinceitadverselyimpactsthe
discriminating property among classes. (4) Contrast and The objective of training losses is to maximize the accu-
brightnessareverycriticalentities.Themaximumallowed racylevelofclassificationoverthevalidationdata.Details
| brightness | variations | of an image | pixels | is 12%, | otherwise |             |             |                |         |        |
| ---------- | ---------- | ----------- | ------ | ------- | --------- | ----------- | ----------- | -------------- | ------- | ------ |
|            |            |             |        |         |           | of training | process and | loss functions | for the | chosen |
the diseases of leaves may not be correctly classified. This architectures have been already explained in section 3.1.
123

| NeuralComputingandApplications(2022)34:14287–14296 |     |     |     |     |     |     |     |     |     |     | 14293 |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- |
Fig.7 (Top:left2right-Allowed
augmentations)Originalimage,
10%brighter,rotation,shearing;
(bottom:left2right-forbidden
augmentations)AddedGaussian
noise,over-increased
brightness,under-decreased
contrast
Table2 Dataaugmentationeffectontheaccuracyofmodels
| Model           |                | w/outAugmentation |              |               | WithAugmentation |              |     |     |     |     |     |
| --------------- | -------------- | ----------------- | ------------ | ------------- | ---------------- | ------------ | --- | --- | --- | --- | --- |
|                 |                | Train             | Valid.       | Test          | Train            | Valid. Test  |     |     |     |     |     |
| Xception        |                | 99.76             | 96.23        | 96.18         | 99.81            | 99.84 99.73  |     |     |     |     |     |
| EfficientNet-B0 |                | 99.76             | 95.52        | 95.37         | 99.78            | 99.85 99.69  |     |     |     |     |     |
| TinyMobileNet   |                | 98.87             | 94.26        | 94.58         | 99.62            | 99.82 95.46  |     |     |     |     |     |
| Figure 8        | visualizes     | the               | general      | training      | strategy         | of deep-     |     |     |     |     |     |
| CNNs being      | used           | in this           | work.        |               |                  |              |     |     |     |     |     |
| The             | training       | and validation    |              | accuracy      | of               | the Xception |     |     |     |     |     |
| model,          | Tiny MobileNet |                   | and          | the distilled | Tiny             | MobileNet    |     |     |     |     |     |
| models          | are depicted   | in                | Fig. 9.      |               |                  |              |     |     |     |     |     |
| In order        | to assess      | our               | hypothesis   |               | over the         | knowledge    |     |     |     |     |     |
| distillation    | idea,          | we                | incorporated |               | a large          | Xception     |     |     |     |     |     |
| model [19],     | as             | well as           | a relatively | large         | EfficientNet     | [2]          |     |     |     |     |     |
| architecture.   | These          | networks,         |              | due to        | being heavily    | large,       |     |     |     |     |     |
aremostlyappropriatefortheserver-sideimplementations.
| Now, the  | accuracy     | of       | these   | models       | along               | with a Tiny |     |     |     |     |     |
| --------- | ------------ | -------- | ------- | ------------ | ------------------- | ----------- | --- | --- | --- | --- | --- |
| MobileNet | architecture |          | being   | individually | trained             | on the      |     |     |     |     |     |
| dataset   | will be      | compared | to      | the multiple | distilled           | Tiny        |     |     |     |     |     |
| MobileNet | versions     | of       | itbeing | trained      | ina teacher-student |             |     |     |     |     |     |
Fig.8 GeneraltrainingschemeofCNNmodelsanddatapartitioning.
| scheme           | using the | large         | Xception | and         | EfficientNet | models  |        |                   |         |                 |              |
| ---------------- | --------- | ------------- | -------- | ----------- | ------------ | ------- | ------ | ----------------- | ------- | --------------- | ------------ |
|                  |           |               |          |             |              |         | 80% is | used for training | and 20% | for validation. | Test data is |
| as the teachers, |           | respectively. |          | The results | of these     | compar- |        |                   |         |                 |              |
separatelyavailable
| isons are          | shown | in Table | 3.                         |     |     |     |              |              |           |            |          |
| ------------------ | ----- | -------- | -------------------------- | --- | --- | --- | ------------ | ------------ | --------- | ---------- | -------- |
| AsitisshowninTable |       |          | 3,thelargeXceptionmodelasa |     |     |     |              |              |           |            |          |
|                    |       |          |                            |     |     |     | our proposed | distillation | mechanism | to achieve | a higher |
teacher outperforms the EfficientNet model with higher performance with respect to the vanilla model is further
| number | of parameters |     | and | the distillation |     | of Xception |                |                                      |     |     |     |
| ------ | ------------- | --- | --- | ---------------- | --- | ----------- | -------------- | ------------------------------------ | --- | --- | --- |
|        |               |     |     |                  |     |             | observed.Table | 4showstheresultsofthesemeasurements. |     |     |     |
model on the Tiny MobileNet achieves a higher accuracy. Finally, the attention-maps of competing models over
| On the | other hand, | the | accuracy | of  | the distilled | Tiny |     |     |     |     |     |
| ------ | ----------- | --- | -------- | --- | ------------- | ---- | --- | --- | --- | --- | --- |
thesickplantsaredepictedinFig.10.Asitisvisualizedin
MobileNet as a student compared to that of the vanilla the figure, the Xception model performs stronger than the
| MobileNet | on  | the same | test | data is | 2:12% | higher. In a |             |            |                |         |           |
| --------- | --- | -------- | ---- | ------- | ----- | ------------ | ----------- | ---------- | -------------- | ------- | --------- |
|           |     |          |      |         |       |              | others with | respect to | the anomalies. | Vanilla | MobileNet |
another evaluation, the Precision, Recall and F1-Score has performed the classification correctly, however the
measureshavealsobeencalculatedinwhichtheefficacyof
attentionofithasbeenscatteredovertheinfectedarea.On
123

| 14294 | NeuralComputingandApplications(2022)34:14287–14296 |     |     |     |
| ----- | -------------------------------------------------- | --- | --- | --- |
Fig.9 Training(blue)versusvalidation(orange)Accuracycurvesof(a)TeacherXceptionmodel,(b)vanillaTinyMobileNet,and(c)Distilled
TinyMobileNetarchitectures.PlotsaredrawnbyTensorBoardsoftware
Table3 Effectofdistillinghuge
|     | Model | Training | Validation | Testing |
| --- | ----- | -------- | ---------- | ------- |
modelsonaTinyMobileNet.
Xceptionisthelargestmodel
|     | Xception[19] | 99.81 | 99.84 | 99.73 |
| --- | ------------ | ----- | ----- | ----- |
andthenisEfficientNet.
|     | EfficientNet[2] | 99.78 | 99.85 | 99.65 |
| --- | --------------- | ----- | ----- | ----- |
PlantVillagedataisusedforall
| models | XceptiondistilledonTinyMobileNet     | 99.69 | 97.62 | 97.58 |
| ------ | ------------------------------------ | ----- | ----- | ----- |
|        | EfficientNetdistilledonTinyMobileNet | 99.64 | 96.56 | 96.41 |
|        | VanillaTinyMobileNet[1]              | 99.62 | 95.82 | 95.46 |
Table4 Evaluationoftestset
|     | Model | Precision | Recall | F1-score |
| --- | ----- | --------- | ------ | -------- |
onPrecision,Recalland
| F1ScoreMetrics,forthe | Xception | 99.18 | 99.65 | 99.41 |
| --------------------- | -------- | ----- | ----- | ----- |
proposeddistilledmodels,as
|     | EfficientNet | 98.96 | 99.59 | 99.27 |
| --- | ------------ | ----- | ----- | ----- |
wellasthebasicones.
PlantVillagetestdataisusedfor XceptiondistilledonTinyMobileNet 97.47 97.56 97.51
allthesemodels EfficientNetdistilledonTinyMobileNet 96.87 96.79 96.83
|     | VanillaTinyMobileNet | 96.02 | 95.88 | 95.94 |
| --- | -------------------- | ----- | ----- | ----- |
123

| NeuralComputingandApplications(2022)34:14287–14296 |     |     |     |     |     |     |     |     |     |     |     | 14295 |
| -------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- |
Fig.10 Samplesickplantsand
theirattentionmapbydifferent
models(Columns-left2right)
1stcolumn-top2bottom:
Tomato:bacterialspot,
Grape:LeafblightIsariopsis
LeafSpot,Cornmaize:Northern
| LeafBlight; | 2ndcolumn- |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
top2bottom:Attentionmapsof
theassociatedleavesbyVanilla
MobileNetdenselayer;
3rdcolumn-top2bottom:
Attentionmapsoftheassociated
leavesbyDistilledMobileNet
| denselayer; | 4thcolumn- |     |     |     |     |     |     |     |     |     |     |     |
| ----------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
top2bottom:Attentionmapsof
theassociatedleavesby
Xceptionmodeldenselayer
| theotherhand,afterdistillationtheattentionofMobileNet |             |     |              |     |               |        | Declarations |     |     |     |     |     |
| ----------------------------------------------------- | ----------- | --- | ------------ | --- | ------------- | ------ | ------------ | --- | --- | --- | --- | --- |
| has been                                              | morefocused |     | on theregion |     | beingaffected | by the |              |     |     |     |     |     |
disease. Conflictofinterest Theauthorsdeclarethattheyhavenoconflictof
interest.
5 Conclusion
References
In this paper, the plant disease recognition task has been 1. Ghofrani A, Mahdian R, Behnegar, H (2021) Plant disease
addressed.Adeeplearningstructurehasbeenincorporated, recognitionusingoptimizeddeepconvolutionalneuralnetworks.
|           |       |           |               |           |           |           | In: Pattern | Recognition | and Artificial    | Intelligence: |           | 4th Mediter- |
| --------- | ----- | --------- | ------------- | --------- | --------- | --------- | ----------- | ----------- | ----------------- | ------------- | --------- | ------------ |
| in which  | two   | different | architectures |           | have been | involved. |             |             |                   |               |           |              |
|           |       |           |               |           |           |           | ranean      | Conference, | MedPRAI           | 2020,         | Hammamet, | Tunisia,     |
| One large | model | which     | could         | be easily | deployed  | on the    |             |             |                   |               |           |              |
|           |       |           |               |           |           |           | December    | 20–22,      | 2020, Proceedings | 4,            | pp 18–30. | Springer     |
server side and be chosen from among large CNN models InternationalPublishing
(e.g., Xception or EfficientNet), and a small model to be 2. TanM,LeQV(2019)Efficientnet:Rethinkingmodelscalingfor
convolutionalneuralnetworks.arXivpreprintarXiv:1905.11946
qualifiedasapropercandidatetobedeployedontheclient
|           |      |             |        |      |        |            | 3. Ghofrani | A, Mahdian | R, Ghanbari | S (2019) | Realtime | face-de- |
| --------- | ---- | ----------- | ------ | ---- | ------ | ---------- | ----------- | ---------- | ----------- | -------- | -------- | -------- |
| side that | is a | user mobile | device | with | a poor | processing |             |            |             |          |          |          |
tectionandemotionrecognitionusingmtcnnandminishufflenet
capability. The small CNN model is potentially limited in v2.In:20195thconferenceonknowledgebasedengineeringand
achievingtheaccuracylevelofthelargeserver-sidemodel. innovation(KBEI),IEEE,pp817–821
4. HintonG,VinyalsO,DeanJ(2015)Distillingtheknowledgeina
| However, | we  | have proposed |     | a knowledge |     | distillation |     |     |     |     |     |     |
| -------- | --- | ------------- | --- | ----------- | --- | ------------ | --- | --- | --- | --- | --- | --- |
neuralnetwork.arXivpreprintarXiv:1503.02531
| technique | in which | a small | model | is  | capable of | achieving |              |         |                |           |     |           |
| --------- | -------- | ------- | ----- | --- | ---------- | --------- | ------------ | ------- | -------------- | --------- | --- | --------- |
|           |          |         |       |     |            |           | 5. Salehi M, | Sadjadi | N, Baselizadeh | S, Rohban | MH, | Rabiee HR |
higher performances using a teacher-student mechanism (2020) Multiresolution knowledge distillation for anomaly
for training. Implementing this idea along with an intelli- detection.arXivpreprintarXiv:2011.11108
6. TouvronH,CordM,DouzeM,MassaF,SablayrollesA,Je´gouH
| gent data        | augmentation |     | technique  | aiming | at improving      | the |        |          |                |                    |     |                |
| ---------------- | ------------ | --- | ---------- | ------ | ----------------- | --- | ------ | -------- | -------------- | ------------------ | --- | -------------- |
|                  |              |     |            |        |                   |     | (2020) | Training | data-efficient | image transformers |     | & distillation |
| model robustness |              | and | preventing | it     | from overfitting, | we  |        |          |                |                    |     |                |
throughattention.arXivpreprintarXiv:2012.12877
could achieve a 2:12% better accuracy level than our pre- 7. Singh K, Kumar S, Kaur P (2019) Automatic detection of rust
viouslyproposedsystembeingthestate-of-the-art,hitherto. disease of lentil by machine learning system using microscopic
images.IntJElectComputEng9(1):660–666
Inthemeantime,thelightnessoftheclient-sidemodelhas
|                |     |          |       |          |               |     | 8. Islam MA, | Yousuf | MSI, Billah | M (2019) | Automatic | plant |
| -------------- | --- | -------- | ----- | -------- | ------------- | --- | ------------ | ------ | ----------- | -------- | --------- | ----- |
| been preserved |     | and this | makes | the idea | implementable | for |              |        |             |          |           |       |
detectionusinghogandlbpfeatureswithsvm.IntJComput(IJC)
| real-world | scenario | in  | an intelligent |     | farming | and a high | 33(1):26–38 |     |     |     |     |     |
| ---------- | -------- | --- | -------------- | --- | ------- | ---------- | ----------- | --- | --- | --- | --- | --- |
performance agricultural productions. 9. SullcaC,MolinaC,Rodr´ıguezC,Ferna´ndezT(2019)Diseases
detectioninblueberryleavesusingcomputervisionandmachine
learningtechniques.IntJMachLearnComput9(5):656–661
10. SunG,JiaX,GengT(2018)Plantdiseasesrecognitionbasedon
imageprocessingtechnology.JElectComputEng,2018
123

14296 NeuralComputingandApplications(2022)34:14287–14296
11. Saradhambal G, Dhivya R, Latha S, Rajesh R (2018) Plant dis- 19. Chollet F (2017) Xception: Deep learning with depthwise sepa-
ease detection and its solution using image classification. Int J rable convolutions. In: proceedings of the IEEE conference on
PureApplMath119(14):879–884 computervisionandpatternrecognition,pp1251–1258
12. PooleL,BrownD(2021)Investigatingpopularcnnarchitectures 20. Deng J, Dong W, Socher R, Li L-J, Li K, Fei-Fei L (2009)
forplantdisease detection.In:2021internationalconference on Imagenet: A large-scale hierarchical image database. In: 2009
artificialintelligence, bigdata,computing anddatacommunica- IEEE conference on computer vision and pattern recognition,
tionsystems(icABCD),IEEE,pp1–5 IEEE,pp248–255
13. Ferentinos KP (2018) Deep learning models for plant disease 21. SandlerM,HowardA,ZhuM,ZhmoginovA,ChenL-C(2018)
detectionanddiagnosis.ComputElectronAgric145:311–318 Mobilenetv2: inverted residuals and linear bottlenecks. In: pro-
14. Hanson A, Joel M, Joy A, Francis J (2017) Plant leaf disease ceedingsoftheIEEEconferenceoncomputervisionandpattern
detectionusingdeeplearningandconvolutionalneuralnetwork. recognition,pp4510–4520
IntJEngSci5324:2–4 22. GhofraniA,MahdianR,TabatabaieSM(2020)Attention-based
15. KurupRV,AnupamaM,VinayakumarR,SowmyaV,SomanK faceantispoofingofrgbcamerausingaminimalend-2-endneural
(2019) Capsule network for plant disease and plant species network.In:2020internationalconferenceonmachinevisionand
classification. In: international conference on computational imageprocessing(MVIP),IEEE,pp1–6
visionandbioinspiredcomputing,pp413–421.Springer 23. GouJ,YuB,MaybankSJ,TaoD(2021)Knowledgedistillation:
16. VermaS,ChugA,SinghAP(2020)Exploringcapsulenetworks asurvey.IntJComputVis129:1–31
for disease classification in plants. J Stat Manag Syst 24. MohantySP,HughesDP,Salathe´ M(2016)Usingdeeplearning
23(2):307–315 for image-based plant disease detection. Frontiers Plant Sci
17. Saleem MH, Potgieter J, Arif KM (2020) Plant disease classifi- 7:14–19
cation: a comparative evaluation of convolutional neural net- 25. HughesD,Salathe´ Metal.(2015)Anopenaccessrepositoryof
worksanddeeplearningoptimizers.Plants9(10):1319 images on plant health to enable the development of mobile
18. AdedojaAO,OwolawiPA,MapayiT,TuC(2021)Progresson diseasediagnostics.arXivpreprintarXiv:1511.08060
deep learning models for plant disease detection: A survey. In:
2021internationalconferenceonartificialintelligence,bigdata,
Publisher’s Note Springer Nature remains neutral with regard to
computing and data communication systems (icABCD), IEEE,
jurisdictionalclaimsinpublishedmapsandinstitutionalaffiliations.
pp1–9
123