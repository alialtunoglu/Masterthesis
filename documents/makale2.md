algorithms
Article
Integrating ShuffleNetV2 with Multi-Scale Feature Extraction
and Coordinate Attention Combined with Knowledge
Distillation for Apple Leaf Disease Recognition
Wei-ChiaLo andChih-ChinLai*
DepartmentofElectricalEngineering,NationalUniversityofKaohsiung,Kaohsiung811726,Taiwan;
m1135113@mail.nuk.edu.tw
* Correspondence:cclai@nuk.edu.tw
Abstract
Misdiagnosingplantdiseasesoftenleadstoarangeofnegativeconsequences,including
theoveruseofpesticidesandunnecessaryfoodwaste. Traditionally,identifyingdiseases
on plant leaves has relied on manual visual inspection, making it a complex and time-
consumingtask. Sincetheadventofconvolutionalneuralnetworks,however,recognition
performance for leaf diseases has improved significantly. Most contemporary studies
thatapplyAItechniquestoplant-leafdiseaseclassificationfocusprimarilyonboosting
accuracy,frequentlyoverlookingthelimitationsposedbyresource-constrainedreal-world
environments. To address these challenges, this thesis employs knowledge distillation
to enable small models to approximate the recognition capabilities of larger ones. We
enhanceaShuffleNetV2-basedmodelbyintegratingmulti-scalefeatureextractionanda
coordinate-attentionmechanism,andwefurtherimprovethelightweightstudentmodel
throughknowledgedistillationtoboostitsrecognitionperformance. Experimentalresults
show that the proposed model achieves 93.15% accuracy on the Plant Pathology 2021-
FGVC8dataset,utilizingonly0.36Mparametersand0.0931GFLOPs. Comparedtothe
ResNet50baseline,ourarchitectureslashesparametersbynearly98%whilelimitingthe
accuracygaptoamere1.6%. Theseresultsconfirmthemodel’sabilitytomaintainrobust
performance with minimal computational overhead, providing a practical solution for
precisionagricultureonresource-limitededgedevices.
Keywords: plantdiseaseidentification;knowledgedistillation;deeplearning;lightweight
models
1. Introduction
Applesareanutrient-rich,high-valuefruitgrownintemperateregionsandplayacru-
AcademicEditor:FrankWerner cialroleintheglobalagribusinesseconomy. IntheUnitedStatesalone,theappleindustry
Received:3December2025 generatesanannualoutputvaluedat$15billion[1]. Despiteitseconomicsignificance,this
Revised:3February2026 vastindustryfacesseriousthreatsfromplantdiseases. Diseaseinfectionsremainoneof
Accepted:8February2026
themostcriticalchallengesincropproduction,oftencausingsubstantialeconomiclosses.
Published:13February2026
Pathogenssuchasviruses, fungi, andbacteriacaninfectvariousplantparts, including
Copyright:©2026bytheauthors.
roots,stems,andleaves.Theirimpactrangesfromdamagingfruitappearanceandreducing
LicenseeMDPI,Basel,Switzerland.
marketvaluetocausingsharpdeclinesinyieldandevenplantdeath,posingamajorthreat
Thisarticleisanopenaccessarticle
distributedunderthetermsand toagriculturalproductivity. Therefore,accurateandtimelydiseasedetectionisessentialto
conditionsoftheCreativeCommons safeguardcropyieldsandmaintainindustrystability[2].
Attribution(CCBY)license.
Algorithms2026,19,151 https://doi.org/10.3390/a19020151

Algorithms2026,19,151 2of23
Traditionalplantdiseasediagnosislargelydependsonvisualinspectionsconductedby
farmersoragriculturalexperts. However,thisapproachhasseverallimitations: itistime-
consuming,labor-intensive,andresource-demanding. Moreimportantly,thediagnostic
accuracyheavilyreliesonindividualexpertiseandisoftensubjective.Additionally,manual
methodsaredifficulttoscaleforlargemodernfarmsandarepronetohumanerror,failing
tomeettheincreasingdemandsfortimeliness,accuracy,andautomationincontemporary
agriculture. Therefore, developing automated, high-precision diagnostic technologies
withearlywarningcapabilitieshasbecomeacriticalnecessitytosupportthesustainable
developmentoftheappleindustry.
Early research on automated plant disease detection combined traditional image
processingtechniqueswithclassicalmachinelearningalgorithms. Thesemethodstypically
follow a fixed multi-stage workflow. First, digital images of plant leaves are acquired.
Next,imagepreprocessingisperformed,includingnoiseremoval,contrastenhancement,
and conversion from device-dependent RGB color spaces to more stable color spaces
suchasHSI(Hue,Saturation,Intensity)orCIEL∗a∗b∗ [3]toreducetheeffectsoflighting
variations. Thisstepisfollowedbyimagesegmentation,wherevarioustechniqueslike
colorthresholding[4],Otsusegmentation[5],andK-meansclustering[6,7]areappliedto
distinguishdiseasedareasfromhealthyleafbackgrounds. Finally,featuresareextracted
fromthesegmentedimagesandusedforclassification. Despitelayingthefoundationfor
automateddiagnosis,thesemethodsreliedonhand-craftedfeatures,whichoftenexhibited
insufficientgeneralizationandstabilitywhenencounteringuncertainfactorslikevarying
illuminationandclutteredbackgroundsinnaturalfieldenvironments.
Inrecentyears,deeplearningmethodshavegainedsignificantprominence,drivenby
theavailabilityoflargedatasetsandadvancesinthecomputationalpowerandmemory
capacity of graphics processing units (GPUs). Convolutional neural networks (CNNs)
have revolutionized plant disease classification by automatically learning hierarchical
anddiscriminativefeaturesdirectlyfromrawimages, eliminatingtheneedformanual
featureextraction.
1.1. EvolutionofCNNsandTransferLearninginPlantPathology
In agricultural scenarios, high-quality and balanced datasets are often difficult to
obtain. Toaddressthis,transferlearninganddataaugmentationarewidelyused. Kumar
etal.[8]appliedtransferlearningbyfine-tuningapre-trainedVGG19model,achieving
97.92% accuracy on an apple leaf disease dataset. To address class imbalance, Sulisty-
owatietal.[9]optimizedVGG16usingSMOTEtechniques. Furthermore,Chenetal.[10]
proposedatwo-stageoptimizationusingamodifiedCycleGANtogeneratehigh-quality
syntheticdiseaseimages,whichimprovedclassificationaccuracyby14.7%comparedtous-
ingonlyoriginaldata. Whiletransferlearningmitigatesdatascarcity,manystudiesrelyon
high-parametermodels,makinglow-latencyinferenceonembeddeddeviceschallenging.
1.2. LightweightArchitecturesandAttention-BasedFeatureEnhancement
To enable deployment on edge devices, lightweight network design has become a
popular research direction. Bi et al. [11] utilized the MobileNet model with depthwise
separable convolutions to significantly reduce computational complexity, achieving an
averageprocessingtimeof0.22s. Chaoetal.[12]proposedXDNet,combiningXception
andDenseNetfeaturestoachieve98.82%accuracywithminimalparameters. Yuetal.[13]
introduced MSO-ResNet, compressing the model size by 84% and reducing inference
latencyto25.84ms. Additionally,Fuetal.[14]developedalightweightdesignbasedon
AlexNet,integratingparallelmulti-scalemodulesandchannelattentionmechanismsto
enhancefeaturecapture.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 3of23
Intermsoffeaturelocalization,attentionmechanismsplayacrucialrole. Yuetal.[15]
proposedtheLeafSpotAttentionNetwork(LSA-Net),whichusesspatialattentiontoguide
themodeltowardlesioninformation. Luoetal.[16]improvedresidualnetworksusing
multi-scale feature fusion to prevent the loss of fine-grained disease symptoms during
downsampling. Despitetheseimprovements,mostdesignsdonotfullyconsidermulti-
scalelesionfeatureloss. Thisstudyaddressesthisbyusingmulti-scaleconvolutionsand
spatialdimensionattention.
1.3. HybridModelsandKnowledgeTransferStrategies
Recent hybrid architectures combine CNN local features with Transformer global
information. Sietal.[17]proposedDBCOST,adual-branchmodelthatoutperformssingle-
branch ResNet or Swin Transformer models. Ait Nasser and Akhloufi [18] introduced
CTPlantNet,anensemblemodelintegratingSEResNext-50,EfficientNet-V2S,andSwin-
Largeformorerobustclassification.
Knowledgedistillationisalsoakeystrategyforbalancingperformanceandefficiency.
Wangetal.[19]usedMobileNetV2asastudentmodelandDenseNet201asateachermodel,
“distilling”complexfeaturesintothelightweightmodel. Theirresultsshowedthatthe
distilledmodelachieved98.32%accuracy,significantlyhigherthanthebaselinewithout
distillation(97.33%),provingthatthismethodeffectivelycompensatesforperformance
lossduetoparameterreduction.
Literaturereviewsrevealthatdespitebreakthroughsinaccuracy,agapremainsbe-
tweenmodelcomplexityandreal-worlddeploymentneeds. High-performancemodels
oftenrequiresubstantialcomputationandstorage,limitingtheiruseonmobiledevicesfor
frontlinefarmers. Effectivelycompressingmodelswithoutsacrificingsignificantaccuracy
remainsamajorchallenge.
Knowledgedistillationpresentsahighlypromisingapproachtosignificantlyenhance
theperformanceoflightweightmodelswithoutincreasingtheirparametercount.Assuch,it
isconsideredakeytechnologyforaddressingchallengesinplantleafdiseaseidentification.
This paper proposes an improved version of ShuffleNetV2 by integrating knowledge
distillationwithmulti-scalefeatureextractionandcoordinateattentionmechanisms. We
applythismethodtothePlantPathology2021-FGVC8datasetandAppleLeaf9dataset,
aimingtodevelopamodelthatachievesrecognitionaccuracycomparabletolarge-scale
modelswhilemaintaininglowlatencyandresourceconsumption. Thisenablesseamless
deploymentonmobiledevices. Themaincontributionsofthispaperareasfollows:
1. WeproposeanenhancedShuffleNetV2architecturebyintegratingmulti-scaledepth-
wiseconvolution(MDC)andcoordinateattention(CA).Thisdesignimprovesthe
model’s ability to perceive lesions at varying scales and enhances localization of
affectedareas,allwithoutsignificantlyincreasingcomputationaloverhead. Thiseffec-
tivelyaddressesthelimitationsoftheoriginalShuffleNetV2inclassifyingcomplex
plantdiseases.
2. Knowledgedistillationisemployedtoboosttheperformanceoflightweightmodels.
Throughcomprehensivehyperparametertuning,complexknowledgeandinter-class
relationshipslearnedbylargeteachermodelsareeffectivelytransferred,substantially
improvingtheaccuracyofthestudentmodel.
3. Visual analysis and validation are conducted using Grad-CAM++ visualizations,
whichconfirmthattheproposedmodelmoreaccuratelyfocusesontheactuallesion
regionscomparedtobaselinemodels.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 4of23
2. TheProposedApproach
This paper tackles the challenge of accurately identifying apple leaf diseases on
resource-constrained devices. The overall research workflow is illustrated in Figure 1.
Asdepicted,theworkflowconsistsoftwomaincomponents: datapreparationandmodel
trainingandtesting.
Figure1. Methodologyworkflowoftheplantleafdiseaserecognitionmodelbasedonmodified
ShuffleNetV2andknowledgedistillation.
Duringthedatapreparationphase,weselectedthepubliclyavailablePlantPathology
2021-FGVC8datasetforsegmentation. Theoriginalimageswerefirstresizedtomatchthe
model’sinputrequirements.Subsequently,thedatasetwasdividedintotraining,validation,
andtestingsets,whichwereusedformodeltraining,hyperparametertuning,andfinal
performanceevaluation,respectively. Toenhancethemodel’sgeneralizationability,data
augmentationtechniquesaredynamicallyappliedtothetrainingimagesthroughoutthe
trainingprocess.
Giventhecomplexityofplantdiseaseidentification, directlyapplyingtheoriginal
ShuffleNetV2poseschallengessuchaslimitedperceptionofmulti-scalediseasesymptoms
andvulnerabilitytobackgroundnoiseinleafimages. Therefore,inmodeldevelopment
andtraining,weselectedthelightweightShuffleNetV20.5×network,optimizedformobile
devices,asthebackbonetomeettheefficiencydemandsofedgecomputing. Thisnetwork
offersafavorablebalancebetweencomputationalefficiencyandperformance. Toimprove
itsfeatureextractioncapabilities,weproposeenhancementstoShuffleNetV20.5×. First,
weintroduceamulti-scaledepthwiseconvolution(MDC)module(labeled“Multi-scale
DWConv”inFigure1)tohandlethevaryingsizesofplantleaflesions. Next,weintegrate
acoordinateattention(CA)mechanism(labeled“CAMechanism”inFigure1)toenhance
the model’s focus on critical lesion regions. Despite these improvements, the inherent
limitationsoflightweightmodelsmaystillresultinperformancegapscomparedtolarger
models. To bridge this gap without increasing inference costs, we employ knowledge
distillation (labeled “Knowledge Distillation” in Figure 1), where a pre-trained, more
powerfulteachermodelguidesthetrainingofthelightweightstudentmodel.
2.1. ShuffleNetV2
Inpracticalsmartagricultureapplications,modelsmustbedeployedonmobilede-
vicesorembeddedsystems,whichhavefarlesscomputationalpowerandstoragecapacity
comparedtoserver-sideplatforms. Whilehigh-performancemodelslikeVisionTransform-
ersdeliverexcellentrecognitionaccuracy,theirlargeparametersizesandcomputational
demands make real-time operation on field devices impractical. Therefore, this paper
selectsShuffleNetV20.5×—thesmallestvariantintheShuffleNetV2family—asthefounda-
tionalarchitectureformobiledevicedeployment. Thislightweightnetworkstrikesanideal
balancebetweenefficiencyandperformance,makingitwellsuitedforresource-constrained
environmentstypicalinagriculturalsettings.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 5of23
ShuffleNetV2[20]employsachannelsplitmechanismthatdividestheinputfeature
mapintotwoparallelbranches. Onebranchfunctionsasashortcutconnection,preserving
theoriginalinformationtofacilitategradientpropagationandefficientinformationflow.
The other branch performs feature extraction and learning tasks. To enable effective
communication between these two independent branches, a channel shuffle operation
fusescross-channelinformationbyrearrangingchannels. Thisdesignsignificantlyreduces
computationalcomplexitywhilemaintaininghighaccuracy.Figure2illustratesthedetailed
networkarchitecture,wheretheInvertedResidualunitservesasthefundamentalbuilding
blockforStages2through4. Thisunitimplementsthechannelsplitandfeaturelearning
operations. Intheoperatorfield,“s2”denotesastrideof2,appliedatthestartofeachstage
todownsamplethespatialdimensionsbyhalf,while“s1”indicatesastrideof1,preserving
spatialdimensionsduringfeatureprocessing.
Figure2.DetailedarchitecturediagramoftheShuffleNetV20.5×deeplearningmodel.
2.2. ModifiedShuffleNetV2
AlthoughShuffleNetV2exhibitsexcellentcomputationalefficiency,itsoriginaldesign
was primarily targeted at general image classification tasks. When applied directly to
complexplantdiseaseidentification,twomajorchallengesarise. First,theuseoffixed-size
convolutionkernelslimitsthemodel’sabilitytoextractfeaturesfromdiseasesymptoms
thatvarywidelyinsize.Second,theabsenceofanattentionmechanismreducesthemodel’s
capacitytoeffectivelyfocusoncriticallesionareasamidstnoisyreal-worldbackgrounds,
suchasfoliage,soil,andlightingvariations,therebylimitingclassificationaccuracy. To
overcometheseissues,weintroducedtwokeyenhancementstoShuffleNetV2’scoreunit:
amulti-scalefeatureextractionmoduleandacoordinateattentionmechanism. Theseim-
provementsenhancethemodel’sabilitytodetectdiversediseasesymptomsandaccurately
localizecriticallesionregions. Webelievethataddressingthesecriticalbottlenecks,even
attheexpenseofaslightincreaseincomputationaloverhead,isessentialforoptimizing
overallperformanceandefficiency.
2.2.1. Multi-ScaleFeatureExtractionModule
Toaddresstheissueofincompletefeatureextractioncausedbyfixed-sizeconvolu-
tionalkernelsintheoriginalShuffleNetV2—particularlywhenprocessingplantdisease
symptomsofvaryingscalesinreal-worldenvironments—wedesignedamulti-scaledepth-
wiseconvolution(MDC)moduleandintegrateditintotheexistingmodelarchitecture.
TheMDCmoduleconsistsofthreeparalleldepthwiseseparableconvolutionbranches
withkernelsizesof3×3,5×5,and7×7. Itskeyfeatureliesinsimultaneouslyprocess-
ingfeatureinformationatmultiplescales,withtheoutputsfusedthroughelement-wise
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 6of23
additiontoproducefeaturemapsenrichedwithmulti-scaleinformation. Tomaintainthe
lightweightdesignprinciplesoftheoriginalShuffleNetV2,weappliedthismoduleselec-
tivelyratherthanuniversallyacrossallconvolutionallayers. Specifically,theMDCmodule
isdeployedexclusivelywithinStage2oftheShuffleNetV2architecture,includingtheinitial
downsamplingunit(stride=2)andthesubsequentbasicunits(stride=1),asillustrated
inFigure3. Stage2featuremapsretainrelativelyhighspatialresolution(56×56),making
this stage ideal for capturing fine pathological details. Moreover, enriching the feature
representationearlyinthenetworkprovideshigherqualityinputstodeeperlayers,thereby
improvingrecognitionperformancewhilecontainingcomputationalcosts.
Figure3.ArchitectureoftheMulti-scaleDepthwiseConvolution(DWConv)moduleforfeatureextraction.
2.2.2. CoordinateAttentionModule
AfterextractingricherlesionfeaturesthroughtheMDCmodule,theabsenceofanat-
tentionmechanismmaycauseimportantlesionfeaturestobedilutedbybackgroundnoise,
reducingdiseaseclassificationaccuracy. Toaddressthis,weintroducedacoordinateatten-
tion(CA)mechanism[21]followingthemulti-scaleconvolutionalmodule. TheCAmodule
helps the model focus its limited computational resources on the most distinguishable
lesionregions. ThearchitectureoftheCAmechanismisillustratedinFigure4.
Figure4.Architectureofthecoordinateattention(CA)modulebasedondirectionalfeatureencoding.
The CA module embeds positional information into channel attention to enhance
feature representation. To avoid spatial information loss caused by traditional two-
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 7of23
dimensionalglobalpooling,CAdecomposestheinputfeaturemapx intotwoseparate
c
one-dimensional feature encodings. Specifically, it performs one-dimensional average
poolingalongthewidth(W)andheight(H)dimensionsforeachchannel(c),producing
height-wise(zh)andwidth-wise(zw)featurerepresentationsthatcapturespatialrelation-
c c
shipsinbothverticalandhorizontaldirections. InFigure4,“XAvgPool”and“YAvgPool”
denotetheaggregationoffeatureinformationacrossallchannels(C)alongtheirrespective
spatial dimensions, resulting in two direction-aware feature tensors (zh and zw). These
conceptsaremathematicallyformalizedinEquations(1)and(2):
zh(h) = 1 ∑ x (h,w), (1)
c W c
0≤w<W
zw(w) = 1 ∑ x (h,w), (2)
c H c
0≤h<H
wherezh andzw representtheone-dimensionalfeaturevaluesobtainedatheight hand
c c
widthwalongchannelc,respectively.
Afterthisstep,toenableeffectivefeatureinformationexchangebetweenthetwodirec-
tions,weperformfeaturefusionbysequentiallyapplyingconvolution,batchnormalization
(BN),andanactivationfunctiontotheone-dimensionalfeatureszh andzw. Theconceptis
showninEquation(3):
f= δ(BN(F ([zh,zw]))), (3)
1
wherefrepresentstheintermediatefeaturemapobtainedbyencodingthexandycoordi-
natefeatures. A1×1convolutionfunctionF isappliedtolearncorrelationsbetweenthe
1
twospatialdimensionswhilereducingthenumberofchannels,therebyloweringcompu-
tationalcomplexity. TheSwishactivationfunction(δ)isthenapplied. Sincethisfeature
map(f)alreadycontainsglobalandcross-dimensionalinformation,splittingitbackinto
separatespatialdimensionsgeneratesindependentattentionmapsforheightandwidth(fh
and fw). Theconvolutionfunctions(F andF ),combinedwiththeactivationfunction(σ),
h w
thenproducetwoattentionweights(gh andgw)thatcapturethespatialdependenciesand
positionalimportanceoffeaturesalongtheheightandwidthdimensions. Theseweights
areusedtoreweighttheoriginalfeaturemapinsubsequentprocessing. Relevantconcepts
areshowninEquations(4)and(5):
gh = σ(F (fh)), (4)
h
gw = σ(F (fw)). (5)
w
Finally,thecomputedheightandwidthattentionweights(gh andgw)areexpanded
and multiplied element-wise with the original input feature map (x ), producing the
c
weightedoutputfeaturemap(y ),asshowninEquation(6):
c
y (h,w) = x (h,w)×gh(h)×gw(w), (6)
c c c c
wherex (h,w)denotesthevalueofthecchannelintheinputfeaturemapatcoordinates
c
(h,w), while y (h,w) represents the corresponding output value. Additionally, gh(h)
c c
representstheweightvalueinthecchannelandhcolumnoftheheightattentionweight
gh,whilegw(w)denotestheweightvalueinthecchannelandwrowofthewidthattention
c
weightgw. Thisoperationenablesthemodeltoautonomouslyenhancefeaturesincritical
regionsbasedonlearnedspatialimportancewhilesuppressingfeaturesinnon-criticalareas.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 8of23
2.2.3. ShuffleNetV2CoreUnitModification
Afterintegratingthemulti-scalefeatureextractionmodulewiththecoordinateatten-
tionmechanism,wedesignedtwomodifiedShuffleNetV2coreunits: abaseunitanda
downsamplingunit. TheirdetailedarchitecturesareillustratedinFigure5. Takingthe
baseunit(Figure5a)asanexample,whenasetoffeaturemapsentersthisunit,itisfirst
splitintotwoparts. Onehalfofthechannelsretainstheoriginalinformationandproceeds
directlytothesubsequentfusionstage,ensuringstablegradientpropagation. Theother
branchentersthefeatureprocessingpath,wherethefeaturemapundergoessequential
1×1convolutionstoexpandthenumberofchannels. Next,theMDCmodulecaptures
spatialfeaturesoflesionsatdifferentscales. Followingthis, theCAmodulefiltersand
weightsthesefeatures,focusingonregionswithhighdiscriminativepower.Finally,thetwo
featurestreamsarefusedthroughconcatenation(Concat)andchannelshuffling(Channel
Shuffle). ThisdesignpreservesShuffleNetV2’shighefficiencywhileenhancingitsabilityto
perceivemulti-scaletargetsandlocalizecriticalregions.
ThedownsamplingunitshowninFigure5bisdeployedatthebeginningofStage2.
Unlikethebaseunit,thisunitperformsstride-2MS-DWConvoperationsonboththeleft
andrightbranchestoachievedownsampling,significantlyreducingthespatialdimensions.
Asubsequent1×1convolutionthenadjuststhenumberofchannelstofacilitatefeature
fusion,effectivelyhalvingthespatialdimensionsattheoutput.
Figure5.ArchitectureofthemodifiedShuffleNetunitbasedonmulti-scaledepthwiseconvolution
andcoordinateattention:(a)baseunitand(b)down-samplingunit.
2.3. KnowledgeDistillation
AlthoughwehaveenhancedtheShuffleNetV2architecturewithmulti-scalefeature
extractionandcoordinateattentionmechanismstobetteradaptitforappleleafdisease
identification, itsfeaturerepresentationcapacityremainslimitedduetoitslightweight
designandreducedchannelcount. Asaresult,itexhibitsaccuracygapscomparedtolarger
models,especiallywhenidentifyingcomplexdiseasesymptoms. Toclosethisperformance
gapwithoutincreasinginferencecosts,weemployknowledgedistillation[22].
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 9of23
Knowledge distillation is an effective model compression and knowledge transfer
techniquethatusesafullytrained,structurallycomplex,high-performanceteachermodel
to guide the training of a smaller, simpler student model with fewer parameters. By
aligningthestudentmodel’soutputswiththeprobabilitydistributionoftheteachermodel’s
outputs,thestudentlearnstoreplicatetheteacher’smappingfrominputdatatooutput
labels,therebyimprovingitsclassificationperformance. GivenResNet50’soutstanding
performanceacrossmultiplevisualtasksandthisdataset[23,24],weselectthepre-trained
ResNet50astheteachermodelinthisstudy. Thetrainingprocessforknowledgedistillation
isillustratedinFigure6.
Figure6. Knowledgedistillationtrainingframeworkandlossfunctioncalculationflowforsmall
modeltraining:(a)teacher–studentmodeloverview,and(b)detailedsoftmaxlosscalculationbased
ontemperature.
Duringtheknowledgedistillationtrainingphase,thestudentmodel’slearningob-
jectiveconsistsoftwoweightedlossfunctions. Thefirstisthetraditionalsupervisedloss
L ,referredtoasthe“StudentLoss”inFigure6b,whichensuresthestudentlearnsto
Student
fitthetruelabels(hardlabels). Thislossiscomputedusingthestandardcross-entropyfunc-
tiontomeasurethedifferencebetweenthestudentmodel’spredictionsσ(z )—processed
s
throughtheSoftmaxfunction(σ)—andthetruelabelT ,asshowninEquation(7):
b
L =CrossEntropy(T ,σ(z )). (7)
Student b s
Thesecondcomponentisthedistillationloss L ,shownasthe“Distillation
Distillation
Loss”inFigure6b. Itspurposeistoalignthestudentmodel’soutputprobabilitydistribu-
tionwiththatoftheteachermodel. Tohelpthestudentlearntheintrinsicrelationships
betweencategories,ratherthanjustthecorrectlabels,atemperatureparameter(Tempera-
ture,T >1)isintroduced. Bydividingthelogitsofboththeteacherandstudentmodels
(denotedasz andz ,respectively)bythistemperature(T)beforeapplyingtheSoftmax
t s
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 10of23
function,asmootherprobabilitydistribution—calledthesoftlabels—isobtained. These
soft labels capture the similarity between different classes as perceived by the teacher.
HighervaluesofTamplifytheprobabilitiesofincorrectclasses,guidingthestudenttopay
attentiontowhichmisclassificationstheteacherconsidersmorelikely. Thisenrichesthe
student’sunderstandingoftheclassstructure. Thedistillationlossiscomputedusingthe
Kullback–Leibler(KL)divergencebetweenthesoftenedoutputsofthestudent(σ(z /T))
s
andteacher(σ(z /T))models,asshowninEquation(8):
t
L = T2×KL(σ(z /T),σ(z /T)), (8)
Distillation s t
wherethecorrectioncoefficientT2 ensuresthatthegradientmagnituderemainsrelatively
stable as the temperature T varies. Ultimately, the total loss function L is formed
total
byaweightedcombinationofthesupervisedlossandthedistillationloss,asshownin
Equation(9):
L = α·L +(1−α)·L . (9)
total Distillation Student
Thehyperparameterαbalancestheweightingbetweenlearningfromtruelabelsand
softlabelsinthestudentmodel. Byminimizingthiscombinedlossfunction,thestudent
modellearnsbothtofitthetruedataandtoemulatetheteachermodel’sdecisionlogic
andclassrelationships. Inpracticalsmartagricultureapplications,knowledgedistillation
effectivelyimprovesrecognitionaccuracywhilemaintainingthelightweightnatureofthe
model. OurgoalistodistillthesuperiorrecognitioncapabilitiesoftheResNet50teacher
modelintoourmodifiedlightweightShuffleNetV2studentmodel.
3. ExperimentalResults
3.1. ExperimentalEnvironment
Our experimental setup includes an 8-core computer powered by an Intel Core i7-
10700 CPU, an NVIDIA GeForce RTX 3080 GPU, and 16 GB of RAM. The system runs
Windows11Professionalastheoperatingsystem. Thedevelopmentenvironmentisbuilt
onAnacondaversion23.7.4withPython3.12.11,utilizingthePyTorch2.7.1deeplearning
frameworkalongwithCUDA12.8andcuDNN9.7.1GPUdrivers.
3.2. ExperimentalDatasets
Toverifythe effectiveness oftheproposed modelanditsgeneralization capability
acrossdifferentenvironments,thisstudyutilizestwolarge-scaleappleleafdiseasedatasets:
PlantPathology2021-FGVC8andAppleLeaf9.
3.2.1. PlantPathology2021-FGVC8Dataset
ThispaperusesthePlantPathology2021-FGVC8dataset[25],whichwasprovided
bytheFine-GrainedVisualCategorizationcompetitionheldatCVPR2021andispublicly
accessibleontheKaggleplatform. Thedatasetconsistsof18,632high-qualityRGBimages
of apple leaves captured under natural agricultural conditions. It represents diverse
lighting,backgrounds,andleafmaturitystages,reflectingreal-worldfieldscenarios. All
imagesareexpert-annotatedintosixclasses: scab,powderymildew,frogeyeleafspot,rust,
complexdisease,andhealthy. The“complex”categorydenotesleavesexhibitingmultiple
symptomssimultaneously,acommonoccurrenceinactualfieldenvironmentsthataddsto
classificationcomplexity. SampleimagesfromthedatasetaredisplayedinFigure7.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151
11of23
Figure7.SampleappleleafimagesfromthePlantPathology2021-FGVC8dataset:(a)scab,(b)pow-
derymildew,(c)frogeyeleafspot,(d)rust,(e)complex,and(f)healthy.
3.2.2. AppleLeaf9Dataset
Tofurthervalidatethestabilityandgeneralizationcapabilityoftheproposedmodel
across different sampling environments, this study introduces the AppleLeaf9 public
dataset [26]. This dataset addresses the challenges of identifying apple leaf diseases in
real-worldfieldenvironments.AppleLeaf9integratesdatafromPVD,PPCD2020,andother
sources,containingimagesofsingleleaves,complexbackgrounds,andmultipledisease
symptoms, making it highly challenging and representative. Table 1 presents the data
distributionforeachcategoryinthisdataset.
Table1.Class-wiseimagecountsanddatasplitsintheAppleLeaf9dataset.
| Types              | ImageNumber | TrainingImg. | ValidationImg. | TestingImg. |
| ------------------ | ----------- | ------------ | -------------- | ----------- |
| Alternarialeafspot | 417         | 292          | 42             | 83          |
| Brownspot          | 411         | 288          | 41             | 82          |
| Frogeyeleafspot    | 3181        | 2227         | 318            | 636         |
| Greyspot           | 339         | 238          | 34             | 67          |
| Health             | 516         | 362          | 51             | 103         |
| Mosaic             | 371         | 260          | 37             | 74          |
| Powderymildew      | 1184        | 829          | 119            | 236         |
| Rust               | 2753        | 1928         | 275            | 550         |
| Scab               | 5410        | 3787         | 541            | 1082        |
| Total              | 14,582      | 10,211       | 1458           | 2913        |
3.3. ExperimentalSetup
3.3.1. ImagePreprocessingandDataAugmentation
Due to the high resolution (e.g., 4000 × 2672 pixels) and variable dimensions of
the original images in the Plant Pathology 2021-FGVC8 dataset, performing dynamic
resizingon-the-flyforeachtrainingbatchcanleadtoI/Obottlenecksandunnecessary
computationaloverhead. Toimprovedataloadingefficiencyandspeedupmodeltraining,
allimageswereuniformlyresizedto384×384pixelspriortotraining.Thedatasetwasthen
randomlydividedintotraining,validation,andtestsetswitha7:1:2ratiotoensurerobust
https://doi.org/10.3390/a19020151

Algorithms2026,19,151
12of23
training,validation,andperformanceevaluation. Thesamplecountsforeachcategoryare
summarizedinTable2.
Table2.Class-wiseimagecountsanddatasplitsinthePlantPathology2021-FGVC8dataset.
| Types           | ImageNumber | TrainingImg. | ValidationImg. | TestingImg. |
| --------------- | ----------- | ------------ | -------------- | ----------- |
| Complex         | 2957        | 2070         | 296            | 591         |
| Frogeyeleafspot | 3181        | 2227         | 318            | 636         |
| Healthy         | 4624        | 3237         | 463            | 924         |
| Powderymildew   | 1184        | 829          | 119            | 236         |
| Rust            | 1860        | 1302         | 186            | 372         |
| Scab            | 4826        | 3379         | 482            | 965         |
| Total           | 18,632      | 13,044       | 1864           | 3724        |
Tofurtherincreasethediversityoftrainingdataandreducemodeloverfitting,we
appliedRandAugment[27]dataaugmentationtothetrainingset. RandAugmentsimplifies
augmentationbyrandomlyselectingafixednumberoftransformations—suchasrotation,
color adjustment, translation, and autocontrast—from a predefined list and applying
themsequentiallytoeachinputimage. Thisprocessgeneratesawidevarietyofaltered
trainingsamples,enhancingdatavariability. Byexposingthemodeltodiversesynthetic
disturbances,RandAugmentimprovesthemodel’sgeneralizationandrobustnesstoreal-
worldvariationslikelightingchangesanddifferentviewingangles.
3.3.2. TrainingParameterSetting
Giventhevariationincomplexityandconvergencebehaviorsacrossdifferentmodel
architectures,weemployedtheReduceLROnPlateaulearningrateschedulerwithadaptive
adjustmentinallexperimentstopromotestableconvergenceduringlatertrainingstages.
Thisstrategymonitorsthevalidationlossandautomaticallyreducesthelearningratewhen
thelossmetricplateausforapredefinednumberofconsecutiveepochs(patience).Bydoing
so,ithelpsaccelerateconvergencebyallowingfineroptimizationstepsonceimprovement
stagnates,therebyimprovingtrainingstabilityandfinalmodelperformance.
Toensurefaircomparisonacrossdifferentlightweightmodelsandmaintainexperi-
mentalconsistency,independenthyperparametertuningforeachmodelwasnotperformed.
Instead,weadoptedwidelyusedhyperparametersettingscommoninlightweightCNN
research,withspecificconfigurationsdetailedinTable3. Forthemulti-classclassification
taskaddressed,weusedthestandardcross-entropylossfunctionasthetrainingobjective.
TheAdamWoptimizerwasselectedduetoitsabilitytoprovidemorestableconvergence
forCNNarchitecturescomparedtotraditionaloptimizerslikeSGDorAdam.
Table3.Thevarioushyperparametersettingusedintheproposedapproach.
| Hyperparameter |     |     | Value/Method |     |
| -------------- | --- | --- | ------------ | --- |
InitialLearningRate 1×10−3
|             | BatchSize |     | 64                |     |
| ----------- | --------- | --- | ----------------- | --- |
| MaxEpochs   |           |     | 150               |     |
| Optimizer   |           |     | AdamW             |     |
| LRScheduler |           |     | ReduceLROnPlateau |     |
LRSchedulerPatience 7
LRReductionFactor 0.1
| EarlyStoppingPatience |     |     | 15            |     |
| --------------------- | --- | --- | ------------- | --- |
| LossFunction          |     |     | Cross-Entropy |     |
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 13of23
Duringtheknowledgedistillationphase,thetrainingobjectivediffersfromlearning
fromscratch. Atthisstage,thestudentmodelnolongerexploresanentirelynewweight
space but instead undergoes fine-tuning guided by a pre-trained teacher model. Con-
sequently, asmallerlearningrateisrequiredduringdistillationcomparedtotheinitial
trainingphasetoensurestablefine-tuning. Tojustifythis,weanalyzedthelearningrate
scheduleduringthestudentmodel’sinitialtraining,asshowninFigure8. Thestudent
model began with an initial learning rate alongside the ReduceLROnPlateau strategy
(patience=7). Whenthelearningratedecreasedatepoch62,validationlossimproved.
However,earlystoppingwithpatiencesetto15wasonlytriggeredatepoch95duringthe
latetrainingphase,bywhichpointthelearningratehaddroppedfurther. Thissuggests
thatlearningratesbelowthisthresholdmaylacksufficientmomentumtomakemeaningful
parameterupdatesorescapelocalminima.
Weultimatelyselectedaninitiallearningratefortheknowledgedistillationphasethat
balancesbetweenthehigherlearningrateusedduringthestudentmodel’sinitialtraining
andthelowerlearningrateobservedwhenthemodel’sperformancebeginstoplateau.
Thischoiceensuresstablefine-tuningwhileretainingsufficientmomentumforeffective
explorationandconvergencetowardanoptimalparameterspaceundertheteachermodel’s
guidance.Asidefromthisadjustmenttothelearningrate,allotherhyperparametersremain
consistentwiththoseusedduringtheinitialtrainingphasetomaintainexperimentalcontrol
overvariability.
Figure8.Theeffectoflearningratescheduleonstudentmodelconvergence:(a)adaptivelearning
ratescheduleviaReduceLROnPlateauand(b)correspondingtrainingandvalidationlosscurves.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 14of23
3.4. ExperimentalResultsandAnalysis
Thissectionpresentsaseriesofexperimentsdesignedtovalidatetheeffectivenessof
ourproposedmethod,alongwithanalysisoftheresults.
• Experiment1establishesthebaselinemodel’sperformanceasafoundationforsubse-
quentimprovementcomparisons.
• Experiment 2 conducts ablation studies to individually verify the contributions of
theMDCmodule, CAattentionmechanism, andknowledgedistillationtooverall
modelperformance.
• Experiment3evaluatestherationaleforselectingtheCAmodulebycomparingits
performanceagainstseveralmainstreamattentionmechanisms.
• Experiment4analyzestheimpactofknowledgedistillationhyperparameters(temper-
ature(T)andweight(α))toidentifytheoptimalparametercombinations.
• Experiments5and6compareourproposedmodelagainstotherleadinglightweight
models and relevant literature to assess its competitiveness in terms of accuracy
andefficiency.
Toquantitativelyevaluatemodelperformance,thispaperemploysaccuracy,precision,
recall, and F1-score as the primary evaluation metrics, with their calculation formulas
presentedinEquations(10)–(13):
TP+TN
Accuracy = , (10)
TP+TN+FP+FN
TP
Precision = , (11)
TP+FP
TP
Recall = , (12)
TP+FN
2×Precision×Recall
F1-score = . (13)
Precision+Recall
Amongtheevaluationmetricsmentioned,TruePositive(TP)meansanactualpositive
sample is correctly predicted as positive, False Positive (FP) means an actual negative
sampleisincorrectlypredictedaspositive,FalseNegative(FN)meansanactualpositive
sample is incorrectly predicted as negative, and True Negative (TN) means an actual
negative sample is correctly predicted as negative. Additionally, we compare models
basedontheirparametercountsandcomputationalcomplexity,measuredinFloatingPoint
Operations(FLOPs),toassessefficiencyalongsidepredictiveperformance.
3.4.1. BaselineModelPerformance
First, we established the baseline model using ShuffleNetV2 0.5×. Without pre-
trainedweightsordataaugmentation,thislightweightmodelachievedanaccuracyofonly
88.08%. IncorporatingImageNetpre-trainedweightsimprovedtheaccuracyto90.12%,
demonstrating the benefits of transfer learning. Further applying RandAugment data
augmentationraisedtheaccuracyto90.84%,indicatingthatthisaugmentationstrategy
effectivelyenhancesthemodel’sgeneralizationability.
Tosimplifythepresentationofsubsequentexperimentaldata,thisstudyreferstothe
original ShuffleNetV2 0.5× model as SN-Base, the version with pre-trained weights as
SN-TL(TransferLearning),andthemodeladditionallyincorporatingdataaugmentation
as SN-DA (Data Augmentation). Since SN-DA demonstrates superior performance, it
servesasthebaselineforcomparisonsinthesubsequentexperiments. Additionalrelevant
performancemetricsarepresentedinTable4.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151
15of23
Table4.EffectoftransferlearninganddataaugmentationonShuffleNet-basedmodelperformance.
|     | Model           |     | Accuracy(%) | Precision(%) |     | Recall(%) | F1-Score(%) |
| --- | --------------- | --- | ----------- | ------------ | --- | --------- | ----------- |
|     | SN-Base         |     | 88.08       | 86.91        |     | 87.03     | 86.87       |
|     | SN-TL           |     | 90.12       | 88.84        |     | 89.11     | 88.87       |
|     | SN-DA(Baseline) |     | 90.84       | 89.74        |     | 89.97     | 89.65       |
3.4.2. AblationStudy
To validate the specific contributions of each modified module in this study, we
conductedablationexperimentsbyprogressivelyaddingtheMDCmodule,CAmodule,
andknowledgedistillation(KD)strategyontopofthebaselinemodel. Theresultsofthese
experimentsarepresentedinTable5.
Table5.PerformanceContributionofIntegratedComponents.
Architecture Acc.(%) Pre.(%) Rec.(%) F1-Score(%) Params(M) FLOPs(G)
| Baseline        |     | 90.84 | 89.74 | 89.97 | 89.65 | 0.3479 | 0.0789 |
| --------------- | --- | ----- | ----- | ----- | ----- | ------ | ------ |
| Baseline+MDC    |     | 91.43 | 90.42 | 90.33 | 90.29 | 0.3568 | 0.0929 |
| Baseline+CA     |     | 91.92 | 90.87 | 90.96 | 90.91 | 0.3512 | 0.0792 |
| Baseline+KD     |     | 92.29 | 90.92 | 91.56 | 91.18 | 0.3479 | 0.0789 |
| Baseline+MDC+CA |     | 92.64 | 91.48 | 91.52 | 91.45 | 0.3601 | 0.0931 |
| Baseline+MDC+KD |     | 92.37 | 91.10 | 91.37 | 91.20 | 0.3568 | 0.0929 |
| Baseline+CA+KD  |     | 92.88 | 91.76 | 92.23 | 91.82 | 0.3512 | 0.0792 |
Baseline+MDC+CA+KD(Ours) 93.15 91.95 92.20 92.04 0.3601 0.0931
First,theMDCmodulewasintegratedintoStage2ofthebaselinemodel.Experimental
resultsshowthataddingtheMDCmoduleincreasesthemodel’saccuracyto91.43%. This
demonstratesthattheMDCmoduleenhancesthemodel’sabilitytocapturefeaturesat
multiplespatialscales,therebyimprovingrecognitionperformance.
Subsequently,buildingontheintegrationoftheMDCmodule,weincorporatedthe
CAmodule. ExperimentalresultsshowthataddingtheCAmoduleincreasedaccuracyto
92.64%,demonstratingitseffectivenessinenhancingfeatureselectionandweighting. The
CAmoduleguidesthemodel’sattentiontowarddiagnosticallyimportantregions,thereby
minimizingtheimpactofbackgroundnoiseandirrelevantfactors.
Finally,knowledgedistillation(KD)wasappliedtothemodelintegratingMDCand
CA(Baseline+MDC+CA)toenhancetherecognitionperformanceandgeneralization
ofthelightweightstudentmodel. Experimentalresultsshowthataccuracyincreasedto
93.15%,alongwithimprovementsinPrecision,Recall,andF1-Score. Thesefindingsdemon-
stratethatknowledgedistillationisaneffectiveoptimizationtechniqueforimprovingthe
performanceoflightweightstudentmodels.
Furthermore,thecomprehensiveablationstudycomprisingeightexperimentalcon-
figurations,asshowninTable5,revealsthesynergisticeffectsbetweenthemodules. The
resultsindicatethatwhileindividualmodulesprovideincrementalgains,theircombination
achievesahigherperformanceceiling. Notably,theintroductionofknowledgedistillation
significantlyimprovesaccuracywithoutincreasinginferencelatency,demonstratingthat
transferring knowledge from a high-capacity teacher model is crucial for enabling the
lightweightmodeltocapturesubtlepathologicalfeatures.
3.4.3. ComparativeAnalysisofAttentionMechanisms
TovalidatetheperformanceoftheCAmodule,weconductedacomparativeanalysis
withfourothermainstreamattentionmechanisms: CBAM[28],ECA[29],SA(ShuffleAt-
tention)[30],andMCA(MultidimensionalCollaborativeAttention)[31].Alltheseattention
https://doi.org/10.3390/a19020151

Algorithms2026,19,151
16of23
modulesweretrainedandevaluatedwithintheBaseline+MDC(B-MDC)architecture.
RelevantexperimentalresultsaresummarizedinTable6.
Table6.PerformancecomparisonofB-MDCwithvariousattentionmodules.
AttentionModule Acc.(%) Pre.(%) Rec.(%) F1-Score(%) Params(M) FLOPs(G)
| B-MDC      | 91.43 | 90.42 | 90.33 | 90.29 | 0.3568 | 0.0929 |
| ---------- | ----- | ----- | ----- | ----- | ------ | ------ |
| B-MDC+CBAM | 92.53 | 91.45 | 91.55 | 91.46 | 0.3626 | 0.0936 |
| B-MDC+ECA  | 92.27 | 91.22 | 91.40 | 91.23 | 0.3568 | 0.0930 |
| B-MDC+SA   | 92.00 | 91.09 | 91.14 | 91.02 | 0.3570 | 0.0930 |
| B-MDC+MCA  | 91.76 | 90.53 | 90.71 | 90.49 | 0.3569 | 0.0930 |
| B-MDC+CA   | 92.64 | 91.48 | 91.52 | 91.45 | 0.3601 | 0.0931 |
Experimental results demonstrate that compared to the B-MDC model without at-
tentionmechanisms(withanaccuracyof91.43%),alltestedattentionmodulesproduced
varying degrees of performance improvement, clearly showcasing the effectiveness of
attentionmechanismsinemphasizingrelevantfeaturesforplantleafdiseaseidentification.
Amongthem, themodelwiththeCAmoduleattainedthehighestaccuracy. Moreover,
comparedtothecomputationallyheavierCBAM,CAachievedbetterperformancewith
only a negligible increase in computational complexity and parameter count over the
B-MDCmodelwithoutattention. ThishighlightsCA’sabilitytobalanceaccuracygains
withefficiency,validatingitsselectionastheattentionmechanisminthisstudy.
|     | 3.4.4. | HyperparameterAnalysisforKnowledgeDistillation |     |     |     |     |
| --- | ------ | ---------------------------------------------- | --- | --- | --- | --- |
The effectiveness of knowledge distillation depends on two key hyperparameters:
temperature(T)andthelossweightingcoefficient(α). Tofindtheoptimalcombination,
this experiment performed a grid search on the validation set, testing various values
of temperature T = {4,8,10} and weight α = {0.3,0.5,0.7,0.9}. The corresponding
experimentalresultsarepresentedinTable7.
Table7.Effectofknowledgedistillationhyperparameters(αandT)onmodelperformance(in%).
|     |     | α\T | 4     | 8     |     | 10    |
| --- | --- | --- | ----- | ----- | --- | ----- |
|     |     | 0.3 | 92.86 | 93.56 |     | 93.88 |
|     |     | 0.5 | 93.29 | 93.56 |     | 93.78 |
|     |     | 0.7 | 93.29 | 93.62 |     | 93.67 |
|     |     | 0.9 | 93.67 | 93.40 |     | 92.86 |
Experimentaldatarevealasignificantinteractionbetweentemperature(T)andloss
weightingcoefficient(α)inknowledgedistillation.Whenαislow,themodelprimarilyrelies
onhardlabels,andincreasingTallowsthemodeltolearnrichercategoricalrelationship
informationfromthesoftlabels,resultinginperformancegains. However,thistrenddoes
not always hold. At higher α values, increasing T can cause accuracy to decline, likely
duetooverlysmoothedsoftlabelscombinedwithhighimitationweights. Thismaylead
the student model to overfit the teacher’s output distribution while neglecting crucial
informationfromthehardlabels.
TheexperimentalresultsinTable7showthatthecombinationoftemperatureT =10
andweightα =0.3achievesthehighestaccuracyof93.88%onthevalidationset. Thisset-
tingeffectivelyleveragestheinter-classinformationcontainedinsoftlabelswhileensuring
themodelfullylearnsaccurateinformationfromhardlabelsthroughtherelativelylowα,
achievinganoptimalbalance. Therefore,thishyperparameterconfigurationisadopted.
Additionally,Figure9illustratestheaccuracytrendsonthetrainingandvalidationsets
https://doi.org/10.3390/a19020151

Algorithms2026,19,151
17of23
underthisconfigurationduringknowledgedistillation. Thepre-trainedstudentmodel
startedwithaninitialaccuracyaround90.0%. Withinthefirst30epochs,validationaccu-
racyexhibitedrapidgrowthwithsomefluctuations,whiletrainingaccuracyincreasedmore
smoothly,reflectingthestudent’slearningandadaptationfromtheteacher. Eventually,
bothaccuraciesconvergedaround93.3%to93.5%,withanarrowgapindicatingminimal
overfitting. Once the validation performance plateaued, early stopping triggered after
15consecutiveepochswithoutimprovement,terminatingtrainingatepoch63.
Figure9.Trainingandvalidationaccuracycurvesofthemodelunderknowledgedistillationhyper-
parameters(α=0.3andT=10).
3.4.5. PerformanceComparisonwithOtherLightweightModels
Tovalidatethecomprehensiveperformanceandefficiencyadvantagesoftheproposed
model,wecompareditwithseveralmainstreamlightweightandmedium-weightmodels.
TheseincludeSqueezeNet1.1[32],knownforitsefficiency;MobileNetV4-Conv-Small[33];
GhostNetV2 1.0x [34]; various scales of ShuffleNetV2 (including the baseline 0.5× and
larger1.0×and1.5×versions);thehigh-performingEfficientNetV2-B0[35];andthewidely
usedResNetseries,includingResNet34dandResNet50astheteachermodels. Relevant
experimentalresultsaresummarizedinTable8.
Table8.Performanceandefficiencycomparisonof’Ours’withstate-of-the-artdeeplearningmodels.
Method Acc.(%) Pre.(%) Rec.(%) F1-Score(%) Params(M) FLOPs(G)
| ResNet34d              | 94.58 | 93.62 | 93.70 | 93.63 | 21.3070 | 7.8058 |
| ---------------------- | ----- | ----- | ----- | ----- | ------- | ------ |
| ResNet50               | 94.76 | 93.61 | 93.99 | 93.77 | 23.5203 | 8.1744 |
| GhostNetV21.0          | 91.97 | 90.37 | 91.08 | 90.66 | 4.8836  | 0.3317 |
| SqueezeNet1.1          | 92.53 | 91.26 | 91.48 | 91.31 | 0.7256  | 0.5311 |
| MobileNetV4-Conv-Small | 88.61 | 87.18 | 87.99 | 87.03 | 2.5007  | 0.7514 |
| EfficientNetV2-B0      | 93.56 | 92.53 | 92.72 | 92.61 | 5.8664  | 1.4337 |
| ShuffleNetV20.5×       | 90.84 | 89.74 | 89.97 | 89.65 | 0.3479  | 0.0789 |
| ShuffleNetV21.0×       | 93.26 | 91.98 | 91.96 | 91.90 | 1.2598  | 0.2878 |
ShuffleNetV21.5×
|      | 92.72 | 91.82 | 91.59 | 91.69 | 2.4848 | 0.5895 |
| ---- | ----- | ----- | ----- | ----- | ------ | ------ |
| Ours | 93.15 | 91.95 | 92.20 | 92.04 | 0.3601 | 0.0931 |
Experimentalresultsdemonstratethattheproposedmethodsignificantlyoutperforms
mostlightweightmodelsintermsofbothparametercountandcomputationalcomplexity.
Forexample,itusesapproximatelyhalftheparametersofSqueezeNet1.1andhassubstan-
tiallylowercomputationalcomplexitythanMobileNetV4-Conv-SmallandGhostNetV2
1.0. WhileitsaccuracyisslightlylowerthanthatofEfficientNetV2-B0,itrequiresfarfewer
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 18of23
parametersandcomputationalresources,showcasingremarkablecomputationalefficiency.
ComparedtothelargerShuffleNetV21.0×model,ourapproachachievesonly0.11%lower
accuracybutreducesparametersandcomputationalcomplexitybyabout71.4%and67.7%,
respectively. Interestingly,theShuffleNetV21.5×modelperformsworsethanboththe1.0×
variantandourmodel,likelyduetooverfittingcausedbytherelativelylimiteddataset
size compared to its model capacity. Against the teacher model ResNet50, our model
usesmerely1.5%oftheparametersand1.1%ofthecomputationalcost,whilenarrowing
the accuracy gap to 1.61%. These findings affirm that our method competes favorably
withmorecomplexmodels,balancingstrongrecognitionperformanceandcomputational
efficiencyatverylowresourceconsumption.
3.4.6. GeneralizationAnalysisontheAppleLeaf9Dataset
Toverifythestabilityofthemodelinreal-worldscenarios,weconductedcomparative
experimentsontheAppleLeaf9dataset.AsshowninTable9,theimprovedmodelproposed
inthisstudyalsoachievedexcellentperformanceonthisdataset.
Table9.PerformanceComparisonwithRepresentativeModelsonAppleLeaf9Dataset.
Method Acc.(%) Pre.(%) Rec.(%) F1-Score(%) Params(M) FLOPs(G)
ResNet34d 97.97 97.05 96.60 96.82 21.3085 7.8058
ResNet50 98.39 97.61 97.67 97.60 23.5265 8.1744
GhostNetV21.0 97.46 96.12 96.20 96.12 4.8874 0.3317
SqueezeNet11.1 97.46 96.68 95.37 96.00 0.7271 0.5316
MobileNetV4-Conv-Small 93.17 90.94 87.81 89.20 2.5046 0.7514
EfficientNetV2-B0 98.18 97.47 96.83 97.10 5.8702 1.4337
ShuffleNetV20.5× 95.67 93.61 92.13 92.82 0.3510 0.0789
ShuffleNetV21.0× 97.80 97.16 96.78 96.97 1.2628 0.2878
ShuffleNetV21.5× 98.15 97.35 97.25 97.29 2.4878 0.5895
Ours 97.85 97.62 97.37 96.93 0.3631 0.0931
4. Discussion
To thoroughly examine the effectiveness of the proposed model and explore the
sourcesofitsadvantagesoverthebaseline,thissectionpresentsexperimentalresultsusing
confusionmatricesandGrad-CAM++visualizationsfordetaileddiscussion.
AsshownintheconfusionmatrixinFigure10,ourproposedmethoddemonstrates
strongalignmentbetweenpredictedcategorydistributionsandactuallabels. Mostsamples
cluster along the main diagonal, indicating a high agreement between predictions and
truelabels. However,someconfusionremainsbetweencertaincategories. Forexample,
sampleslabeledas“complex”areoftenmisclassifiedasfrogeyeleafspotorrust. Sincethe
“complex”categoryrepresentsleavesexhibitingmultiplediseasesymptoms,itsvisualfea-
turesarediverseandcancloselyresemblesingle-diseasesymptoms,blurringclassification
boundaries. Therefore,forsuchmixed-featuresamples,itismoreappropriatetoclassify
themaccordingtothemostprominentsinglediseasepresent.
Toanalyzethereasonsbehindthemodel’sperformanceimprovement,weemployed
Grad-CAM++[36]tovisualizethecontributionsofdifferentimageregionstoclassification
outcomes. Figure11highlightstwocases. Inthefirstcase(truelabel: rust),thebaselineSN-
DAmodelincorrectlypredictsthesampleas“complex,”withitsheatmapfocusingmainly
on leafedges andbackgroundareas. Conversely, ourmodel accuratelypredicts“rust,”
concentratingitsattentionpreciselyontherustlesionattheleaf’sleadingedge. Inthe
secondcase(truelabel: scab),subtleearlysymptomsresemblehealthyleaves,leadingthe
baselinemodeltomisclassifyitashealthy,withattentionfocusedonhealthyleafregions.
Ourmodel,however,successfullydetectsthesesubtlepathologicalfeatures,correctlyclas-
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 19of23
sifyingtheleafasscab,withtheheatmapaccuratelypinpointingthelesionlocation. These
resultsdemonstratethatourmethodpossessessuperiorsymptomlocalizationcapabilities,
effectivelysuppressingbackgroundnoiseandenhancingrecognitionaccuracy.
Figure10.ConfusionmatrixoftheproposedmodelonthePlantPathology2021-FGVC8testingset.
Figure11. VisualanalysisofGrad-CAM++visualizationsbetweenthebaselinemodelandours:
(a)originalimages,(b)baselinemodel,and(c)proposedmodel.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 20of23
Figure12presentsthevisualizationresultswhenbothmodelspredictcorrectly. The
findingsrevealthatevenwhendifferentmethodsyieldidenticalpredictions,theirregions
offocusexhibitsignificantdifferences. Inthefrogeyeleafspotcase,ourmodeldemon-
stratesmoreaccurateandfocusedattentioncomparedtothebaselinemodel. Theheatmap
clearlyandpreciselymarkstheexactlocationoffrogeyeleafspotsymptoms,eliminating
interferencefromirrelevantbackgroundareas. Incontrast,theheatmapgeneratedbythe
baselinemodelforthepowderymildewcaseshowsalarge,broad,andnon-specificarea,
failingtopinpointthelocationofsymptomoccurrence. Incontrast,theheatmapsgenerated
byourmethodnarrowedandconcentratedonthecoreareasofpowderymildewlesions
ontheleaves. Thesevisualizationsvalidatethatourapproachcanlearnmorepreciseand
relevantexpressionsoflesionfeatures. Evenoncorrectlyclassifiedsamples,thefeature
saliencyremainsmorestableandreliable.
Figure12. VisualanalysisofGrad-CAM++visualizations,demonstratingtheimprovedattention
focusofourproposedmodel:(a)originalimages,(b)baselinemodel’sattention,and(c)proposed
model’sattention.
Intheconfusionmatrixanalysis,significantcross-classificationerrorsexistbetween
thescabandhealthycategories. Figure13illustratesthevisualsimilaritybetweenimages
classifiedasscabandhealthy. Itisobservedthatscabexhibitsnoobvioussymptomson
leafsurfacesduringtheearlystagesofinfection,oronlyslightcolorchangesattheleaftips.
Mostoftheaffectedarearemainsvisuallyindistinguishablefromhealthyleaves,leading
themodeltomisclassifythesesamples.Thisvisualindistinguishabilityislikelytheprimary
causeofcurrentclassificationerrors.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 21of23
Figure13.Visualanalysisof‘Scab’and‘Healthy’misclassifications:(a)originalimages,(b)baseline
model,and(c)proposedmodel.
5. Conclusions
Amidtheriseofsmartagriculture,deeplearninghasbecomeadominanttechnology
forautomated,real-timeplantdiseaseidentification. However,high-performancemod-
elstypicallydemandsubstantialcomputationalresources,limitingtheirdeploymenton
resource-constrained edge devices. To address this, this paper proposes a lightweight,
modified architecture based on ShuffleNetV2 0.5×. It incorporates a multi-scale deep
convolutionalmoduletocapturesymptomsacrossscales,combinesacoordinateattention
mechanismtofocusonkeyvisualfeatures,andemploysknowledgedistillationtotransfer
knowledge from a large teacher model. This design significantly enhances the student
model’srecognitionperformancewhilemaintainingcomputationalefficiency.
Althoughourapproacheffectivelyclassifiesmostdiseasecategories,severallimita-
tionsremain. First,theconfusionmatrixanalysisindicatesthatthemodel’srecognition
performancecanimprovewhenhandlingcategorieswithmixedsymptoms,suchasthe
complexcategory. Second,validationwasprimarilyperformedonasinglepublicdataset;
futureevaluationsshouldincludemorediversedatasetsencompassingvaryinggeographic
regions,imagingconditions,andapplevarietiestothoroughlyassessgeneralizationcapabil-
ities. Basedontheseinsights,futureresearchdirectionsinclude: (1)developingspecialized
optimizationstrategiesforthecomplexcategory,suchasmulti-labelclassification,atten-
tion mechanisms tailored to overlapping features, or multi-task learning architectures;
(2)deployingthemodelonactualedgecomputingdevicesforfieldtestingandexploring
advancedmodelcompressiontomeetstricthardwareconstraints;and(3)integratingob-
jectdetectionorsegmentationtechniquestoenablemorepreciselesionlocalizationand
severityassessment.
AuthorContributions: Conceptualization,W.-C.L.;methodology,W.-C.L.;software,W.-C.L.;val-
idation, W.-C.L.; formalanalysis, W.-C.L.; investigation, W.-C.L.; resources, W.-C.L.andC.-C.L.;
datacuration,W.-C.L.;writing—originaldraftpreparation,W.-C.L.;writing—reviewandediting,
W.-C.L.andC.-C.L.;visualization,W.-C.L.andC.-C.L.;supervision,C.-C.L.;projectadministration,
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 22of23
C.-C.L.;fundingacquisition,C.-C.L.Allauthorshavereadandagreedtothepublishedversionof
themanuscript.
Funding:ThisworkispartiallysupportedbytheNationalScienceandTechnologyCouncil,Taiwan,
R.O.C.,underGrantNSTC114-2221-E-390-007.
DataAvailabilityStatement:Thedatapresentedinthisstudyareopenlyavailableinthefollowing
repositories: (1) Plant Pathology 2021-FGVC8 dataset on Kaggle at: https://www.kaggle.com/
competitions/plant-pathology-2021-fgvc8(accessedon2December2025).(2)AppleLeaf9dataseton
GitHubathttps://github.com/JasonYangCode/AppleLeaf9(accessedon2December2025).
ConflictsofInterest:Theauthorsdeclarenoconflictsofinterest.
References
1. Thapa,R.;Zhang,K.;Snavely,N.;Belongie,S.;Khan,A.ThePlantPathologyChallenge2020DataSettoClassifyFoliarDiseaseof
Apples.Appl.PlantSci.2020,8,e11390.[CrossRef]
2. Jafar,A.;Bibi,N.;Naqvi,R.A.;Sadeghi-Niaraki,A.;Jeong,D.RevolutionizingAgriculturewithArtificialIntelligence: Plant
DiseaseDetectionMethods,Applications,andTheirLimitations.Front.PlantSci.2024,15,1356260.[CrossRef]
3. Prasad,S.;Peddoju,S.K.;Ghosh,D.Multi-ResolutionMobileVisionSystemforPlantLeafDiseaseDiagnosis.SignalImageVideo
Process.2016,10,379–388.[CrossRef]
4. Chuanlei, Z.; Shanwen, Z.; Jucheng, Y.; Yancui, S.; Jia, C. Apple Leaf Disease Identification Using Genetic Algorithm and
CorrelationBasedFeatureSelectionMethod.Int.J.Agric.Biol.Eng.2017,10,74–83.
5. Sabrol,H.;Satish,K.TomatoPlantDiseaseClassificationinDigitalImagesUsingClassificationTree.InProceedingsofthe2016
InternationalConferenceonCommunicationandSignalProcessing,Melmaruvathur,India,6–8April2016;pp.1242–1246.
6. Al Bashish, D.; Braik, M.; Bani-Ahmad, S. A Framework for Detection and Classification of Plant Leaf and Stem Diseases.
InProceedingsofthe2010InternationalConferenceonSignalandImageProcessing,Chennai,India,15–17December2010;
pp.113–118.
7. Anand,R.;Veni,S.;Aravinth,J.AnApplicationofImageProcessingTechniquesforDetectionofDiseasesonBrinjalLeavesUsing
K-MeansClusteringMethod.InProceedingsofthe2016InternationalConferenceonRecentTrendsinInformationTechnology,
Chennai,India,8–9April2016;pp.1–6.
8. Kumar,A.;Nelson,L.;Gomathi,S.TransferLearningofVGG19fortheClassificationofAppleLeafDiseases.InProceedingsof
the20242ndInternationalConferenceonIntelligentDataCommunicationTechnologiesandInternetofThings,Bengaluru,India,
4–6January2024;pp.1643–1648.
9. Sulistyowati,T.;Purwanto,P.;Zami,F.A.;Pramunendar,R.A.VGG16DeepLearningArchitectureUsingImbalanceDataMethods
fortheDetectionofAppleLeafDiseases. Monet.J.Keuang.DanPerbank.2023,11,41–53.[CrossRef]
10. Chen,Y.;Pan,J.;Wu,Q. AppleLeafDiseaseIdentificationviaImprovedCycleGANandConvolutionalNeuralNetwork. Soft
Comput.2023,27,9773–9786.[CrossRef]
11. Bi,C.;Wang,J.;Duan,Y.;Fu,B.;Kang,J.R.;Shi,Y.MobileNetBasedAppleLeafDiseasesIdentification.Mob.Netw.Appl.2022,27,
172–180.[CrossRef]
12. Chao,X.;Sun,G.;Zhao,H.;Li,M.;He,D. IdentificationofAppleTreeLeafDiseasesBasedonDeepLearningModels. Symmetry
2020,12,1065.[CrossRef]
13. Yu,H.;Cheng,X.;Chen,C.;Heidari,A.A.;Liu,J.;Cai,Z.;Chen,H.AppleLeafDiseaseRecognitionMethodwithImproved
ResidualNetwork.Multimed.ToolsAppl.2022,81,7759–7782.[CrossRef]
14. Fu,B.;Li,S.;Sun,Y.;Mu,Y.;Hu,T.;Gong,H.Lightweight-ConvolutionalNeuralNetworkforAppleLeafDiseaseIdentification.
Front.PlantSci.2022,13,831219.[CrossRef][PubMed]
15. Yu,H.J.;Son,C.H.LeafSpotAttentionNetworkforAppleLeafDiseaseIdentification.InProceedingsoftheIEEE/CVFConference
onComputerVisionandPatternRecognitionWorkshops,Seattle,WA,USA,14–19June2020;pp.52–53.
16. Luo,Y.;Sun,J.;Shen,J.;Wu,X.;Wang,L.;Zhu,W.AppleLeafDiseaseRecognitionandSub-ClassCategorizationBasedon
ImprovedMulti-ScaleFeatureFusionNetwork.IEEEAccess2021,9,95517–95527.[CrossRef]
17. Si,H.;Li,M.;Li,W.;Zhang,G.;Wang,M.;Li,F.;Li,Y.ADual-BranchModelIntegratingCNNandSwinTransformerforEfficient
AppleLeafDiseaseClassification.Agriculture2024,14,142.[CrossRef]
18. AitNasser,A.;Akhloufi,M.A.AHybridDeepLearningArchitectureforAppleFoliarDiseaseDetection.Computers2024,13,116.
[CrossRef]
19. Wang,S.;Miao,Z.;Cao,Y.ALightweightLeafDiseaseRecognitionModelforMobileNetV2BasedonTransferLearningand
KnowledgeDistillation.InProceedingsofthe20246thInternationalConferenceonCommunications,InformationSystemand
ComputerEngineering,Guangzhou,China,10–12May2024;pp.459–462.
https://doi.org/10.3390/a19020151

Algorithms2026,19,151 23of23
20. Ma,N.;Zhang,X.;Zheng,H.T.;Sun,J.ShuffleNetV2:PracticalGuidelinesforEfficientCNNArchitectureDesign.InProceedings
ofthe15thEuropeanConferenceonComputerVision;LectureNotesinComputerScience;Springer:Berlin/Heidelberg,Germany,
2018;Volume11218,pp.122–138.
21. Hou,Q.;Zhou,D.;Feng,J.CoordinateAttentionforEfficientMobileNetworkDesign.InProceedingsofthe2021IEEE/CVF
ConferenceonComputerVisionandPatternRecognition,Nashville,TN,USA,19–25June2021;pp.13708–13717.
22. Hinton,G.;Vinyals,O.;Dean,J.DistillingtheKnowledgeinaNeuralNetwork.arXiv2015,arXiv:1503.02531.[CrossRef]
23. Wang,Y.;Wang,Y.;Zhao,J. MGA-YOLO:ALightweightOne-StageNetworkforAppleLeafDiseaseDetection. Front.PlantSci.
2022,13,927424.
24. Chang, D.; Tong, Y.; Du, R.; Hospedales, T.; Song, Y.Z.; Ma, Z. An Erudite Fine-Grained Visual Classification Model. In
Proceedingsofthe2023IEEE/CVFConferenceonComputerVisionandPatternRecognition,Vancouver,BC,Canada,17–24June
2023;pp.7268–7277.
25. PlantPathology2021-FGVC8. Availableonline:https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8(accessed
on2December2025).
26. Yang,Q.;Duan,S.;Wang,L. EfficientIdentificationofAppleLeafDiseasesintheWildUsingConvolutionalNeuralNetworks.
Agronomy2022,12,2784.[CrossRef]
27. Cubuk,E.D.;Zoph,B.;Shlens,J.;Le,Q.V. Randaugment:PracticalAutomatedDataAugmentationwithaReducedSearchSpace.
InProceedingsofthe2020IEEE/CVFConferenceonComputerVisionandPatternRecognitionWorkshops,Seattle,WA,USA,
14–19June2020;pp.3008–3017.
28. Woo,S.;Park,J.;Lee,J.Y.;Kweon,I.S.CBAM:ConvolutionalBlockAttentionModule. InProceedingsofthe15thEuropeanConference
onComputerVision;LectureNotesinComputerScience;Springer:Berlin/Heidelberg,Germany,2018;Volume11211,pp.3–19.
29. Wang,Q.;Wu,B.;Zhu,P.;Li,P.;Zuo,W.;Hu,Q. ECA-Net:EfficientChannelAttentionforDeepConvolutionalNeuralNetworks.
InProceedingsofthe2020IEEE/CVFConferenceonComputerVisionandPatternRecognition,Seattle,WA,USA,13–19June
2020;pp.11531–11539.
30. Zhang,Q.L.;Yang,Y.B. SA-Net:ShuffleAttentionforDeepConvolutionalNeuralNetworks. InProceedingsofthe2021IEEE
InternationalConferenceonAcoustics,SpeechandSignalProcessing,Toronto,ON,Canada,6–11June2021;pp.2235–2239.
31. Yu,Y.;Zhang,Y.;Cheng,Z.;Song,Z.;Tang,C. MCA:MultidimensionalCollaborativeAttentioninDeepConvolutionalNeural
NetworksforImageRecognition. Eng.Appl.Artif.Intell.2023,126,107079.[CrossRef]
32. Iandola,F.N.;Han,S.;Moskewicz,M.W.;Ashraf,K.;Dally,W.J.;Keutzer,K.SqueezeNet:AlexNet-LevelAccuracywith50xFewer
Parametersand<0.5MBModelSize. arXiv2016,arXiv:1602.07360.
33. Qin,D.;Leichner,C.;Delakis,M.;Fornoni,M.;Luo,S.;Yang,F.;Howard,A. MobileNetV4: UniversalModelsfortheMobile
Ecosystem. InProceedingsofthe18thEuropeanConferenceonComputerVision; LectureNotesinComputerScience; Springer:
Berlin/Heidelberg,Germany,2025;Volume15098,pp.78–96.
34. Tang,Y.;Han,K.;Guo,J.;Xu,C.;Xu,C.;Wang,Y. GhostNetV2:EnhanceCheapOperationwithLong-RangeAttention. Adv.
NeuralInf.Process.Syst.2022,35,9969–9981.
35. Tan,M.;Le,Q.EfficientNetV2:SmallerModelsandFasterTraining.InProceedingsofthe38thInternationalConferenceonMachine
Learning;PMLR139;MLResearchPress:CambridgeMA,USA,2021;pp.10096–10106.
36. Chattopadhay,A.;Sarkar,A.;Howlader,P.;Balasubramanian,V.N. Grad-CAM++:GeneralizedGradient-BasedVisualExplana-
tionsforDeepConvolutionalNetworks. InProceedingsofthe2018IEEEWinterConferenceonApplicationsofComputerVision,
LakeTahoe,NV,USA,12–15March2018;pp.839–847.
Disclaimer/Publisher’sNote: Thestatements, opinionsanddatacontainedinallpublicationsaresolelythoseoftheindividual
author(s)andcontributor(s)andnotofMDPIand/ortheeditor(s).MDPIand/ortheeditor(s)disclaimresponsibilityforanyinjuryto
peopleorpropertyresultingfromanyideas,methods,instructionsorproductsreferredtointhecontent.
https://doi.org/10.3390/a19020151