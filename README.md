Paper & Code 
========================
**Works for deep model intellectual property (IP) protection.**


# Contents 
+ [Survey](#Survey)
+ [Infringement](#Infringement)
  + [White-box&nbsp;model-dependent](#White-box&nbsp;model-dependent) | [Black-box](#Black-box)

+ [Access&nbsp;Control](#Access&nbsp;Control)
    + [User&nbsp;Authentication](#User&nbsp;Authentication)
      + [Software-level](#Software-level) | [Hardware-level](#Hardware-level) 
    + [Model&nbsp;Encryption](#Model&nbsp;Encryption)
      + [Encrpted&nbsp;Data](#Encrpted&nbsp;Data) | [Encrpted&nbsp;Architecture](#Encrpted&nbsp;Architecture) | [Encrpted&nbsp;Weights](#Encrpted&nbsp;Weights)

+ [DNN&nbsp;Watermarking&nbsp;Mechanism](#DNN&nbsp;Watermarking&nbsp;Mechanism)
+ [White-box&nbsp;DNN&nbsp;Watermarking](#White-box&nbsp;DNN&nbsp;Watermarking)
    + [First&nbsp;Attempt](#First&nbsp;Attempt)
    + [Improvement](#Improvement)
      + [Loss&nbsp;Constrains&nbsp;|&nbsp;Verification&nbsp;Approach&nbsp;|&nbsp;Training&nbsp;Strategies](#Loss&nbsp;Constrains&nbsp;|&nbsp;Verification&nbsp;Approach&nbsp;|&nbsp;Training&nbsp;Strategies) 
    + [Approaches&nbsp;Based&nbsp;on&nbsp;Muliti-task&nbsp;Learning](#Approaches&nbsp;Based&nbsp;on&nbsp;Muliti-task&nbsp;Learning) 

+ [Black-box&nbsp;DNN&nbsp;Watermarking&nbsp;(Input-output&nbsp;Style)](#Data-based&nbsp;DNN&nbsp;Watermarking&nbsp;(Input-output&nbsp;Style))
    + [Unrelated&nbsp;Trigger](#Unrelated&nbsp;Trigger)
    + [Related&nbsp;Trigger](#Related&nbsp;Trigger)
      + [Adversarial&nbsp;Examples](#Adversarial&nbsp;Examples)
    + [Related&unrelated](#Related&unrelated)
    + [Clean&nbsp;image&nbsp;&&nbsp;Wrong&nbsp;Label](#Clean&nbsp;image&nbsp;&&nbsp;Wrong&nbsp;Label)
+ [Black-box&nbsp;DNN&nbsp;Watermarking&nbsp;(Output-dependent&nbsp;Style)](#Black-box&nbsp;DNN&nbsp;Watermarking&nbsp;(Output-independent&nbsp;Style))
    + [Output-Classifier&nbsp;Type](#Output-Classifier&nbsp;Type)
    + [Output-Extractor&nbsp;Type](#Output-Extractor&nbsp;Type)


+ [Evaluation](#Evaluation)
+ [Security](#Security)
    + [Model&nbsp;Modifications](#Model&nbsp;Modifications) ([Fine-tuning](#Fine-tuning), [Model&nbsp;Pruning&nbsp;or&nbsp;Parameter&nbsp;Pruning](#Model&nbsp;Pruning&nbsp;or&nbsp;Parameter&nbsp;Pruning), [Model&nbsp;Compression](#Model&nbsp;Compression), [Model&nbsp;Retraining](#Model&nbsp;Retraining))
     | [Removal&nbsp;Attack](#Removal&nbsp;Attack) | [Collusion&nbsp;Attack](#Collusion&nbsp;Attack) | [Overwriting&nbsp;Attack](#Overwriting&nbsp;Attack) | [Evasion&nbsp;Attack](#Evasion&nbsp;Attack) | [Ambiguity&nbsp;Attack](#Ambiguity&nbsp;Attack) | [Surrogate&nbsp;Model&nbsp;Attack&nbsp;/&nbsp;Model&nbsp;Stealing&nbsp;Attack](#Surrogate&nbsp;Model&nbsp;Attack&nbsp;/&nbsp;Model&nbsp;Stealing&nbsp;Attack) 

+ [Applications](#Applications)
    + [Image&nbsp;Processing](#Image&nbsp;Processing) | [Image&nbsp;Generation](#Image&nbsp;Generation) | [Image&nbsp;Caption](#Image&nbsp;Caption) | [Automatic&nbsp;Speech&nbsp;Recognition&nbsp;(ASR)](#Automatic&nbsp;Speech&nbsp;Recognition&nbsp;(ASR)) | [NLP](#NLP) | [GNN](#GNN) | [Federated&nbsp;Learning](#Federated&nbsp;Learning) | [Deep&nbsp;Reinforcement&nbsp;Learning](#Deep&nbsp;Reinforcement&nbsp;Learning) | [Document&nbsp;Analysis](#Document&nbsp;Analysis) | [3D](#3D)

+ [Identification&nbsp;Tracing](#DNN&nbsp;Verification)
    + [Fingerprints](#Fingerprints)
      + [Dataset](#Dataset) | [Gradient](#Gradient) | [Repair](#Repair)
    + [Fingerprinting](#Fingerprinting)


+ [Integrity&nbsp;verification](#Integrity&nbsp;verification)

+ [Similarity&nbsp;Comparison](#Similarity&nbsp;Comparison)

+ [Perspective](#Perspective)
    + [Digital&nbsp;Rights&nbsp;Management&nbsp;(DRM)](#Digital&nbsp;Rights&nbsp;Management&nbsp;(DRM)) | [Hardware](#Hardware) | [Software&nbsp;Watermarking](#Software&nbsp;Watermarking) | [Software&nbsp;Analysis](#Software&nbsp;Analysis) | [Graph&nbsp;Watermarking](#Graph&nbsp;Watermarking) | [Privacy&nbsp;Risk&nbsp;(inference&nbsp;atteck)](#Privacy&nbsp;Risk&nbsp;(inference&nbsp;atteck))


# Survey 

1. [Machine Learning IP Protection](https://dl.acm.org/doi/pdf/10.1145/3240765.3270589): Major players in the semiconductor industry provide mechanisms on device to protect the IP at rest and during execution from being copied, altered, reverse engineered, and abused by attackers. 参考硬件领域的保护措施（静态动态） | [BitTex](): cammarota2018machine | Cammarota et al, *Proceedings of the International Conference on Computer-Aided Design(ICCAD)* 2018

2. [A Survey on Model Watermarking Neural Networks](https://arxiv.org/pdf/2009.12153.pdf)： This document at hand provides the first extensive literature review on ML model watermarking schemes and attacks against them.  | [BibTex](): boenisch2020survey | Franziska Boenisch, 2020.9

3. [DNN Intellectual Property Protection: Taxonomy, Methods, Attack Resistance, and Evaluations](https://arxiv.org/pdf/2011.13564.pdf)： This paper attempts to provide a review of the existing DNN IP protection works and also an outlook. | [BibTex](): xue2020dnn | Xue et al, *GLSVLSI '21: Proceedings of the 2021 on Great Lakes Symposium on VLSI* 2020.11

4. [A survey of deep neural network watermarking techniques](https://arxiv.org/pdf/2103.09274.pdf) | [BibTex](): li2021survey | Li et al, 2021.3

5. [Protecting artificial intelligence IPs: a survey of watermarking and fingerprinting for machine learning](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/cit2.12029): The majority of previous works are focused on watermarking, while more advanced methods such as fingerprinting and attestation are promising but not yet explored in depth; provide a table to show the resilience of existing watermarking methods against attacks | [BibTex](): regazzoni2021protecting | Regazzoni et al, *CAAI Transactions on Intelligence Technology* 2021

6. [Watermarking at the service of intellectual property rights of ML models?](https://hal.archives-ouvertes.fr/hal-03206297/document#page=76) | [BibTex](): kapusta2020watermarking | Kapusta et al, *In Actes de la conférence CAID 2020*

7. [神经网络水印技术研究进展/Research Progress of Neural Networks Watermarking Technology](https://crad.ict.ac.cn/EN/article/downloadArticleFile.do?attachType=PDF&id=4425): 首先, 分析水印及其基本需求,并对神经网络水印涉及的相关技术进行介绍;对深度神经网络水印技术进行对 比,并重点对白盒和黑盒水印进行详细分析;对神经网络水印攻击技术展开对比,并按照水印攻击目标 的不同,对水印鲁棒性攻击、隐蔽性攻击、安全性攻击等技术进行分类介绍;最后对未来方向与挑战进行 探讨 ． | [BibTex](): yingjun2021research | Zhang et al, *Journal of Computer Research and Development* 2021

8. [20 Years of research on intellectual property protection](http://web.cs.ucla.edu/~miodrag/papers/Potkonjak_ISCAS_2017.pdf) | [BibTex](): potkonjak201720 | Potkonjak et al, *IEEE International Symposium on Circuits and Systems (ISCAS).* 2017

9. [DNN Watermarking: Four Challenges and a Funeral](https://dl.acm.org/doi/pdf/10.1145/3437880.3460399) | [BibTex](): barni2021four | *IH&MMSec '21*

# Infringement
## White-box&nbsp;model-dependent 
1. illegitimate plagiarism, unauthorized distribution or reproduction [lou2021when]()

2. The second party, the adversary, is an entity that doesn’t have the required resources for designing and
training a top-notch model, and wishes to make a profit out of model M without paying any copyright fee to the model vendor. The adversary can be a company that has purchased the license of M for one of their products and want to deploy it on another one without paying additional copyright fees. They can also be any entity who somehow has got their hands on the model, and wish to sell it on the darknet. Model vendor’s goal is to protect M against IP infringements by means that enable the vendors to prove their ownership, and possibly detect the source of theft. On the other hand, the adversary’s ultimate goal is to continue  profiting from M without getting caught by law enforcement [aramoon2021don]()


## Black-box 
### inference results
1. [Stealing machine learning models via prediction apis](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_tramer.pdf): protecting against an adversary with physical access to the host device of the policy is often impractical or disproportionately costly | [BibTex](): tramer2016stealing | Tramer et al, *25th USENIX* 2016

2. [Model Extraction Warning in MLaaS Paradigm](https://dl.acm.org/doi/pdf/10.1145/3274694.3274740):  | [BibTex](): kesarwani2018model | Kesarwani et al, *Proceedings of the 34th Annual Computer Security Applications Conference(ACSAC)* 2018

3. [Knockoff nets: Stealing functionality of black-box models]() | [BibTex]():  orekondy2019knockoff | *CVPR* 2019

4. [High Accuracy and High Fidelity Extraction of Neural Networks](https://arxiv.org/pdf/1909.01838.pdf): distinguish between two types of model extraction-fidelity extraction and accuracy extraction | [BibTex](): jagielski2020high | Jagielski et al, *29th {USENIX} Security Symposium (S&P)* 2020

5. [Stealing hyperparameters in machine learning](https://arxiv.org/pdf/1802.05351.pdf) | [BibTex]():  | Wang et al, *2018 IEEE Symposium on Security and Privacy (SP)*

6. [CloudLeak: Large-scale deep learning models stealing through adversarial examples](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf) | Yu et al, *Proceedings of Network and Distributed Systems Security Symposium (NDSS). 2020.*

7. [Towards Reverse-Engineering Black-Box Neural Networks](https://arxiv.org/pdf/1711.01768.pdf) | [BibTex](): oh2019towards | *ICLR 2018*

### execution behavior
1. [CSI NN: Reverse engineering of neural network architectures through electromagnetic side channel](https://www.usenix.org/system/files/sec19-batina.pdf) | Batina et al, *28th {USENIX} Security Symposium ({USENIX} Security 19)*

2. [DeepSniffer: A DNN model extraction framework based on learning architectural hints](https://sites.cs.ucsb.edu/~sherwood/pubs/ASPLOS-20-deepsniff.pdf) | Hu et al, *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems.(APLOS) 2020*

3. [Cache telepathy: Leveraging shared resource attacks to learn DNN architectures](https://www.usenix.org/system/files/sec20spring_yan_prepub.pdf) | Yan et al, *29th {USENIX} Security Symposium ({USENIX} Security 20)*


### different application
1. [Stealing Deep Reinforcement Learning Models for Fun and Profit](https://arxiv.org/pdf/2006.05032.pdf): first model extraction attack against Deep Reinforcement Learning (DRL), which enables an external adversary to precisely recover a black-box DRL model only from its interaction with the environment | [Bibtex](): chen2020stealing | Chen et al, 2020.6

2. [Good Artists Copy, Great Artists Steal: Model Extraction Attacks Against Image Translation Generative Adversarial Networks](https://arxiv.org/pdf/2104.12623.pdf): we show the first model extraction attack against real-world generative adversarial network (GAN) image translation models | [BibTex](): szyller2021good | Szyller et al, 2021.4

3. [Stealing neural networks via timing side channels](https://arxiv.org/pdf/1812.11720.pdf): Here, an adversary can extract the Neural Network parameters, infer the regularization hyperparameter, identify if a data point was part of the training data, and generate effective transferable adversarial examples to evade classifiers; this paper is exploiting the timing side channels to infer the depth of the network; using reinforcement learning to reduce the search space | [BibTex](): duddu2018stealing | Duddu et al, 2018.12

4. [Killing Two Birds with One Stone: Stealing Model and Inferring Attribute from BERT-based APIs](https://arxiv.org/pdf/2105.10909.pdf): BERT | [BibTex](): lyu2021killing | Lyu et al, 2021.5


# Access&nbsp;Control

## User&nbsp;Authentication
### Software-level
[inputting]
1. [Protect Your Deep Neural Networks from Piracy](https://www.jianguoyun.com/p/DdrMupcQ0J2UCRjaou4D): using the key to enable correct image transformation of triggers; 对trigger进行加密 | [BibTex](): chen2018protect  | Chen et al, *IEEE International Workshop on Information Forensics and Security (WIFS)* 2018

2. [Learning to Confuse: Generating Training Time Adversarial Data with Auto-Encoder](https://arxiv.org/pdf/1905.09027.pdf): modifying training data with bounded perturbation, hoping to manipulate the behavior (both targeted or non-targeted) of any corresponding trained classifier during test time when facing clean samples. 可以用来做水印 | [Code](https://github.com/kingfengji/DeepConfuse) | [BibTex](): feng2019learning | Feng et al, *NeurIPS* 2019

    simialr idea for data privacy protection -- [unlearnable_examples_making_personal_data_unexploitable](https://arxiv.org/pdf/2101.04898.pdf) | [BibTex](): huang2021unlearnable | Huang et al, *ICLR 2021*

3. [Non-Transferable Learning: A New Approach for Model Verification and Authorization](https://arxiv.org/pdf/2106.06916.pdf): propose the idea is feasible to both ownership verification (target-specified cases) and usage authorization (source-only NTL).; 反其道行之，只要加了扰动就下降，利用脆弱性，或者说是超强的转移性，exclusive | [BibTex](): wang2021nontransferable | Wang et al, *NeurIPS 2021 submission* [Mark]: for robust black-box watermarking


[processing]
1. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex]():fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | Fan et al, *NeuraIPS* 2019, 2019.9

[outputting]
1. [Active DNN IP Protection: A Novel User Fingerprint Management and DNN Authorization Control Technique](https://www.jianguoyun.com/p/DdZ92TMQ0J2UCRjt4O0D): using trigger sets as copyright management | [BibTex](): xue2020active | Xue et al, *Security and Privacy in Computing and Communications (TrustCom)* 2020

2. [ActiveGuard: An Active DNN IP Protection Technique via Adversarial Examples](https://www.jianguoyun.com/p/DdZ92TMQ0J2UCRjt4O0D): different compared with [xue2020active]: adversarial  example based | [BibTex](): xue2021activeguard | Xue et al, 2021.3


### Hardware-level
1. [MLCapsule: Guarded Offline Deployment of Machine Learning as a Service](https://arxiv.org/pdf/1808.00590.pdf):  if the user’s input is sensitive, sending it to the server is undesirable and sometimes even legally not possible. Equally, the service provider does not want to share the model by sending it to the client for protecting its intellectual property and pay-per-query business model; Beyond protecting against direct model access, we couple the  <font color=red> secure offline deployment </font> with defenses against advanced attacks on machine learning models such as model stealing, reverse engineering, and membership inference. | [BibTex](): hanzlik2018mlcapsule | Hanzlik et al, *In Proceedings of ACM Conference (Conference’17). ACM* 2019

2. [DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks](http://cseweb.ucsd.edu/~jzhao/files/DeepAttest-isca2019.pdf): the first on-device DNN attestation method that certifies the legitimacy of the DNN program mapped to the device; device-specific fingerprint | [BibTex](): chen2019deepattest | Chen et al, *ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)* 2019

3. [Hardware-Assisted Intellectual Property Protection of Deep Learning Models](https://eprint.iacr.org/2020/1016.pdf): ensures that only an authorized end-user who possesses a trustworthy hardware device (with the secret key embedded on-chip) is able to run intended DL applications using the published model | [BibTex](): chakraborty2020hardware | Chakraborty et al, *57th ACM/IEEE Design Automation Conference (DAC)* 2020


## Model&nbsp;Encryption
[Encrpted&nbsp;Data]

  (privacy-perserving)
1. [Machine Learning Classification over Encrypted Data](http://iot.stanford.edu/pubs/bost-learning-ndss15.pdf): privacy-preserving classiﬁers | [BibTex](): bost2015machine | Bost et al, *NDSS 2015*

2. [CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf) | [BibTex](): dowlin2016cryptonets | Dowlin et al, *ICML 2016*

1. [Security for Distributed Deep Neural Networks: Towards Data Confidentiality & Intellectual Property Protection](https://arxiv.org/pdf/1907.04246.pdf): Making use of Fully Homomorphic Encryption (FHE), our approach enables the protection of Distributed Neural Networks, while processing encrypted data. | [BibTex](): gomez2019security | Gomez et al, 2019.7

2. [Deep Learning as a Service Based on Encrypted Data](https://ieeexplore.ieee.org/abstract/document/9353769): we combine deep learning with homomorphic encryption algorithm and design a deep learning network model based on secure Multi-party computing (MPC); 用户不用拿到模型，云端只拿到加密的用户，在加密测试集上进行测试 | [BibTex](): hei2020deep | Hei et al, *International Conference on Networking and Network Applications (NaNA)* 2020

    (AprilPyone -- access control)
3. [Training DNN Model with Secret Key for Model Protection](https://www-isys.sd.tmu.ac.jp/local/2020/gcce2020_10_maung.pdf): main paper of AprilPyone, inpsired by perceptual image encryption ([sirichotedumrong2019pixel](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8931606),[chuman2018encryption](https://arxiv.org/pdf/1811.00236.pdf)) | [BibTex](): pyone2020training | AprilPyone et al, *2020 IEEE 9th Global Conference on Consumer Electronics (GCCE)*

4. [Transfer Learning-Based Model Protection With Secret Key](https://arxiv.org/pdf/2103.03525.pdf)：using the key to enable correct image transformation of triggers; 对trigger进行加密; improved version by enable transfer learning | [BibTex](): aprilpyone2021transfer | AprilPyone et al, 2021.3

5. [A Protection Method of Trained CNN Model with Secret Key from Unauthorized Access](): NeurIPS2021 submission | [BibTex](): maungmaung2021protection | AprilPyone et al, 2021.5


    (AprilPyone -- adversarial robustness)
1. [Encryption inspired adversarial defense for visual classification]() | [BibTex](): maung2020encryption |  AprilPyone et al, *In 2020 IEEE International Conference on Image Processing (ICIP)* 

2. [Block-wise Image Transformation with Secret Key for Adversarially Robust Defense](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9366496): propose a novel defensive transformation that enables us to maintain a high classification accuracy under the use of both clean images and adversarial examples for adversarially robust defense. The proposed transformation is a block-wise preprocessing technique with a secret key to input images [BibeTex](): aprilpyone2021block | AprilPyone et al, *IEEE Transactions on Information Forensics and Security (TIFS)* 2021

    (AprilPyone -- piracy)
1. [Piracy-Resistant DNN Watermarking by Block-Wise Image Transformation with Secret Key](https://arxiv.org/pdf/2104.04241.pdf)：using the key to enable correct image transformation of triggers; 对trigger进行加密; it is piracy-resistant, so the original watermark cannot be overwritten by a pirated watermark, and adding a new watermark decreases the model accuracy unlike most of the existing DNN watermarking methods | [BibTex](): AprilPyone2021privacy | AprilPyone et al, 2021.4 | [IH&MMSec'21 version](https://dl.acm.org/doi/pdf/10.1145/3437880.3460398)

[Encrpted&nbsp;Architecture]
1. [DeepObfuscation: Securing the Structure of Convolutional Neural Networks via Knowledge Distillation](https://arxiv.org/pdf/1806.10313.pdf): . Our obfuscation approach is very effective to protect the critical structure of a deep learning model from being exposed to attackers; limitation: weights may be more important than the architecture | [BibTex](): xu2018deepobfuscation | Xu et al, 2018.6


[Encrpted&nbsp;Weights]
1. [Enabling Secure in-Memory Neural Network Computing by Sparse Fast Gradient Encryption](https://nicsefc.ee.tsinghua.edu.cn/media/publications/2019/ICCAD19_286.pdf): utilized parameter encryption (FGSM, additive noise pattern) to prevent malicious users from using DNNs normally.  把对抗噪声加在权值上，解密时直接减去相应权值 , run-time encryption scheduling to resist confidentiality attack | [BibTex](): cai2019enabling | Cai et al, *ICCAD* 2019

2. [Deep-Lock : Secure Authorization for Deep Neural Networks](https://arxiv.org/pdf/2008.05966.pdf):  utilizes S-Boxes with good security properties to encrypt each parameter of a trained DNN model with secret keys generated from a master key via a key scheduling algorithm, same threat model with [chakraborty2020hardware] | [BibTex](): alam2020deep | Alam et al, 2020.8

3. [Chaotic Weights- A Novel Approach to Protect Intellectual Property of Deep Neural Networks 09171904](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9171904): exchanging the weight positions to obtain a satisfying encryption effect, instead of using the conventional idea of encrypting the weight values; CV, NLP tasks; | [BibTex](): lin2020chaotic | Lin et al, *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (2020)* 2020

4. [AdvParams: An Active DNN Intellectual Property Protection Technique via Adversarial Perturbation Based Parameter Encryption](https://arxiv.org/pdf/2105.13697.pdf) | [BibTex](): xue2021advparams | Xue et al, 2021.5

[Encrpted&nbsp;Weights -- Hierarchical Service]
1. [Probabilistic Selective Encryption of Convolutional Neural Networks for Hierarchical Services](https://arxiv.org/pdf/2105.12344.pdf): Probabilistic Selection Strategy (PSS); Distribution Preserving Random Mask (DPRM) | [Code]() | [BibTex](): tian2021probabilistic | *CVPR2021*


# DNN&nbsp;Watermarking&nbsp;Mechanism
1. [Machine Learning Models that Remember Too Much](https://arxiv.org/pdf/1709.07886.pdf)：redundancy: embedding secret information into network parameters | [BibTex](): song2017machine  | Song et al, *Proceedings of the 2017 ACM SIGSAC Conference on computer and communications security* 2017

1. [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf?from=timeline&isappinstalled=0)：overfitting: The capability
of neural networks to “memorize” random noise | [BibTex](): zhang2016understanding | Zhang et al, 2016.11


# White-box&nbsp;DNN&nbsp;Watermarking
## First&nbsp;Attempt
1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082)：第一篇模型水印工作 | [Code](https://github.com/yu4u/dnn-watermark) | [BibTex]): uchida2017embedding | Uchia et al, *ICMR* 2017.1

2. [Digital Watermarking for Deep Neural Networks](https://arxiv.org/pdf/1802.02601.pdf)：Extension of [1] | [BibTex](): nagai2018digital | Nagai et al, 2018.2


## Improvement
### Watermark&nbsp;Carriers
1. [DeepSigns: An End-to-End Watermarking Framework for Protecting the Ownership of Deep Neural Networks](http://www.aceslab.org/sites/default/files/deepsigns.pdf)：using activation map as cover | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](): rouhani2019deepsigns | Rouhani et al, *ASPLOS* 2019

2. [Don’t Forget To Sign The Gradients! ](https://proceedings.mlsys.org/paper/2021/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf)： imposing a statistical bias on the expected gradients of the cost function with respect to the model’s input. **introduce some adaptive watermark attacks** [Pros](): The watermark key set for GradSigns is constructed from samples of training data without any modification or relabeling, which renders this attack (Namba) futile against our method  | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](): aramoon2021don | Aramoon et al, *Proceedings of Machine Learning and Systems* 2021

3. [ When NAS Meets Watermarking: Ownership Verification of DNN Models via Cache Side Channels ](https://arxiv.org/pdf/2102.03523.pdf)：dopts a conventional NAS method with mk
to produce the watermarked architecture and a verification key vk; the owner collects the inference execution trace (by side-channel), and identifies any potential watermark based on vk | [BibTex](): lou2021when | Lou et al, 2021.2

### Loss&nbsp;Constrains&nbsp;|&nbsp;Verification&nbsp;Approach&nbsp;|&nbsp;Training&nbsp;Strategies 
[Stealthiness]
1. [Attacks on digital watermarks for deep neural networks](https://scholar.harvard.edu/files/tianhaowang/files/icassp.pdf)：weights variance or weights standard deviation, will increase noticeably and systematically during the process of watermark embedding algorithm by Uchida et al; using L2 regulatization to achieve stealthiness; w tend to mean=0, var=1 | [BibTex](): wang2019attacks | Wang et al, *ICASSP* 2019

2. [RIGA Covert and Robust White-Box Watermarking of Deep Neural Networks](https://arxiv.org/pdf/1910.14268.pdf)：improvement of [1] in stealthiness, constrain the weights distribution with advesarial training;  white-box watermark that does not impact accuracy; [Cons]() but cannot possibly protect against model stealing and  distillation attacks, since model stealing and distillation are black-box attacks and the black-box interface is unmodified by the white-box watermark. However, white-box watermarks still have important applications when the model needs to be highly accurate, or model stealing attacks are not feasible due to rate limitation or available computational resources. | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](): wang2019riga | Wang et al, *WWW* 2021

3. [Adam and the ants: On the influence of the optimization algorithm on the detectability of dnn watermarks](https://www.mdpi.com/1099-4300/22/12/1379/pdf)：improvement of [1] in stealthiness, adoption of the Adam optimiser introduces a dramatic variation on the histogram distribution of the weights after watermarking, constrain Adam optimiser is run on the projected weights using the projected gradients | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](): cortinas2020adam | Cortiñas-Lorenzo et al, *Entropy* 2020

[Capacity]
1. [RIGA Covert and Robust White-Box Watermarking of Deep Neural Networks](https://arxiv.org/pdf/1910.14268.pdf)：improvement of [1] in stealthiness, constrain the weights distribution with advesarial training;  white-box watermark that does not impact accuracy; [Cons]() but cannot possibly protect against model stealing and  distillation attacks, since model stealing and distillation are black-box attacks and the black-box interface is unmodified by the white-box watermark. However, white-box watermarks still have important applications when the model needs to be highly accurate, or model stealing attacks are not feasible due to rate limitation or available computational resources. | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](): wang2019riga | Wang et al, *WWW* 2021

[Fidelity]
1. [Spread-Transform Dither Modulation Watermarking of Deep Neural Network ](https://arxiv.org/pdf/2012.14171.pdf)：changing the activation method of [1], whcih increase the payload (capacity), couping the spread spectrum and dither modulation | [BibTex](): li2020spread | Li et al, 2020.12

2. [Watermarking in Deep Neural Networks via Error Back-propagation](https://www.ingentaconnect.com/contentone/ist/ei/2020/00002020/00000004/art00003?crawler=true&mimetype=application/pdf)：using an independent network (weights selected from the main network) to embed and extract watermark; provide some suggestions for watermarking; **introduce model isomorphism attack** | [BibTex](): wang2020watermarking | Wang et al, *Electronic Imaging* 2020.4

[Robustness]
1. [Delving in the loss landscape to embed robust watermarks into neural networks](https://www.jianguoyun.com/p/DfA64QMQ0J2UCRjlw-0D)：using partial weights to embed watermark information and keep it untrainable, optimize the non-chosen weights; denoise training strategies; robust to fine-tune and model parameter quantization  | [BibTex](): tartaglione2020delving | Tartaglione et al, *ICPR* 2020


[security]
1. [Watermarking Neural Network with Compensation Mechanism](https://www.jianguoyun.com/p/DV0-NowQ0J2UCRjey-0D): using spread spectrum (capability) and a noise sequence for security; 补偿机制指对没有嵌入水印的权值再进行fine-tune; measure changes with norm (energy perspective) | [BibTex](): feng2020watermarking | Feng et al, *International Conference on Knowledge Science, Engineering and Management* 2020 [Fidelity] | [Compensation&nbsp;Mechanism]

2. [DeepWatermark: Embedding Watermark into DNN Model](http://www.apsipa.org/proceedings/2020/pdfs/0001340.pdf)：using dither modulation in FC layers  fine-tune the pre-trainde model; the amount of changes in weights can be measured (energy perspective )  | [BibTex](): kuribayashi2020deepwatermark | Kuribayashi et al, *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* 2020  (only overwriting attack) | *IH&MMSec 21 WS* [White-Box Watermarking Scheme for Fully-Connected Layers in Fine-Tuning Model](https://dl.acm.org/doi/pdf/10.1145/3437880.3460402)

3. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex]():fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | Fan et al, *NeuraIPS* 2019, 2019.9

4. [Passport-aware Normalization for Deep Model Protection](https://proceedings.neurips.cc/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf): Improvemnet of [1] | [Code](https://github.com/ZJZAC/Passport-aware-Normalization) | [BibTex](): zhang2020passport | Zhang et al, *NeuraIPS* 2020, 2020.9

### Approaches&nbsp;Based&nbsp;on&nbsp;Muliti-task&nbsp;Learning


1. [Secure Watermark for Deep Neural Networks with Multi-task Learning](https://arxiv.org/pdf/2103.10021.pdf):  The proposed scheme explicitly meets various security requirements by using corresponding regularizers; With a decentralized consensus protocol, the entire framework is secure against all possible attacks. ;We are looking forward to using cryptological protocols such as zero-knowledge proof to improve the ownership verification process so it is possible to use one secret key for multiple notarizations. 白盒水印藏在不同地方，互相不影响，即使被擦除也没事儿？ | [BibTex](): li2021secure | Li et al, 2021.3

2. [HufuNet: Embedding the Left Piece as Watermark and Keeping the Right Piece for Ownership Verification in Deep Neural Networks](https://arxiv.org/pdf/2103.13628.pdf)：Hufu(虎符), left piece for embedding watermark, right piece as local secret; introduce some attack: model pruning, model fine-tuning, kernels cutoff/supplement and crafting adversarial samples, structure adjustment or parameter adjustment; Table12 shows the number of backoors have influence on the performance; cosine similarity is robust even weights or sturctures are adjusted, can restore the original structures or parameters; satisfy Kerckhoff's principle | [Code](https://github.com/HufuNet/HufuNet) | [BibTex](): lv2021hufunet | Lv et al, 2021.3

3. [TrojanNet: Embedding Hidden Trojan Horse Models in Neural Networks](https://arxiv.org/pdf/2002.10078.pdf): We show that this opaqueness provides an opportunity for adversaries to embed unintended functionalities into the network in the form of Trojan horses; Our method utilizes excess model capacity to simultaneously learn a public and secret task in a single network | [Code](https://github.com/wrh14/trojannet) | [NeurIPS2021 submission](https://arxiv.org/pdf/2002.10078.pdf) | [BibTex](): guo2020trojannet | Guo et al, 2020.2

## Black-box&nbsp;DNN&nbsp;Watermarking&nbsp;(Input-output&nbsp;Style)

## Unrelated&nbsp;Trigger

1. [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf)：thefirst backdoor-based， abstract image; 补充材料： From Private to Public Verifiability, Zero-Knowledge Arguments. | [Code](https://github.com/adiyoss/WatermarkNN) | [BibTex](): adi2018turning | Adi et al, *27th {USENIX} Security Symposium* 2018


<!-- 5. [TrojanNet: Embedding Hidden Trojan Horse Models in Neural Networks](https://arxiv.org/pdf/2002.10078.pdf): We show that this opaqueness provides an opportunity for adversaries to embed unintended functionalities into the network in the form of Trojan horses; Our method utilizes excess model capacity to simultaneously learn a public and secret task in a single network  | [BibTex](): guo2020trojannet | Guo et al, 2020.2 -->

2. [‘‘Identity Bracelets’’ for Deep Neural Networks](https://arxiv.org/pdf/1911.08053.pdf)：using MNIST (unrelated to original dataset) as trigger set; exploit the discarded capacity in the intermediate distribution of DL models’ output to embed the WM information; SN is a vector that contains n decimal units where n is the number of neurons in the output layer. | [BibTex](): xu2020identity  | [Initial Version: A novel method for identifying the deep neural network model with the Serial Number](https://arxiv.org/pdf/1911.08053.pdf) | Xu et al, *IEEE Access* 2020.8


3. [Secure neural network watermarking protocol against forging attack](https://www.jianguoyun.com/p/DVsuU1IQ0J2UCRic_-0D)：引入单向哈希函数，使得用于证明所有权的触发集样本必须通过连续的哈希逐个形成，并且它们的标签也按照样本的哈希值指定。 | [BibTex](): zhu2020secure | Zhu et al, *EURASIP Journal on Image and Video Processing* 2020.1

4. [A Protocol for Secure Verification of Watermarks Embedded into Machine Learning Models](https://dl.acm.org/doi/pdf/10.1145/3437880.3460409) | [BibTex
](): Kapusta2021aprotocol | Kapusta et al, *IH&MMSec 21*


2. [Protecting the Intellectual Properties of Deep Neural Networks with an Additional Class and Steganographic Images](https://arxiv.org/pdf/2104.09203.pdf):  use a set of watermark key samples to embed an additional class into the DNN; adopt the least significant bit (LSB) image steganography to embed users’ fingerprints for authentication and management of fingerprints | [BibTex](): sun2021protecting | Sun et al, 2021.4

2. [KeyNet An Asymmetric Key-Style Framework for Watermarking Deep Learning Models](https://www.mdpi.com/2076-3417/11/3/999/htm): append a private model after pristine network, the additive model for verification | [BibTex](): jebreel2021keynet| Jebreel2021keynet  et al, *Applied Sciences * 2021

## Related&nbsp;Trigger
1. [Watermarking Deep Neural Networks for Embedded Systems](http://web.cs.ucla.edu/~miodrag/papers/Guo_ICCAD_2018.pdf)：One clear drawback of their Adi is the difficulty to associate abstract images with the author’s identity. Their answer is to use a cryptographic commitment scheme, incurring a lot of overhead to the proof of authorship; using message mark as the watermark information; unlike cloud-based MLaaS that usually charge users based on the number of queries made, there is no cost associated with querying embedded systems | [BibTex](): guo2018watermarking | Guo et al, *IEEE/ACM International Conference on Computer-Aided Design (ICCAD)* 2018


1. [How to prove your model belongs to you: a blind-watermark based framework to protect intellectual property of DNN](https://arxiv.org/pdf/1903.01743.pdf)：combine some ordinary data samples with an exclusive ‘logo’ and train the model to predict them into a specific label, embedding logo into the trigger image | [BibTex](): li2019prove | Li et al, *Proceedings of the 35th Annual Computer Security Applications Conference* 2019

3. [Protecting the intellectual property of deep neural networks with watermarking: The frequency domain approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9343235): explain the failure to forgery attack of zhang-noise method. | [BibTex](): li2021protecting | Li et al, *19th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)* 2021

3. [Piracy Resistant Watermarks for Deep Neural Networks](https://arxiv.org/pdf/1910.01226.pdf): out-of-bound values; null embedding; wonder filter | [Video](https://www.youtube.com/watch?v=yb0_GwRvF4k&ab_channel=stanfordonline) | [BibTex](): li2019piracy | Li et al, 2019.10 | [Initial version](http://web.stanford.edu/class/ee380/Abstracts/191030-paper.pdf): Persistent and Unforgeable Watermarks for Deep Neural Networks | [BibTex](): li2019persistent | Li et al, 2019.10

1. [Evolutionary Trigger Set Generation for DNN Black-Box Watermarking](https://arxiv.org/pdf/1906.04411.pdf)：proposed an evolutionary algorithmbased method to generate and optimize the trigger pattern of the backdoor-based watermark to reduce the false alarm rate. | [Code](https://github.com/guojia-git/watermarking-cnn-classifiers) | [BibTex](): guo2019evolutionary | Guo et al, 2019.6


4. [Entangled Watermarks as a Defense against Model Extraction ](https://arxiv.org/pdf/2002.12200.pdf)：forcing the model to learn features which are jointly used to analyse both the normal and the triggers; using soft nearest neighbor loss (SNNL) to measure entanglement over labeled data | [Code](https://github.com/cleverhans-lab/entangled-watermark) | [BibTex](): jia2020entangled |  et al, *30th USENIX* 2020

3. [Protecting IP of Deep Neural Networks with Watermarking: A New Label Helps](https://link.springer.com/content/pdf/10.1007%2F978-3-030-47436-2_35.pdf):  adding a new label will not twist the original decision boundary but can help the model learn the features of key samples better;  investigate the relationship between model accuracy, perturbation strength, and key samples’ length.; reports more robust than zhang's method in pruning and | [BibTex]():  zhong2020protecting | Zhong et a;, *Pacific-Asia Conference on Knowledge Discovery and Data Mining* 2020

8. [Defending against Model Stealing via Verifying Embedded External Features](https://openreview.net/pdf?id=g6zfnWUg8A1): We embed the external features by poisoning a few training samples via style transfer; train a meta-classifier, based on the gradient of predictions; | [BibTex](): zhu2021defending | Zhu et al, *ICML 2021 workshop on A Blessing in Disguise: The Prospects and Perils of Adversarial Machine Learning* 



### Adversarial&nbsp;Examples
1. [Adversarial frontier stitching for remote neural network watermarking](https://arxiv.org/pdf/1711.01894.pdf)：propose a novel zero-bit watermarking algorithm that makes
use of adversarial model examples,  slightly adjusts the decision boundary of the model so that a specific set of queries can verify the watermark information.  | [Code](https://github.com/dunky11/adversarial-frontier-stitching) | [BibTex](): le2020adversarial | Merrer et al, *Neural Computing and Applications 2020* 2017.11 | [Repo by Merrer: awesome-audit-algorithms](https://github.com/erwanlemerrer/awesome-audit-algorithms): A curated list of audit algorithms for getting insights from black-box algorithms.

2. [Watermarking-based Defense against Adversarial Attacks on Deep Neural Networks](https://faculty.ist.psu.edu/wu/papers/IJCNN.pdf): we propose a new defense mechanism that creates a knowledge gap between attackers and defenders by imposing a designed watermarking system into standard deep neural networks. | [BibTex](): liwatermarking | Li et al, 2021.4

3. [A Watermarking-Based Framework for Protecting Deep Image Classifiers Against Adversarial Attacks](https://openaccess.thecvf.com/content/CVPR2021W/TCV/papers/Sun_A_Watermarking-Based_Framework_for_Protecting_Deep_Image_Classifiers_Against_Adversarial_CVPRW_2021_paper.pdf) | [BibTex](): sun2021watermarking | Sun et al, *CVPR W 2021*


5. [BlackMarks: Blackbox Multibit Watermarking for Deep Neural Networks](https://arxiv.org/pdf/1904.00344.pdf)： The first end-toend multi-bit watermarking framework ; Given the owner’s watermark signature (a binary string), a set of key image and label pairs are designed using targeted adversarial attacks; provide evaluation method | [BibTex](): chen2019blackmarks | Chen et al, 2019.4


## Related&unrelated

1. [Protecting Intellectual Property of Deep Neural Networks with Watermarking](https://www.researchgate.net/profile/Zhongshu-Gu/publication/325480419_Protecting_Intellectual_Property_of_Deep_Neural_Networks_with_Watermarking/links/5c1cfcd4a6fdccfc705f2cd4/Protecting-Intellectual-Property-of-Deep-Neural-Networks-with-Watermarking.pdf)：Three backdoor-based watermark
schemes | [BibTex](): zhang2018protecting | Zhang et al, *Asia Conference on Computer and Communications Security* 2018

1. [Certified Watermarks for Neural Networks](https://openreview.net/forum?id=Im43P9kuaeP)：Using the
randomized smoothing technique proposed in Chiang et al., we show that our watermark is guaranteed to be unremovable unless the model parameters are changed by more than a certain `2 threshold | [BibTex](): chiang2020watermarks | Bansal et al, 2018.2

1. [Visual Decoding of Hidden Watermark in Trained Deep Neural Network](https://ieeexplore.ieee.org/abstract/document/8695386)：The proposed method has a remarkable feature for watermark detection process, which can decode the embedded pattern cumulatively and visually. 关注提取端，进行label可视化成二位图片，增加关联性 | [BibTex](): sakazawa2019visual | Sakazawa et al, * IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)* 2019



## Clean&nbsp;image&nbsp;&&nbsp;Wrong&nbsp;Label
1. [Robust Watermarking of Neural Network with Exponential Weighting](https://arxiv.org/pdf/1901.06151.pdf)：using original training data with wrong label as triggers; increase the weight value exponentially so that model modification cannot change the prediction behavior of samples (including key samples) before and after model modification; introduce query modification attack, namely, pre-processing to query | [BibTex](): namba2019robust |  et al, *Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security (AisaCCS)* 2019

1. [DeepTrigger: A Watermarking Scheme of Deep Learning Models Based on Chaotic Automatic Data Annotation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9264250)：fraudulent ownership claim attacks may occur, we turn our attention to data annotation and propose a black-box watermarking scheme based on chaotic automatic data annotation; Anti-counterfeiting | [BibTex](): zhang2020deeptrigger | Zhang et al, * IEEE Access 8* 2018


## Black-box&nbsp;DNN&nbsp;Watermarking&nbsp;(Output-dependent&nbsp;Style)

<!-- 1. [Open-sourced Dataset Protection via Backdoor Watermarking](https://arxiv.org/pdf/2010.05821.pdf): use a hypothesis test guided method for dataset verification based on the posterior probability generated by the suspicious third-party model of the benign samples and their correspondingly watermarked samples  | [BibTex](): li2020open | Li ea al, *NeurIPS Workshop on Dataset Curation and Security* 2020 -->

### Output-Classifier&nbsp;Type
provide a proof of theft; verify model's origin

1. [Do gans leave artificial fingerprints?](https://arxiv.org/pdf/1812.11842.pdf): visualize GAN fingerprints based on PRNU, and show their application to GAN source identification, which is hand-crafted  fingerprint formulation | [BibTex](): marra2019gans | Marra et al, *IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)* 2019

4. [Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints](https://arxiv.org/pdf/1811.08180.pdf): We replace their hand-crafted fingerprint (of [1]) formulation with a learning-based one, decoupling model fingerprint from image fingerprint, and show superior performances in a variety of experimental conditions. | [Supplementary Material](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Yu_Attributing_Fake_Images_ICCV_2019_supplemental.pdf) | [Code](https://github.com/ningyu1991/GANFingerprints) | [Ref Code](https://github.com/cleverhans-lab/deepfake_attribution) | [BibTex]: yu2019attributing | [Homepage](https://ningyu1991.github.io/) | Yu et al, *ICCV* 2019


5. [Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data](https://arxiv.org/pdf/2007.08457.pdf): We first embed artificial fingerprints into training data, then validate a surprising discovery on the transferability of such fingerprints from training data to generative models, which in turn appears in the generated deepfakes; proactive method for deepfake detection; leverage [4] | [Empirical Study](https://www-inst.eecs.berkeley.edu/~cs194-26/fa20/upload/files/projFinalProposed/cs194-26-aek/CS294_26_Final_Project_Write_Up.pdf) | [BibTex](): yu2020artificial | Yu et al, 2020.7

6. [Responsible Disclosure of Generative Models Using Scalable Fingerprinting](https://arxiv.org/pdf/2012.08726.pdf): 和5的目的一样，具体细节diff在看 | [BibTex](): yu2020responsible | Yu et al, 2020.12


6. [Decentralized Attribution of Generative Models](https://arxiv.org/pdf/2010.13974.pdf): Each binary classifier is parameterized by a user-specific key and distinguishes its associated model distribution from the authentic data distribution. We develop sufficient conditions of the keys that guarantee an attributability lower bound.| [Code](https://github.com/ASU-Active-Perception-Group/decentralized_attribution_of_generative_models) | [BibTex](): kim2020decentralized | Kim et al, *ICLR* 2021


7. [Learning to Disentangle GAN Fingerprint for Fake Image Attribution](https://arxiv.org/pdf/2106.08749.pdf): Existing works on fake image attribution mainly rely on a direct classification framework. Without additional supervision, the extracted features could include many content-relevant components and  poorly. | [BibTex](): yang2021learning | Yang et al, 2021.6

8. [A Stealthy and Robust Fingerprinting Scheme for Generative Models](https://arxiv.org/pdf/2106.11760.pdf): We propose a new backdoor embedding approach with Unique-Triplet Loss and fine-grained categorization to enhance the effectiveness of our fingerprints. | [BibTex](): guanlin2021stealthy | Li et al, 2021.6

**阐明fingerprints和fingerprinting的不同：一个类似相机噪声，设备指纹；一个是为了进行用户追踪的分配指纹，序列号**


### Output-Extractor&nbsp;Type
1. [DeepTag: Robust Image Tagging for DeepFake Provenance](https://arxiv.org/pdf/2009.09869.pdf): using watermarked image to resist the facial manipulation; but white-box method I think,suing employ mask to embed more information, or enhance the robustness | [BibTex](): wang2020deeptag | Wang et al, 2020.9


1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VVOq5e67uCEJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1YmLg:AAGBfm0AAAAAYG5egLj8-8TdhW-OrFR5PtcTAgXDBsUU&scisig=AAGBfm0AAAAAYG5egJ2W418j7bkygIvLDr7B5IUgFq1r&scisf=4&ct=citation&cd=-1&hl=en): zhang2020model | Zhang et al, *AAAI* 2020.2

2. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:_r5iMZdEAsAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1bYNo:AAGBfm0AAAAAYG5deNoV3ooCjF9U9Rk5ckk8f8_ZS956&scisig=AAGBfm0AAAAAYG5deM2L5_2I2AvaWBetKSrL4CFclBGM&scisf=4&ct=citation&cd=-1&hl=en):    zhang2021deep | Zhang al, *TPAMI* 2021.3


4. [Watermarking Neural Networks with Watermarked Images](https://www.jianguoyun.com/p/DWcYeY8Q0J2UCRiaue4D)：Image Peocessing, similar to [1] but exclude surrogate model attack | [BibTex](): wu2020watermarking | Wu et al, *TCSVT* 2020

5. [Adversarial Watermarking Transformer: Towards Tracing Text Provenance with Data Hiding](https://arxiv.org/pdf/2009.03015.pdf):  towards marking and tracing the provenance of machine-generated text ; While the main purpose of model watermarking is to prove ownership and protect against model stealing or extraction, our language watermarking scheme is designed to trace provenance and to prevent misuse. Thus, it should be consistently present in the output, not only a response to a trigger set. | [BibTex](): abdelnabi2020adversarial | Abdelnabi et al, 2020.9


3. [Watermarking the outputs of structured prediction with an application in statistical machine translation](https://www.aclweb.org/anthology/D11-1126.pdf): proposed a method to watermark the outputs of machine learning models, especially machine translation, to be distinguished from the human-generated productions. | [BibTex](): venugopal2011watermarking | Venugopal et al, *Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing* 2011


<!-- 
8. [Radioactive data tracing through training](http://proceedings.mlr.press/v119/sablayrolles20a/sablayrolles20a.pdf): Our radioactive mark is resilient to strong data augmentations and variations of the model architecture; Our assumption is different: in our case, we control the training data, but the training process is not controlled; resite to distillation; not satisfy Kerckhoff's principle | [](): sablayrolles2020radioactive | Sablayrolles et al, *ICML* 2020

9. [Learning to Confuse: Generating Training Time Adversarial Data with Auto-Encoder](https://arxiv.org/pdf/1905.09027.pdf): modifying training data with bounded perturbation, hoping to manipulate the behavior (both targeted or non-targeted) of any corresponding trained classifier during test time when facing clean samples. 可以用来做水印 | [Code](https://github.com/kingfengji/DeepConfuse) | [BibTex](): feng2019learning | Feng et al, *NeurIPS* 2019 -->


# Evaluation

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="3"><a href="(https://arxiv.org/pdf/2009.12153.pdf"> A Survey on Model Watermarking Neural Networks </a> </th>
  </tr>
</thead>
<thead>
  <tr>
    <th class="tg-0pky">Requirement</th>
    <th class="tg-0pky">Explanation</th>
    <th class="tg-0pky">Motivation</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Fidelity</td>
    <td class="tg-0pky">Prediction quality of the model on its original task should not be degraded significantly</td>
    <td class="tg-0pky">Ensures the model's performance on the original task</td>
  </tr>
  <tr>
    <td class="tg-0pky">Robustness</td>
    <td class="tg-0pky">Watermark should be robust against removal attacks</td>
    <td class="tg-0pky">Prevents attacker from removing the watermark to avoid copyright claims of the original owner</td>
  </tr>
  <tr>
    <td class="tg-0pky">Reliability</td>
    <td class="tg-0pky">Exhibit minimal false negative rate</td>
    <td class="tg-0pky">Allows legitimate users to identify their intellectual property with a high probability</td>
  </tr>
  <tr>
    <td class="tg-0pky">Integrity</td>
    <td class="tg-0pky">Exhibit minimal false alarm rate</td>
    <td class="tg-0pky">Avoids erroneously accusing honest parties with similar models of theft</td>
  </tr>
  <tr>
    <td class="tg-0pky">Capacity</td>
    <td class="tg-0pky">Allow for inclusion of large amounts of information</td>
    <td class="tg-0pky">Enables inclusion of potentially long watermarks \eg a signature of the legitimate model owner</td>
  </tr>
  <tr>
    <td class="tg-0pky">Secrecy</td>
    <td class="tg-0pky">Presence of the watermark should be secret, watermark should be undetectable</td>
    <td class="tg-0pky">Prevents watermark detection by an unauthorized party</td>
  </tr>
  <tr>
    <td class="tg-0pky">Efficiency</td>
    <td class="tg-0pky">Process of including and verifying a watermark to ML model should be fast</td>
    <td class="tg-0pky">Does not add large overhead</td>
  </tr>
  <tr>
    <td class="tg-0pky">Unforgeability</td>
    <td class="tg-0pky">Watermark should be unforgeable</td>
    <td class="tg-0pky">No adversary can add additional watermarks to a model, or claim ownership of existing watermark from different party</td>
  </tr>
  <tr>
    <td class="tg-0pky">Authentication</td>
    <td class="tg-0pky">Provide strong link between owner and watermark that can be verified</td>
    <td class="tg-0pky">Proves legitimate owner's identity</td>
  </tr>
  <tr>
    <td class="tg-0pky">Generality</td>
    <td class="tg-0pky">Watermarking algorithm should be independent of the dataset and the ML algorithms used</td>
    <td class="tg-0pky">Allows for broad use</td>
  </tr>
</tbody>
</table>

## &nbsp;
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"><a href="(https://arxiv.org/pdf/2011.13564.pdf"> DNN Intellectual Property Protection: Taxonomy, Methods, Attack Resistance, and Evaluations </a> </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Fidelity</td>
    <td class="tg-0pky">The function and performance of the model cannot be affected by embedding the watermark.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Robustnesss</td>
    <td class="tg-0pky">The watermarking method should be able to resist model modification, such as compression, pruning, fine-tuning, or watermark overwriting.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Functionality</td>
    <td class="tg-0pky">It can support ownership verification, can use watermark to uniquely identify the model, and clearly associate the model with the identity of the IP owner.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Capacity</td>
    <td class="tg-0pky">The amount of information that the watermarking method can embed.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Efficiency</td>
    <td class="tg-0pky">The watermark embedding and extraction processes should be fast with negligible computational and communication overhead.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Reliability</td>
    <td class="tg-0pky">The watermarking method should generate the least false negatives (FN) and false positives (FP); the relevant key can be used to effectively detect the watermarked model.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Generality</td>
    <td class="tg-0pky">The watermarking method/the DNN authentication framework can be applicable to white-box and black-box scenarios, various data sets and architectures, various computing platforms.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Uniqueness</td>
    <td class="tg-0pky">The watermark/fingerprint should be unique to the target classifier. Further, each user's identity (fingerprint) should also be unique.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Indistinguishability</td>
    <td class="tg-0pky">The attacker cannot distinguish the wrong prediction from the correct model prediction.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Scalability</td>
    <td class="tg-0pky">The watermark verification technique should be able to verify DNNs of different sizes.</td>
  </tr>
</tbody>
</table>

## &nbsp;
<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-0pky{border-color:inherit;text-align:left;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"><a href="https://arxiv.org/pdf/2103.09274.pdf"> A survey of deep neural network watermarking techniques </a></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Evaluations</td>
    <td class="tg-0pky">Description</td>
  </tr>
  <tr>
    <td class="tg-0pky">Robustness</td>
    <td class="tg-0pky">The embedded watermark should resist different kinds of processing.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Security</td>
    <td class="tg-0pky">The watermark should be secure against intentional attacks from an unauthorized party.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Fidelity</td>
    <td class="tg-0pky">The watermark embedding should not significantly affect the accuracy of the target DNN architectures.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Capacity</td>
    <td class="tg-0pky">A multi-bit watermarking scheme should allow to embed as much information as possible into the host target DNN.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Integrity</td>
    <td class="tg-0pky">The bit error rate should be zero (or negligible) for multibit watermarking and the false alarm and missed detection probabilities should be small for the zero-bit case.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Generality</td>
    <td class="tg-0pky">The watermarking methodology can be applied to various DNN architectures and datasets.</td>
  </tr>
  <tr>
    <td class="tg-0pky">Efficiency</td>
    <td class="tg-0pky">The computational overhead of watermark embedding and extraction processes should be negligible.</td>
  </tr>
</tbody>
</table>


## Reference:
1. 数字水印技术及应用2004（孙圣和）1.7.1 评价问题
2. 数字水印技术及其应用2018（楼偶俊） 2.3 数字水印系统的性能评价
3. 数字水印技术及其应用2015（蒋天发）1.6 数字水印的性能评测方法
4. [Digital Rights Management The Problem of Expanding Ownership Rights](https://books.google.ca/books?id=IgSkAgAAQBAJ&lpg=PP1&ots=tA7ZrVoYx-&dq=Digital%20Rights%20Management%20The%20Problem%20of%20Expanding%20Ownership%20Rights&lr&pg=PA16#v=onepage&q=Digital%20Rights%20Management%20The%20Problem%20of%20Expanding%20Ownership%20Rights&f=false)


# Robustness&nbsp;Security
1. [Forgotten siblings: Unifying attacks on machine learning and digital watermarking](https://www.sec.cs.tu-bs.de/pubs/2018-eurosp.pdf): The two research communities have worked in parallel so far, unnoticeably developing similar attack and defense strategies. This paper is a first effort to bring these communities together. To this end, we present a unified notation of blackbox attacks against machine learning and watermarking. | [Cited by](): [Protecting artificial intelligence IPs: a survey of watermarking and fingerprinting for machine learning] | [BibTex](): quiring2018forgotten | Quiring et al, *IEEE European Symposium on Security and Privacy (EuroS&P)* 2018 

2. [Evaluating the Robustness of Trigger Set-Based Watermarks Embedded in Deep Neural Networks](https://arxiv.org/pdf/2106.10147.pdf): FT, model stealing, parameter pruning, evasion, 

## Model&nbsp;Modifications
### Fine-tuning 
1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082)：第一篇模型水印工作 | [Code](https://github.com/yu4u/dnn-watermark) | [BibTex]): uchida2017embedding | Uchia et al, *ICMR* 2017.1

### Model&nbsp;Pruning&nbsp;or&nbsp;Parameter&nbsp;Pruning
1. [DeepSigns: An End-to-End Watermarking Framework for Protecting the Ownership of Deep Neural Networks](http://www.aceslab.org/sites/default/files/deepsigns.pdf)：using activation map as cover | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](): rouhani2019deepsigns | Rouhani et al, *ASPLOS* 2019

### Model&nbsp;Compression
1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082)：第一篇模型水印工作 | [Code](https://github.com/yu4u/dnn-watermark) | [Pruning](https://arxiv.org/pdf/1510.00149)): han2015deep; *ICLR* 2016 | [BibTex]): uchida2017embedding | Uchia et al, *ICMR* 2017.1

### Model&nbsp;Retraining
1. [IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary](https://arxiv.org/pdf/1910.12903.pdf): Based on this observation, IPGuard extracts some data points near the classification boundary of the model owner’s classifier and uses them to fingerprint the classifier  | [BibTex](): cao2019ipguard | Cao et al, *AsiaCCS* 2021

2. [Robust Watermarking of Neural Network with Exponential Weighting](https://arxiv.org/pdf/1901.06151.pdf)：using original training data with wrong label as triggers; increase the weight value exponentially so that model modification cannot change the prediction behavior of samples (including key samples) before and after model modification; introduce query modification attack, namely, pre-processing to query | [BibTex](): namba2019robust |  et al, *Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security (AisaCCS)* 2019


## Removal&nbsp;Attack 
(like robust defense)

1. [Effectiveness of Distillation Attack and Countermeasure on DNN watermarking](https://arxiv.org/pdf/1906.06046.pdf)：Distilling the model's knowledge to another model of smaller size from scratch destroys all the watermarks because it has a fresh model architecture and training process; countermeasure: embedding the watermark into NN in an indiret way rather than directly overfitting the model on watermark, specifically, let the target model learn the general patterns of the trigger not regarding it as noise. evaluate both embedding and trigger watermarking | [Distillation](https://arxiv.org/pdf/1503.02531.pdf): yang2019effectiveness;  *NIPS 2014 Deep Learning Workshop* | [BibTex](): yang2019effectiveness  | Yang et al, 2019.6

2. [Attacks on digital watermarks for deep neural networks](https://scholar.harvard.edu/files/tianhaowang/files/icassp.pdf)：weights variance or weights standard deviation, will increase noticeably and systematically during the process of watermark embedding algorithm by Uchida et al; using L2 regulatization to achieve stealthiness; w tend to mean=0, var=1 | [BibTex](): wang2019attacks | Wang et al, *ICASSP* 2019

3. [On the Robustness of the Backdoor-based Watermarking in Deep Neural Networks](https://arxiv.org/pdf/1906.07745.pdf): white-box: just surrogate model attack with limited data; black-box: L2 regularization to prevent over-fitting to backdoor noise and compensate with fine-tuning; property inference attack: detect whether the backdoor-based watermark is embedded in the model | [BibTex](): shafieinejad2019robustness | Shafieinejad et al, 2019.6

4. [Leveraging unlabeled data for watermark removal of deep neural networks](https://ruoxijia.info/wp-content/uploads/2020/03/watermark_removal_icml19_workshop.pdf)：carefully-designed fine-tuning method; Leveraging auxiliary unlabeled data significantly decreases the amount of labeled training data needed for effective watermark removal, even if the unlabeled data samples are not drawn from the same distribution as the benign data for model evaluation | [BibTex](): chen2019leveraging | Chen et al, *ICML workshop on Security and Privacy of Machine Learning* 2019

5. [REFIT: A Unified Watermark Removal Framework For Deep Learning Systems With Limited Data](https://arxiv.org/pdf/1911.07205.pdf)：) an adaption of the elastic weight consolidation (EWC) algorithm, which is originally proposed for mitigating the catastrophic forgetting phenomenon;  unlabeled data augmentation (AU), where we leverage auxiliary unlabeled data from other sources | [Code](https://github.com/sunblaze-ucb/REFIT) | [BibTex](): chen2019refit | Chen et al, *ASIA CCS* 2021

6. [Removing Backdoor-Based Watermarks in Neural Networks with Limited Data](https://arxiv.org/pdf/2008.00407.pdf)：we benchmark the robustness of watermarking; propose "WILD" (data augmentation and alignment of deature distribution) with the limited access to training data| [BibTex](): liu2020removing | Liu et al, *ICASSP* 2019

7. [The Hidden Vulnerability of Watermarking for Deep Neural Networks](https://arxiv.org/pdf/2009.08697.pdf): First, we propose a novel preprocessing function, which embeds imperceptible patterns and performs spatial-level transformations over the input. Then, conduct fine-tuning strategy using unlabelled and out-ofdistribution samples. | [BibTex](): guo2020hidden | Guo et al, 2020.9 | PST is analogical to [Backdoor attack in the physical world](https://arxiv.org/pdf/2104.02361.pdf) | [BibTex](): li2021backdoor | Li et al, *ICLR 2021 Workshop on Robust and Reliable Machine Learning in the Real World*

8. [Neural network laundering: Removing black-box backdoor watermarks from deep neural networks](https://arxiv.org/pdf/2004.11368.pdf): propose a ‘laundering’ algorithm aiming to remove watermarks‐based black‐box methods ([adi, zhang]) using low‐level manipulation of the neural network based on the relative activation of neurons. | Aiken et al, *Computers & Security (2021)* 2021

9. [Re-markable: Stealing Watermarked Neural Networks Through Synthesis](https://www.jianguoyun.com/p/Da714ncQ0J2UCRiBwe8D)：using DCGAN to synthesize own training data, and using transfer learning to execute removal; analyze the failure of evasion attack, e.g., [Hitaj](https://www.researchgate.net/profile/Dorjan-Hitaj/publication/334698259_Evasion_Attacks_Against_Watermarking_Techniques_found_in_MLaaS_Systems/links/5dd6a6e692851c1feda559db/Evasion-Attacks-Against-Watermarking-Techniques-found-in-MLaaS-Systems.pdf) ; introduce the MLaaS | [BibTex](): chattopadhyay2020re | Chattopadhyay et al, *International Conference on Security, Privacy, and Applied Cryptography Engineering* 2020

10. [Neural cleanse: Identifying and mitigating backdoor attacks in neural networks](https://par.nsf.gov/servlets/purl/10120302): reverse the backdoor| [BibTex](): wang2019neural | Wang et al, *IEEE Symposium on Security and Privacy (SP)* 2019

11. [Fine-pruning: Defending against backdooring attacks on deep neural networks]() | [Fine-tuning](https://openaccess.thecvf.com/content_cvpr_2014/papers/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.pdf): girshick2014rich; *CVPR* 2014 | [BibTex](): liu2018fine | Liu et al, *International Symposium on Research in Attacks, Intrusions, and Defenses* 2018


12. [SPECTRE: Defending Against Backdoor Attacks Using Robust Statistics](https://arxiv.org/pdf/2104.11315.pdf): We propose a novel defense algorithm using robust covariance estimation to amplify the spectral signature of corrupted data. | [BibTex]():hayase2021spectre | Hayase et al, 2021.4

13. [Watermarking in Deep Neural Networks via Error Back-propagation](https://www.ingentaconnect.com/contentone/ist/ei/2020/00002020/00000004/art00003?crawler=true&mimetype=application/pdf)：using an independent network (weights selected from the main network) to embed and extract watermark; provide some suggestions for watermarking; **introduce model isomorphism attack** | [BibTex](): wang2020watermarking | Wang et al, *Electronic Imaging* 2020.4

14. [Detect and remove watermark in deep neural networks via generative adversarial networks](https://arxiv.org/pdf/2106.08104.pdf):  backdoorbased DNN watermarks are vulnerable to the proposed GANbased watermark removal attack, like Neural Cleanse, replacing the optimized method with GAN | [BibTex](): wang2021detect | Wang et al, 2021.6

15. [Fine-tuning Is Not Enough: A Simple yet Effective Watermark Removal Attack for DNN Models](https://personal.ntu.edu.sg/tianwei.zhang/paper/ijcai21.pdf) | [BibTex](): guofine | *IJCAI 2021*


## Collusion&nbsp;Attack
1. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID) | [BibTex](): chen2019deepmarks | Chen et al, *ICMR* 2019


## Evasion&nbsp;Attack  
Query-modification (like detection-based defense)
1. [Evasion Attacks Against Watermarking Techniques found in MLaaS Systems](https://www.researchgate.net/profile/Dorjan-Hitaj/publication/334698259_Evasion_Attacks_Against_Watermarking_Techniques_found_in_MLaaS_Systems/links/5dd6a6e692851c1feda559db/Evasion-Attacks-Against-Watermarking-Techniques-found-in-MLaaS-Systems.pdf)：after the verification algorithm is run against a stolen model, the adversary is in possession of the trigger set, which enables him to fine-tune the model on those data points in order to remove the watermark. | [BibTex](): hitaj2019evasion | Hitaj et al, *Sixth International Conference on Software Defined Systems (SDS)* 2019 | [Initial Version: Have You Stolen My Model? Evasion Attacks Against Deep Neural Network Watermarking Techniques](https://arxiv.org/pdf/1809.00615.pdf)

2. [Robust Watermarking of Neural Network with Exponential Weighting](https://arxiv.org/pdf/1901.06151.pdf)：using original training data with wrong label as triggers; increase the weight value exponentially so that model modification cannot change the prediction behavior of samples (including key samples) before and after model modification; introduce query modification attack, namely, pre-processing to query | [BibTex](): namba2019robust |  et al, *Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security (AisaCCS)* 2019

3. [An Evasion Algorithm to Fool Fingerprint Detector for Deep Neural Networks](https://crad.ict.ac.cn/EN/article/downloadArticleFile.do?attachType=PDF&id=4415): ． 该逃避算法的核心是设计了一个指纹样本检测器— —— FingerprintＧGAN． 利用生成对抗网络(generative adversarial network,) 原理,学习正常样本在隐空间的特征表示及其分布,根据指纹 GAN 样本与正常样本在隐空间中特征表示的差异性,检测到指纹样本,并向目标模型所有者返回有别于预测的标签,使模型所有者的指纹比对方法失效. | [BibTex](): yaguan2021evasion | Qian et al, *Journal of Computer Research and Development* 2021



## Overwriting&nbsp;Attack
1. [Watermarking Neural Network with Compensation Mechanism](https://www.jianguoyun.com/p/DV0-NowQ0J2UCRjey-0D): using spread spectrum and a noise sequence for security; 补偿机制指对没有嵌入水印的权值再进行fine-tune; measure changes with norm (energy perspective) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:xed2zy5YT5YJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD-kI4:AAGBfm0AAAAAYHP4iI6opse7jxpYkvyx4yzXtNjTcNYl&scisig=AAGBfm0AAAAAYHP4iKhXdKnITn4E9R_eO2rFPPPjZQXs&scisf=4&ct=citation&cd=-1&hl=en): feng2020watermarking | Feng et al, *International Conference on Knowledge Science, Engineering and Management* 2020

2. [DeepWatermark: Embedding Watermark into DNN Model](http://www.apsipa.org/proceedings/2020/pdfs/0001340.pdf)：using dither modulation in FC layers  fine-tune the pre-trainde model; the amount of changes in weights can be measured (energy perspective )  | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:lOUUCIYAZlQJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD2tFU:AAGBfm0AAAAAYHPwrFVFWPMAEfxbgngLdZrnTjmviyTG&scisig=AAGBfm0AAAAAYHPwrN0IxuIMpGcX6opjL56pCnY0EFHK&scisf=4&ct=citation&cd=-1&hl=en): kuribayashi2020deepwatermark | Kuribayashi et al, *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* 2020  (only overwriting attack)


<!-- 3. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:and7Xl29vpgJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3B9Rc:AAGBfm0AAAAAYG7H7Rd27DlL3WE79fbcPDcHgVpDQKuZ&scisig=AAGBfm0AAAAAYG7H7bejDswez8m_t6Y9zhsbsPAMtZ2c&scisf=4&ct=citation&cd=-1&hl=en): chen2019deepmarks | Chen et al, *ICMR* 2019

4. [BlackMarks: Blackbox Multibit Watermarking for Deep Neural Networks](https://arxiv.org/pdf/1904.00344.pdf)： The first end-toend multi-bit watermarking framework ; Given the owner’s watermark signature (a binary string), a set of key image and label pairs are designed using targeted adversarial attacks; provide evaluation method | [BibTex](): chen2019blackmarks | Chen et al, 2019.4

5. [Watermarking Deep Neural Networks in Image Processing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9093125)：Image Peocessing, using the trigger pair, target label replaced by target image; inspired by Adi and deepsigns | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:EsQcYz3vGkcJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwuwVdtk:AAGBfm0AAAAAYG8TbtkJQ70EfBd6_y4SUgSsCqJiBYKM&scisig=AAGBfm0AAAAAYG8TbkOlrwiGNIYMKOVYBzGpP7VC11zM&scisf=4&ct=citation&cd=-1&hl=en): quan2020watermarking | Quan et al, *TNNLS* 2020 -->



## Ambiguity&nbsp;Attack
forgery attack; protocol attack; invisible attack

1. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:BKAV-WKeJ1AJ:scholar.google.com/&output=citation&scisdr=CgWVvEwREJLC_OQ2dGI:AAGBfm0AAAAAYGcwbGKgqKY6a88Qf5KSWhJ1cZDTLhKp&scisig=AAGBfm0AAAAAYGcwbFH6YVqAHUeAAN6Prl_2T1s73g_a&scisf=4&ct=citation&cd=-1&hl=en):fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | [PAMI:DeepIPR: Deep Neural Network Ownership Verification with Passports](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9454280): fan2021deepip | Fan et al, *NeuraIPS* 2019, 2019.9

2. [Passport-aware Normalization for Deep Model Protection](https://proceedings.neurips.cc/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf): Improvemnet of [1] | [Code](https://github.com/ZJZAC/Passport-aware-Normalization) | [BibTex](): zhang2020passport | Zhang et al, *NeuraIPS* 2020, 2020.9

3. [Secure neural network watermarking protocol against forging attack](https://www.jianguoyun.com/p/DVsuU1IQ0J2UCRic_-0D)：引入单向哈希函数，使得用于证明所有权的触发集样本必须通过连续的哈希逐个形成，并且它们的标签也按照样本的哈希值指定。 | [BibTex](): zhu2020secure | Zhu et al, *EURASIP Journal on Image and Video Processing* 2020.1

4. [Combatting ambiguity attacks via selective detection of embedded watermarks](https://www.researchgate.net/profile/Nasir-Memon/publication/3455345_Combatting_Ambiguity_Attacks_via_Selective_Detection_of_Embedded_Watermarks/links/02e7e529fec5813232000000/Combatting-Ambiguity-Attacks-via-Selective-Detection-of-Embedded-Watermarks.pdf): | [BibTex](): sencar2007combatting | *IEEE Transactions on Information Forensics and Security (TIFS)* 2007

5. [Preventing Watermark Forging Attacks in a MLaaS Environment](https://hal.archives-ouvertes.fr/hal-03220414/): | [BibTex](): sofiane2021preventing | Sofiane et al. *SECRYPT 2021, 18th International Conference on Security and Cryptography*

## Surrogate&nbsp;Model&nbsp;Attack&nbsp;/&nbsp;Model&nbsp;Stealing&nbsp;Attack
Shall distinguishing surrogate model attack and inference attack

1. [Stealing machine learning models via prediction apis](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_tramer.pdf): protecting against an adversary with physical access to the host device of the policy is often impractical or disproportionately costly | [BibTex](): tramer2016stealing | Tramer et al, *25th USENIX* 2016

2. [Knockoff nets: Stealing functionality of black-box models]() | [BibTex](): orekondy2019knockoff | *CVPR* 2019

8. [Stealing Deep Reinforcement Learning Models for Fun and Profit](https://arxiv.org/pdf/2006.05032.pdf): first model extraction attack against Deep Reinforcement Learning (DRL), which enables an external adversary to precisely recover a black-box DRL model only from its interaction with the environment | [Bibtex](): chen2020stealing | Chen et al, 2020.6

9. [Good Artists Copy, Great Artists Steal: Model Extraction Attacks Against Image Translation Generative Adversarial Networks](https://arxiv.org/pdf/2104.12623.pdf): we show the first model extraction attack against real-world generative adversarial network (GAN) image translation models | [BibTex](): szyller2021good | Szyller et al, 2021.4

<!-- 10. [Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers](https://arxiv.org/pdf/1306.4447.pdf): This kind of information leakage can be exploited, for example, by a vendor to build more effective classifiers or to simply acquire trade secrets from a competitor’s apparatus, potentially violating its intellectual property rights. 训练数据会泄露，可以用来做模型版权溯源| [BibTex](): ateniese2015hacking | Ateniese et al, *International Journal of Security and Networks* 2015 -->

11. [High Accuracy and High Fidelity Extraction of Neural Networks](https://arxiv.org/pdf/1909.01838.pdf): distinguish between two types of model extraction-fidelity extraction and accuracy extraction | [BibTex](): jagielski2020high | Jagielski et al, *29th {USENIX} Security Symposium (S&P)* 2020

12. [Model Extraction Warning in MLaaS Paradigm](https://dl.acm.org/doi/pdf/10.1145/3274694.3274740):  | [BibTex](): kesarwani2018model | Kesarwani et al, *Proceedings of the 34th Annual Computer Security Applications Conference(ACSAC)* 2018

1. [Stealing neural networks via timing side channels](https://arxiv.org/pdf/1812.11720.pdf): Here, an adversary can extract the Neural Network parameters, infer the regularization hyperparameter, identify if a data point was part of the training data, and generate effective transferable adversarial examples to evade classifiers; this paper is exploiting the timing side channels to infer the depth of the network; using reinforcement learning to reduce the search space | [BibTex](): duddu2018stealing | Duddu et al, 2018.12



[Countermeasures]
1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](): zhang2020model | Zhang et al, *AAAI* 2020.2

4. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](): zhang2021deep | Zhang al, *TPAMI* 2021.3

5. [PRADA: Protecting Against DNN Model Stealing Attacks](https://arxiv.org/pdf/2103.04980.pdf)：detect query patterns associated with some distillation attacks | [BibTex](): juuti2019prada | Juuti al, *IEEE European Symposium on Security and Privacy (EuroS&P)* 2019

4. [Hardness of Samples Is All You Need: Protecting Deep Learning Models Using Hardness of Samples](https://arxiv.org/pdf/2106.11424.pdf): outperforms PRADA by a large margin and has significantly less computational overhead; Hardness-Oriented Detection Approach (HODA) can detect JBDA, JBRAND, and Knockoff Net attacks with a high success rate by only watching 100 samples of attack. | [BibTex](): mahdi2021hardness | Sadeghzadeh et al, 2021.6

6. [Extraction of complex DNN models: Real threat or boogeyman?](https://arxiv.org/pdf/1910.05429.pdf)：we introduce a defense based on distinguishing queries used for Knockoff nets from benign queries. | [Slide](https://asokan.org/asokan/research/ModelStealing-master.pdf) | [BibTex](): atli2020extraction | Atli et al, *International Workshop on Engineering Dependable and Secure Machine Learning Systems. Springer, Cham* 2020

7. [DAWN: Dynamic Adversarial Watermarking of Neural Networks](https://arxiv.org/pdf/1906.00830.pdf): dynamically changing the responses for a small subset of queries (e.g., <0.5%) from API clients | [BibTex](): szyller2019dawn | Szyller et al, 2019,6

8. [Protecting Decision Boundary of Machine Learning Model With Differentially Private Perturbation](https://ieeexplore.ieee.org/abstract/document/9286504)：we propose boundary differential privacy (BDP) against surrogate model attacks by obfuscating the prediction responses with noises | [BibTex](): zheng2020protecting | Zheng et al, *IEEE Transactions on Dependable and Secure Computing* 2020



13. [Prediction poisoning: Towards defenses against dnn model stealing attacks](https://arxiv.org/pdf/1906.10908v2.pdf): In this paper, we propose the first defense which actively perturbs predictions targeted at poisoning the training objective of the attacker. | [BibTex](): orekondy2019prediction | Orekondy et al, *ICLR*2020

14. [MimosaNet: An Unrobust Neural Network Preventing Model Stealing](https://arxiv.org/pdf/1907.01650.pdf): . In this paper, we propose a method for creating an equivalent version of an already trained fully connected deep neural network that can prevent network stealing: namely, it produces the same responses and classification accuracy, but it is extremely sensitive to weight changes.  | [BibTex](): szentannai2019mimosanet | Szentannai et al, 2019.7



# Applications

## Image&nbsp;Processing
1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VVOq5e67uCEJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1YmLg:AAGBfm0AAAAAYG5egLj8-8TdhW-OrFR5PtcTAgXDBsUU&scisig=AAGBfm0AAAAAYG5egJ2W418j7bkygIvLDr7B5IUgFq1r&scisf=4&ct=citation&cd=-1&hl=en): zhang2020model | Zhang et al, *AAAI* 2020.2

2. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:_r5iMZdEAsAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1bYNo:AAGBfm0AAAAAYG5deNoV3ooCjF9U9Rk5ckk8f8_ZS956&scisig=AAGBfm0AAAAAYG5deM2L5_2I2AvaWBetKSrL4CFclBGM&scisf=4&ct=citation&cd=-1&hl=en):    zhang2021deep | Zhang al, *TPAMI* 2021.3

3. [Watermarking Deep Neural Networks in Image Processing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9093125)：Image Peocessing, using the trigger pair, target label replaced by target image; inspired by Adi and deepsigns | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:EsQcYz3vGkcJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwuwVdtk:AAGBfm0AAAAAYG8TbtkJQ70EfBd6_y4SUgSsCqJiBYKM&scisig=AAGBfm0AAAAAYG8TbkOlrwiGNIYMKOVYBzGpP7VC11zM&scisf=4&ct=citation&cd=-1&hl=en): quan2020watermarking | Quan et al, *TNNLS* 2020

4. [Watermarking Neural Networks with Watermarked Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9222304)：Image Peocessing, similar to [1] but exclude surrogate model attack | [BibTex](): wu2020watermarking | Wu et al, *TCSVT* 2020


## Image&nbsp;Generation
1. [Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data](https://arxiv.org/pdf/2007.08457.pdf): We first embed artificial fingerprints into training data, then validate a surprising discovery on the transferability of such fingerprints from training data to generative models, which in turn appears in the generated deepfakes | [Empirical Study](https://www-inst.eecs.berkeley.edu/~cs194-26/fa20/upload/files/projFinalProposed/cs194-26-aek/CS294_26_Final_Project_Write_Up.pdf) | [BibTex](): yu2020artificial | Yu et al, 2020.7

2. [Responsible Disclosure of Generative Models Using Scalable Fingerprinting](https://arxiv.org/pdf/2012.08726.pdf): 和1的目的一样，具体细节diff在看 | [BibTex](): yu2020responsible | Yu et al, 2020.12


3. [Decentralized Attribution of Generative Models](https://arxiv.org/pdf/2010.13974.pdf): Each binary classifier is parameterized by a user-specific key and distinguishes its associated model distribution from the authentic data distribution. We develop sufficient conditions of the keys that guarantee an attributability lower bound.| [Code](https://github.com/ASU-Active-Perception-Group/decentralized_attribution_of_generative_models) | [BibTex](): kim2020decentralized | Kim et al, *ICLR* 2021

4. [Protecting Intellectual Property of Generative Adversarial Networks from Ambiguity Attack](https://arxiv.org/pdf/2102.04362.pdf): using trigger noise to generate trigger pattern on the original image; using passport to implenment white-box verification


## Image&nbsp;Caption 
1. [Protect, Show, Attend and Tell: Empower Image Captioning Model with Ownership Protection](https://arxiv.org/pdf/2008.11009.pdf)：Image Caption | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:Hq9e_KZON_EJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3PMvw:AAGBfm0AAAAAYG7JKvyqTMhTSiim967PfzvghJH_-Afl&scisig=AAGBfm0AAAAAYG7JKpGZo5fN_dho1v9lJBI2VxMu3iAH&scisf=4&ct=citation&cd=-1&hl=en): lim2020protect  | Lim et al, 2020.8
(surrogate model attck)


## Automatic&nbsp;Speech&nbsp;Recognition&nbsp;(ASR)
1. [SpecMark: A Spectral Watermarking Framework for IP Protection of Speech Recognition Systems](https://indico2.conference4me.psnc.pl/event/35/contributions/3413/attachments/489/514/Wed-1-8-8.pdf): Automatic Speech Recognition (ASR) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:1mZFIe2pNnAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3Pais:AAGBfm0AAAAAYG7JcivwfswRKTpDRKVkYNWU4P_fbXQ3&scisig=AAGBfm0AAAAAYG7Jcpms1fjVvGSPVAa8en4_OwmscaUY&scisf=4&ct=citation&cd=-1&hl=en): chen2020specmark | Chen et al, *Interspeech* 2020

2. [Entangled Watermarks as a Defense against Model Extraction ](https://arxiv.org/pdf/2002.12200.pdf)：forcing the model to learn features which are jointly used to analyse both the normal and the triggers; using soft nearest neighbor loss (SNNL) to measure entanglement over labeled data | [Code](https://github.com/cleverhans-lab/entangled-watermark) | [BibTex](): jia2020entangled |  et al, *30th USENIX* 2020


## NLP
1. [Watermarking Neural Language Models based on Backdooring](https://github.com/TIANHAO-WANG/nlm-watermark/blob/master/nlpwatermark.pdf): NLP | Fu et al, 2020.12

2. [Adversarial Watermarking Transformer- Towards Tracing Text Provenance with Data Hiding](https://arxiv.org/pdf/2009.03015.pdf):  towards marking and tracing the provenance of machine-generated text ; While the main purpose of model watermarking is to prove ownership and protect against model stealing or extraction, our language watermarking scheme is designed to trace provenance and to prevent misuse. Thus, it should be consistently
present in the output, not only a response to a trigger set. | [BibTex](): abdelnabi2020adversarial | Abdelnabi et al, 2020.9

3. [Watermarking the outputs of structured prediction with an application in statistical machine translation](https://www.aclweb.org/anthology/D11-1126.pdf): proposed a method to watermark the outputs of machine learning models, especially machine translation, to be distinguished from the human-generated productions. | [BibTex](): venugopal2011watermarking | Venugopal et al, *Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing* 2011

## GNN
1. [Watermarking Graph Neural Networks by Random Graphs](https://arxiv.org/pdf/2011.00512.pdf): Graph Neural Networks (GNN) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:JufA1FwKhYAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwuwS9bg:AAGBfm0AAAAAYG8U7biP-vr3I4mzYcQD9Ym4MEdwLjlL&scisig=AAGBfm0AAAAAYG8U7TJdjXkL2ClDHPdjSxMsnJ9CjcBY&scisf=4&ct=citation&cd=-1&hl=en): zhao2020watermarking | Zhao et al, *Interspeech* 2020


## Federated&nbsp;Learning
1. [WAFFLE: Watermarking in Federated Learning](https://arxiv.org/pdf/2011.00512.pdf): WAFFLE leverages capabilities of the aggregator to embed a backdoor-based watermark by re-training the global model with the watermark during each aggregation round. | [BibTex](): atli2020waffle | Atli et al, 2020.8

2. [Watermarking Federated Deep Neural Network Models](https://aaltodoc.aalto.fi/bitstream/handle/123456789/43561/master_Xia_Yuxi_2020.pdf?sequence=1): for degree of master, advisor: Buse Atli | [BibTex](): xia2020watermarking | Xia et al, 2020

3. [Towards Practical Watermark for Deep Neural Networks in Federated Learning](https://arxiv.org/pdf/2105.03167.pdf): we demonstrate a watermarking protocol for protecting deep neural networks in the setting of FL. | [BibTex](): li2021towards | Li et al, 2021.5

## Deep&nbsp;Reinforcement&nbsp;Learning
1. [Sequential Triggers for Watermarking of Deep Reinforcement Learning Policies](https://arxiv.org/pdf/1906.01126.pdf): experimental evaluation of watermarking a DQN policy trained in the Cartpole environment | [BibTex](): behzadan2019sequential | Behzadan et al, 2019,6

2. [Temporal Watermarks for Deep Reinforcement Learning Models](https://personal.ntu.edu.sg/tianwei.zhang/paper/aamas2021.pdf): Deep Reinforcement Learning (DRL) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:pafSRYDd6L8J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCqb8c:AAGBfm0AAAAAYHOsd8cEKOGOFCslTLOkJ-G7iKF_eCee&scisig=AAGBfm0AAAAAYHOsd6tpE4fU7r41NEcQsfHCyeNPpHaJ&scisf=4&ct=citation&cd=-1&hl=en): chen2021temporal | Chen et al, *International Conference on Autonomous Agents and Multiagent Systems* 2021

## Document&nbsp;Analysis 
1. [Robust Black-box Watermarking for Deep Neural Network using Inverse Document Frequency](https://arxiv.org/pdf/2103.05590.pdf): modified text as trigger;  divided into the following three categories: Watermarking the training data, network's parameters, model's output; Dataset: IMDB users' reviews, HamSpam spam detraction | [BibTex](): yadollahi2021robust | Yadollahi et al, 2021.3


## 3D
1. [Passport-aware Normalization for Deep Model Protection](https://proceedings.neurips.cc/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf): Improvemnet of [1] | [Code](https://github.com/ZJZAC/Passport-aware-Normalization) | [BibTex](): zhang2020passport | Zhang et al, *NeuraIPS* 2020, 2020.9



# Identification&nbsp;Tracing

## Fingerprints
### Dataset
1. [Open-sourced Dataset Protection via Backdoor Watermarking](https://arxiv.org/pdf/2010.05821.pdf): use a hypothesis test guided method for dataset verification based on the posterior probability generated by the suspicious third-party model of the benign samples and their correspondingly watermarked samples  | [BibTex](): li2020open | Li ea al, *NeurIPS Workshop on Dataset Curation and Security* 2020

2. [Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers](https://arxiv.org/pdf/1306.4447.pdf): This kind of information leakage can be exploited, for example, by a vendor to build more effective classifiers or to simply acquire trade secrets from a competitor’s apparatus, potentially violating its intellectual property rights. 训练数据会泄露，可以用来做模型版权溯源| [BibTex](): ateniese2015hacking | Ateniese et al, *International Journal of Security and Networks* 2015

[coded information]
3. [Honest-but-Curious Nets: Sensitive Attributes of Private Inputs can be Secretly Coded into the Entropy of Classifiers’ Outputs](https://arxiv.org/pdf/2105.12049.pdf): Our work highlights a vulnerability that can be exploited by malicious machine learning service providers to attack their user’s privacy in several seemingly safe scenarios; such as encrypted inferences, computations at the edge, or private knowledge distillation. W | 2021.5
<!-- 3. [Robust and Verifiable Information Embedding Attacks to Deep Neural Networks via Error-Correcting Codes](https://arxiv.org/pdf/2010.13751.pdf)： 使用纠错码对trigger进行annotation, 分析了和现有information embedding attack 以及 model watermarking的区别； 可以recover的不只是label, 也可以是训练数据， property， 类似inference attcak | [BibTex](): jia2020robust | Jia et al, 2020.10 -->

4. [Dataset inference: Ownership resolution in machine learning](https://openreview.net/pdf/f677fca9fd0a50d90120a4a823fcbbe889d8ca28.pdf): we identify stolen models because they possess knowledge contained in the private training set of the victim; model stealing attack;  | [Code](https://github.com/cleverhans-lab/dataset-inference) | [BibTex](): maini2021dataset | Maini et al, *International Conference on Learning Representations (ICLR)* 2021

8. [Radioactive data tracing through training](http://proceedings.mlr.press/v119/sablayrolles20a/sablayrolles20a.pdf): Our radioactive mark is resilient to strong data augmentations and variations of the model architecture; Our assumption is different: in our case, we control the training data, but the training process is not controlled; resite to distillation; not satisfy Kerckhoff's principle | [](): sablayrolles2020radioactive | Sablayrolles et al, *ICML* 2020

### Generation
1. [Proof-of-Learning: Definitions and Practice](https://arxiv.org/pdf/2103.05633.pdf): 证明训练过程的完整性，要求：验证花费小于训练花费，训练花费小于伪造花费；通过特定初始化下，梯度更新的随机性，以及逆向highly costly, 来作为交互验证的信息。可以用来做模型版权和模型完整性认证(分布训练，确定client model 是否trusty) | [Code](https://github.com/cleverhans-lab/Proof-of-Learning) | [BibTex](): jia2021proof | Jia et al, *42nd S&P* 2021.3

2. [Towards Smart Contracts for Verifying DNN Model Generation Process with the Blockchain](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9403138): we propose a smart contract that is based on the dispute resolution protocol for verifying DNN model generation process. | [BibTex](): seike2021towards | Seike et al, *IEEE 6th International Conference on Big Data Analytics (ICBDA)* 2021

<!-- ## Special
1. [Minimal Modifications of Deep Neural Networks using Verification](https://easychair-www.easychair.org/publications/download/CWhF): Adi 团队；利用模型维护领域的想法， 模型有漏洞，需要重新打补丁，但是不能使用re-train, 如何修改已经训练好的模型；所属领域：model verification, model repairing ...; <font color=red>提出了一种移除水印需要多少的代价的评价标准，measure the resistance of model watermarking </font> | [Coide](https://github.com/jjgold012/MinimalDNNModificationLpar2020) | [BibTex](): goldberger2020minimal | Goldberger et al, *LPAR* 2020

### Repair
2. [An Abstraction-Based Framework for Neural Network Verification](https://link.springer.com/chapter/10.1007/978-3-030-53288-8_3)

3. [Provable Repair of Deep Neural Networks](https://arxiv.org/pdf/2104.04413.pdf)

### DNN verification  
1. [Safety Veriﬁcation of Deep Neural Networks](https://arxiv.org/pdf/1610.06940.pdf): 论文来自牛津大学，论文也是提出希望基于可满足性模理论对神经网络的鲁棒性做一些验证  | [BibTex](): huang2017safety | et al, *International conference on computer aided verification* 2017

2. [Reluplex: An Efficient  SMT Solver for Verifying Deep Neural Networks](https://arxiv.org/pdf/1702.01135.pdf&xid=25657,15700023,15700124,15700149,15700186,15700191,15700201,15700237,15700242.pdf): 文来自斯坦福大学，论文提出了一种用于神经网络错误检测的新算法 Reluplex。Reluplex 将线性编程技术与 SMT（可满足性模块理论）求解技术相结合，其中神经网络被编码为线性算术约束; 论文的核心观点就是避免数学逻辑永远不会发生的测试路径，这允许测试比以前更大的数量级的神经网络。Reluplex 可以在一系列输入上证明神经网络的属性，可以测量可以产生虚假结果的最小或阈值对抗性信号 | [BibTex](): katz2017reluplex | et al, *International conference on computer aided verification* 2017 -->


**阐明fingerprints和fingerprinting的不同：一个类似相机噪声，设备指纹；一个是为了进行用户追踪的分配指纹，序列号**


## Fingerprinting

1. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID); The main difference between watermarking and fingerprinting is that the WM remains the same for all copies of the IP while the FP is unique for each copy. As such, FPs address the ambiguity of WMs and enables tracking of IP misuse conducted by a specific user. | [BibTex](): chen2019deepmarks | Chen et al, *ICMR* 2019

<!-- 2. [DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks](http://cseweb.ucsd.edu/~jzhao/files/DeepAttest-isca2019.pdf): the first on-device DNN attestation method that certifies the legitimacy of the DNN program mapped to the device; device-specific fingerprint | [BibTex](): chen2019deepattest | Chen et al, *ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)* 2019 -->

2. [IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary](https://arxiv.org/pdf/1910.12903.pdf): Based on this observation, IPGuard extracts some data points near the classification boundary of the model owner’s classifier and uses them to fingerprint the classifier  | [BibTex](): cao2019ipguard | Cao et al, *AsiaCCS* 2021

3. [Protecting the Intellectual Properties of Deep Neural Networks with an Additional Class and Steganographic Images](https://arxiv.org/pdf/2104.09203.pdf):  use a set of watermark key samples to embed an additional class into the DNN; adopt the least significant bit (LSB) image steganography to embed users’ fingerprints for authentication and management of fingerprints | [BibTex](): sun2021protecting | Sun et al, 2021.4

4. [Deep Serial Number: Computational Watermarking for DNN Intellectual Property Protection](https://arxiv.org/pdf/2011.08960.pdf): we introduce the first attempt to embed a serial number into DNNs,  DSN is implemented in the knowledge distillation framework, During the distillation process, each customer DNN is augmented with a unique serial number, | [BibTex](): tang2020deep | Tang et al, 2020.11

5. [AFA Adversarial fingerprinting authentication for deep neural networks](https://www.sciencedirect.com/science/article/abs/pii/S014036641931686X)：Use the adversarial examples as the model’s fingerprint | [BibTex](): zhao2020afa | Zhao et al, * Computer Communications* 2020

6. [‘‘Identity Bracelets’’ for Deep Neural Networks](https://arxiv.org/pdf/1911.08053.pdf)：using MNIST (unrelated to original dataset) as trigger set | [BibTex](): xu2020identity  | [Initial Version: A novel method for identifying the deep neural network model with the Serial Number](https://arxiv.org/pdf/1911.08053.pdf) | Xu et al, *IEEE Access* 2020.8

7. [Deep neural network fingerprinting by conferrable adversarial examples](https://arxiv.org/pdf/1912.00888.pdf): conferrable adversarial examples that exclusively transfer with a target label from a source model to its surrogates | [BibTex](): lukas2019deep | Lukas et al, *ICLR* 2021

6. [Fingerprinting Deep Neural Networks - A DeepFool Approach](): In this paper, we utilize the geometry characteristics inherited in the DeepFool algorithm to extract data points near the classification boundary of the target model for ownership verification.  | [BibTex](): wang2021fingerprinting | Wang et al, *IEEE International Symposium on Circuits and Systems (ISCAS)* 2021

9. [Forensicability of Deep Neural Network Inference Pipelines](): identification of the hardware platform used to produce deep neural network predictions. Finally, we introduce boundary samples that amplify the numerical deviations in order to distinguish machines by their predicted label only. | [BibTex](): schlogl2021forensicability | Schlogl et al, *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* 2021 

10. [iNNformant: Boundary Samples as Telltale Watermarks](https://informationsecurity.uibk.ac.at/pdfs/SKB2021_IH.pdf): Improvement of [schlogl2021forensicability];This is relevant if, in the above example, the model owner wants to probe the inference pipeline inconspicuously in order to avoid that the licensee can process obvious boundary samples in a different pipeline (the legitimate one) than the bulk of organic samples. We propose to generate transparent boundary samples as perturbations of natural input samples and measure the distortion by the peak  signal-to-noise ratio (PSNR). | [BibTex](): schlogl2021innformant | Schlogl et al, * IH&MMSEC '21* 2021

11. [A Deep Learning Framework Supporting Model Ownership Protection and Traitor Tracing](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6917&context=sis_research) | [BibTex](): xu2020deep | Xu et al, *2020 IEEE 26th International Conference on Parallel and Distributed Systems (ICPADS)*

12. [Teacher Model Fingerprinting Attacks Against Transfer Learning](https://arxiv.org/pdf/2106.12478.pdf): the choice of its teacher model certainly belongs to the model owner’s intellectual property (IP); we propose a teacher model fingerprinting attack to infer the origin of a student model, i.e., the teacher model it transfers from. | [BibTex](): chen2021teacher | Chen et al, 2021.6

# Integrity&nbsp;verification
The user may want to be sure of the provenance fo the model in some security applications or senarios

1. [Verideep: Verifying integrity of deep neural networks through sensitive-sample fingerprinting](https://arxiv.org/pdf/1808.03277.pdf) | [BibTex](): he2018verideep | He et al, 2018.8

2. [Sensitive-Sample Fingerprinting of Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.pdf): we define Sensitive-Sample fingerprints, which are a small set of human unnoticeable transformed inputs that make the model outputs sensitive to the model’s parameters. | [BibTex](): he2019sensitive | He et al, *CVPR* 2019

3. [MimosaNet: An Unrobust Neural Network Preventing Model Stealing](https://arxiv.org/pdf/1907.01650.pdf): . In this paper, we propose a method for creating an equivalent version of an already trained fully connected deep neural network that can prevent network stealing: namely, it produces the same responses and classification accuracy, but it is extremely sensitive to weight changes; focus on three consecutive FC layer | [BibTex](): szentannai2019mimosanet | Szentannai et al, 2019.7

4. [TamperNN: Efficient Tampering Detection of Deployed Neural Nets](https://arxiv.org/pdf/1903.00317.pdf): In the remote interaction setup we consider, the proposed strategy is to identify markers of the model input space that are likely to change class if the model is attacked, allowing a user to detect a possible tampering.| [BibTex](): merrer2019tampernn | Merrer et al, *IEEE 30th International Symposium on Software Reliability Engineering (ISSRE)* 2019

5. [Reversible Watermarking in Deep Convolutional Neural Networks for Integrity Authentication](https://arxiv.org/pdf/2101.04319.pdf): chose the least important weights as the cover, can reverse the original model performance, can authenticate the integrity | [BibTex](): guan2020reversible | Guan et al, *ACM MM* 2020

6. [DeepiSign: Invisible Fragile Watermark to Protect the Integrity and Authenticity of CNN](https://arxiv.org/pdf/2101.04319.pdf): convert to DCT domain, choose the high frequency to adopt LSB for information hiding | [BibTex](): abuadbba2021deepisign | Abuadbba et al, *SAC* 2021

7. [NeuNAC: A Novel Fragile Watermarking Algorithm for Integrity Protection of Neural Networks](): | [BibTex](): botta2021neunac | Botta et al, *Information Sciences (2021)*


## Special
1. [Minimal Modifications of Deep Neural Networks using Verification](https://easychair-www.easychair.org/publications/download/CWhF): Adi 团队；利用模型维护领域的想法， 模型有漏洞，需要重新打补丁，但是不能使用re-train, 如何修改已经训练好的模型；所属领域：model verification, model repairing ...; <font color=red>提出了一种移除水印需要多少的代价的评价标准，measure the resistance of model watermarking </font> | [Coide](https://github.com/jjgold012/MinimalDNNModificationLpar2020) | [BibTex](): goldberger2020minimal | Goldberger et al, *LPAR* 2020

### Repair
2. [An Abstraction-Based Framework for Neural Network Verification](https://link.springer.com/chapter/10.1007/978-3-030-53288-8_3)

3. [Provable Repair of Deep Neural Networks](https://arxiv.org/pdf/2104.04413.pdf)

<!-- ### DNN verification 
软工领域：[布尔表达式](https://baike.baidu.com/item/%E5%B8%83%E5%B0%94%E8%A1%A8%E8%BE%BE%E5%BC%8F) | [SMT (satisfiability modulo theories)](https://www.zhihu.com/question/29586582)
将模型看做软件，进行鲁棒性测试，或者是认证

1. [Safety Veriﬁcation of Deep Neural Networks](https://arxiv.org/pdf/1610.06940.pdf): 论文来自牛津大学，论文也是提出希望基于可满足性模理论对神经网络的鲁棒性做一些验证  | [BibTex](): huang2017safety | et al, *International conference on computer aided verification* 2017

2. [Reluplex: An Efficient  SMT Solver for Verifying Deep Neural Networks](https://arxiv.org/pdf/1702.01135.pdf&xid=25657,15700023,15700124,15700149,15700186,15700191,15700201,15700237,15700242.pdf): 文来自斯坦福大学，论文提出了一种用于神经网络错误检测的新算法 Reluplex。Reluplex 将线性编程技术与 SMT（可满足性模块理论）求解技术相结合，其中神经网络被编码为线性算术约束; 论文的核心观点就是避免数学逻辑永远不会发生的测试路径，这允许测试比以前更大的数量级的神经网络。Reluplex 可以在一系列输入上证明神经网络的属性，可以测量可以产生虚假结果的最小或阈值对抗性信号 | [BibTex](): katz2017reluplex | et al, *International conference on computer aided verification* 2017

3. [Boosting the Robustness Verification of DNN by Identifying the Achilles’s Heel](https://arxiv.org/pdf/1811.07108.pdf) | 2018.11

4. [DISCO Verification: Division of Input Space into COnvex polytopes for neural network verification](https://arxiv.org/pdf/2105.07776.pdf) | 2021.5 -->

# Similarity&nbsp;Comparison
1. [ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection](https://arxiv.org/pdf/2106.08890.pdf): Specifically, the behavioral pattern of a model is represented as a decision distance vector (DDV), in which each element is the distance between the model’s reactions to a pair of inputs | [BibTex](): li2021modeldiff | Li et al, *In Proceedings of the 30th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA ’21)* 

# Perspective
## Digital&nbsp;Rights&nbsp;Management&nbsp;(DRM)
1. [Survey on the Technological Aspects of Digital Rights Management](https://link.springer.com/content/pdf/10.1007%2F978-3-540-30144-8_33.pdf): Digital Rights Management (DRM) has emerged as a multidisciplinary measure to protect the copyright of content owners and to facilitate the consumption of digital content. | [BibTex](): ku2004survey | Ku et al, *International Conference on Information Security* 2004

2. [Digital rights management](http://www.medien.ifi.lmu.de/lehre/ws0607/mmn/mmn2a.pdf): slides | [BibTex](): rosenblatt2002digital | Rosenblatt et al,  *New York* 2002 


## Hardware
1. [SIGNED- A Challenge-Response Based Interrogation Scheme for Simultaneous Watermarking and Trojan Detection](https://arxiv.org/pdf/2010.05209.pdf)：半导体电路的版权保护，电路通断的选择是否可以运用到神经网络？ | [BibTex](): nair2020signed | Nair et al, 2020.10

2. [Scanning the Cycle: Timing-based Authentication on PLCs](https://arxiv.org/pdf/2102.08985.pdf): a novel
technique to authenticate PLCs is proposed that aims at raising the bar against powerful attackers while being compatible with real-time systems | [BibTex](): mujeeb2021scanning | Mujeeb et al, *AsiaCCS* 2021

4. [DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks](http://cseweb.ucsd.edu/~jzhao/files/DeepAttest-isca2019.pdf): the first on-device DNN attestation method that certifies the legitimacy of the DNN program mapped to the device; device-specific fingerprint | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:eZpX8EPeuCsJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvc4Wto:AAGBfm0AAAAAYHQ-Qtr3ntZVqjetGSROZcyDovSrlk7q&scisig=AAGBfm0AAAAAYHQ-QujApYeVnUD8NdhVflqQc31eSj4o&scisf=4&ct=citation&cd=-1&hl=en): chen2019deepattest | Chen et al, *ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)* 2019

5. [Hardware-Assisted Intellectual Property Protection of Deep Learning Models](https://eprint.iacr.org/2020/1016.pdf): nsures that only an authorized end-user who possesses a trustworthy hardware device (with the secret key embedded on-chip) is able to run intended DL applications using the published model | [BibTex](): chakraborty2020hardware | Chakraborty et al, *57th ACM/IEEE Design Automation Conference (DAC)* 2020

6. [Machine Learning IP Protection](https://dl.acm.org/doi/pdf/10.1145/3240765.3270589): Major players in the semiconductor industry provide mechanisms on device to protect the IP at rest and during execution from being copied, altered, reverse engineered, and abused by attackers. **参考硬件领域的保护措施（静态动态）** | [BitTex](): cammarota2018machine | Cammarota et al, *Proceedings of the International Conference on Computer-Aided Design(ICCAD)* 2018

7. [Preventing DNN Model IP Theft via Hardware Obfuscation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9417217): This paper presents a novel solution to defend against DL IP theft in NPUs during model distribution and deployment/execution via lightweight, keyed model obfuscation scheme. | [BibTex](): goldstein2021preventing | Goldstein et al, *IEEE Journal on Emerging and Selected Topics in Circuits and Systems* 2021

threats from side-channel attacks
1. [Stealing neural networks via timing side channels](https://arxiv.org/pdf/1812.11720.pdf): Here, an adversary can extract the Neural Network parameters, infer the regularization hyperparameter, identify if a data point was part of the training data, and generate effective transferable adversarial examples to evade classifiers; this paper is exploiting the timing side channels to infer the depth of the network; using reinforcement learning to reduce the search space | [BibTex](): duddu2018stealing | Duddu et al, 2018.12

## IC designs
1. [Analysis of watermarking techniques for graph coloring problem](https://drum.lib.umd.edu/bitstream/handle/1903/9032/c003.pdf?sequence=1)  | [BibTex](): qu1998analysis | Qu et al, *IEEE/ACM international conference on Computer-aided design* 1998 

2. [Intellectual property protection in vlsi designs: Theory and practice]() | [BibTex](): qu2007intellectual | Qu et al, *Springer Science & Business Media* 2007

3. [Hardware IP Watermarking and Fingerprinting](http://web.cs.ucla.edu/~miodrag/papers/Chang_SecureSystemDesign_2016.pdf) | [BibTex](): chang2016hardware | Chang et al, * Secure System Design and Trustable Computing* 2016


## Software&nbsp;Watermarking
1. [Software Watermarking: Models and Dynamic Embeddings](http://users.rowan.edu/~tang/courses/ref/watermarking/collberg.pdf) | [BibTex](): collberg1999software | Collberg et al, *Proceedings of the 26th ACM SIGPLAN-SIGACT symposium on Principles of programming languages* 1999

2. [A Graph Theoretic Approach to Software](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.5287&rep=rep1&type=pdf) | [BibTex](): venkatesan2001graph | Venkatesan et al, *In International Workshop on Information Hiding (pp. 157-168). Springer, Berlin, Heidelberg* 2001

## Software&nbsp;Analysis
1. [How are Deep Learning Models Similar?: An Empirical Study on Clone Analysis of Deep Learning Software](https://dl.acm.org/doi/pdf/10.1145/3387904.3389254): first study how the existing clone analysis techniques perform in the deep learning software. Secrion6.2(deep learning testing anf analysis) | [BibTex](): wu2020deep | Wu et al, *Proceedings of the 28th International Conference on Program Comprehension(ICPC)* 2020

## Graph&nbsp;Watermarking

## Privacy&nbsp;Risk&nbsp;(inference&nbsp;atteck) 

1. [Privacy risk in machine learning: Analyzing the connection to overfitting](https://arxiv.org/pdf/1709.01604.pdf): This paper examines the effect that overfitting and influence have on the ability of an attacker to learn information about the training data from machine learning models, either through training set membership inference or attribute inference attacks; our formal analysis also shows that overfitting is not necessary for these attacks and begins to shed light on what other factors may be in play | [BibTex](): yeom2018privacy| Yeom et al, *31st Computer Security Foundations Symposium (CSF)* 2018

2. [Membership Leakage in Label-Only Exposures](https://arxiv.org/pdf/2007.15528.pdf): we propose decision-based membership inference attacks and demonstrate that label-only exposures are also vulnerable to membership leakage | [BibTex](): li2020membership | Li et al, *CCS* 2021

3. [GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative Models](https://arxiv.org/pdf/1909.03935v3.pdf):  we present the first taxonomy of membership inference attacks, encompassing not only existing attacks but also our novel ones | [BibTex](): chen2020gan | Chen et al, *CCS* 2020

4. [MLCapsule: Guarded Offline Deployment of Machine Learning as a Service](https://arxiv.org/pdf/1808.00590.pdf):  if the user’s input is sensitive, sending it to the server is undesirable and sometimes even legally not possible. Equally, the service provider does not want to share the model by sending it to the client for protecting its intellectual property and pay-per-query business model; Beyond protecting against direct model access, we couple the  <font color=red> secure offline deployment </font> with defenses against advanced attacks on machine learning models such as model stealing, reverse engineering, and membership inference. | [BibTex](): hanzlik2018mlcapsule | *In Proceedings of ACM Conference (Conference’17). ACM* 2019
