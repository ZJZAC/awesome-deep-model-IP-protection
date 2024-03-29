Paper & Code 
========================
**Works for deep model intellectual property (IP) protection.**
  <!-- <br/><img src='./IP-images/220122-1.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/> -->
  
## <span id="back">Contents</span>
- [Survey](#Survey)
- [Preliminary](#Preliminary)
  + [IP Plagiarism](#IP-Plagiarism) | [IP Security](#IP-Security)

+ [Access Control](#Access-Control)
  + [Encryption Scheme](#Encryption-Scheme)
      <!-- + [Software-level](#Software-level) | [Hardware-level](#Hardware-level)  -->
  + [Product Key Scheme](#Product-Key-Scheme)
      <!-- + [Encrpted Data](#Encrpted-Data) | [Encrpted Architecture](#Encrpted-Architecture) | [Encrpted Weights](#Encrpted-Weights) -->

+ [Model Retrieval](#Model-Retrieval)

+ [DNN Watermarking](#DNN-Watermarking-Mechanism)
  + [White-box DNN Watermarking](#White-box-DNN-Watermarking)
    + [First Attempt](#First-Attempt)
    + [Improvement](#Improvement)
      + [Watermark Carriers](#Watermark-Carriers) | [Loss Constrains & Verification Approach & Training Strategies](#lvt) 
    + [Approaches Based on Muliti-task Learning](#MTL) 

  + [Black-box DNN Watermarking](#Black-box-DNN-Watermarking)
    + [Attempts with Unrelated Prediction ](#Attempts-with-Unrelated-Prediction) 
      + [Unrelated Trigger & Unrelated Prediction](#Unrelated-Trigger-&-Unrelated-Prediction)
      + [Related Trigger & Unrelated Prediction](#Related-Trigger-&-Unrelated-Prediction)
        + [Adversarial Examples](#Adversarial-Examples)
      + [Clean Image & Unrelated Prediction](#Clean-Image-&-Unrelated-Prediction)
      + [Other Attempts with Unrelated Prediction](#Other-Attempts-with-Unrelated-Prediction)
    + [Attempts with Related Prediction ](#Attempts-with-Related-Prediction) 
    + [Attempts with Clean Prediction ](#Attempts-with-Clean-Prediction) 

  + [Applications](#Applications)
    + [Image Processing](#Image-Processing) | [Image Generation](#Image-Generation) |  [Automatic Speech Recognition (ASR)](#Automatic-Speech-Recognition(ASR)) | [NLP](#NLP) | [Image Captioning](#Image-Captioning) | [3D & Graph](#3D-&-Graph) | [Federated Learning](#Federated-Learning) | [Deep Reinforcement Learning](#Deep-Reinforcement-Learning) | [Transformer](#Transformer) | [Pretrained Encoders](#Pretrained-Encoders) | [Dataset](#Dataset)
    
  + [Evaluation](#Evaluation)
  + [Robustness & Security](#Robustness-&-Security)
    + [Model Modifications](#Model-Modifications) ([Fine-tuning](#Fine-tuning), [Model Pruning or Parameter Pruning](#Model-Pruning-o-Parameter-Pruning), [Model Compression](#Model-Compression), [Model Retraining](#Model-Retraining))
     | [Removal Attack](#Removal-Attack) | [Collusion Attack](#Collusion-Attack) | [Overwriting Attack](#Overwriting-Attack) | [Evasion Attack](#Evasion-Attack) | [Ambiguity Attack](#Ambiguity-Attack) | [Surrogate Model Attack / Model Stealing Attack](#Surrogate-Model-Attack-/-Model-Stealing-Attack) 


+ [Identification Tracing](#Idetification-Tracing)
    + [Fingerprints](#Fingerprints)
      + [Boundary](#Data) | [Training](#Training) | [Inference](#Inference) 
    + [Fingerprinting](#Fingerprinting)

+ [Integrity verification](#Integrity-verification)

+ [Perspective](#Perspective)
    + [Digital Rights Management (DRM)](#Digital-Rights-Management(DRM)) | [Hardware](#Hardware) | [Software Watermarking](#Software-Watermarking) | [Software Analysis](#Software-Analysis) | [Graph Watermarking](#Graph-Watermarking) | [Privacy Risk (inference attack)](#Privacy-Risk(inference-attack))


# <span id="Survey">Survey</span> [^](#back)

1. [Machine Learning IP Protection](https://dl.acm.org/doi/pdf/10.1145/3240765.3270589): Major players in the semiconductor industry provide mechanisms on device to protect the IP at rest and during execution from being copied, altered, reverse engineered, and abused by attackers. 参考硬件领域的保护措施（静态动态） | [BitTex](): cammarota2018machine | Cammarota et al, *Proceedings of the International Conference on Computer-Aided Design(ICCAD)* 2018

2. [A Survey on Model Watermarking Neural Networks](https://arxiv.org/pdf/2009.12153.pdf)： This document at hand provides the first extensive literature review on ML model watermarking schemes and attacks against them.  | [BibTex](): boenisch2020survey | Franziska Boenisch, 2020.9

3. [DNN Intellectual Property Protection: Taxonomy, Methods, Attack Resistance, and Evaluations](https://dl.acm.org/doi/abs/10.1145/3453688.3461752)： This paper attempts to provide a review of the existing DNN IP protection works and also an outlook. | [BibTex](): xue2020dnn | Xue et al, *GLSVLSI '21: Proceedings of the 2021 on Great Lakes Symposium on VLSI* 2020.11 [TAI](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9645219)

4. [A survey of deep neural network watermarking techniques](https://arxiv.org/pdf/2103.09274.pdf) | [BibTex](): li2021survey | Li et al, 2021.3

5. [Protecting artificial intelligence IPs: a survey of watermarking and fingerprinting for machine learning](https://cora.ucc.ie/bitstream/handle/10468/12026/cit2.12029.pdf?sequence=1): The majority of previous works are focused on watermarking, while more advanced methods such as fingerprinting and attestation are promising but not yet explored in depth; provide a table to show the resilience of existing watermarking methods against attacks | [BibTex](): regazzoni2021protecting | Regazzoni et al, *CAAI Transactions on Intelligence Technology* 2021

6. [Watermarking at the service of intellectual property rights of ML models?](https://hal.archives-ouvertes.fr/hal-03206297/document#page=76) | [BibTex](): kapusta2020watermarking | Kapusta et al, *In Actes de la conférence CAID 2020*

7. [神经网络水印技术研究进展/Research Progress of Neural Networks Watermarking Technology](https://crad.ict.ac.cn/EN/article/downloadArticleFile.do?attachType=PDF&id=4425): 首先, 分析水印及其基本需求,并对神经网络水印涉及的相关技术进行介绍;对深度神经网络水印技术进行对 比,并重点对白盒和黑盒水印进行详细分析;对神经网络水印攻击技术展开对比,并按照水印攻击目标 的不同,对水印鲁棒性攻击、隐蔽性攻击、安全性攻击等技术进行分类介绍;最后对未来方向与挑战进行 探讨 ． | [BibTex](): yingjun2021research | Zhang et al, *Journal of Computer Research and Development* 2021

8. [20 Years of research on intellectual property protection](http://web.cs.ucla.edu/~miodrag/papers/Potkonjak_ISCAS_2017.pdf) | [BibTex](): potkonjak201720 | Potkonjak et al, *IEEE International Symposium on Circuits and Systems (ISCAS).* 2017

9. [DNN Watermarking: Four Challenges and a Funeral](https://dl.acm.org/doi/pdf/10.1145/3437880.3460399) | [BibTex](): barni2021four | *IH&MMSec '21*

10. [SoK: How Robust is Deep Neural Network Image Classification Watermarking?](https://arxiv.org/pdf/2108.04974.pdf#page=16&zoom=100,416,614): | Lukas, et al, *S&P2022* | [[Toolbox](https://github.com/dnn-security/Watermark-Robustness-Toolbox)]

11. [Regulating Ownership Verification for Deep Neural Networks: Scenarios, Protocols, and Prospects](https://arxiv.org/pdf/2108.09065.pdf):  we study the deep learning model intellectual property protection in three scenarios: the ownership proof, the federated learning, and the intellectual property transfer | Li, et al, * IJCAI 2021 Workshop on Toward IPR on Deep Learning as Services*

12. [Selection Guidelines for Backdoor-based Model Watermarking](https://repositum.tuwien.at/bitstream/20.500.12708/18543/1/Lederer%20Isabell%20-%202021%20-%20Selection%20Guidelines%20for%20Backdoor-based%20Model...pdf) 本科毕业论文 | lederer2021selection | Lederer, 2021 

13. [Survey on the Technological Aspects of Digital Rights Management](http://profs.sci.univr.it/~giaco/download/Watermarking-Obfuscation/survey%20DRM.pdf) | William Ku, Chi-Hung Chi | International Conference on Information Security,2004

14. [Robustness and Security of Digital Watermarks](https://d1wqtxts1xzle7.cloudfront.net/34469773/fc98_2-libre.pdf?1408337885=&response-content-disposition=inline%3B+filename%3DRobustness_and_Security_of_Digital_Water.pdf&Expires=1646298827&Signature=Fk0rJwdxEEREcRfgqA15PIjUbzoinHoqYoX4f2-mSgjS~0ueeOb0QX5tXCxEj7PDOw4qkzVcJ4WyhGNkRncpeMv~hBMTD3AcwlEsYD4P065Kkah69Ar~kJfQh2FJm-0VRCaREoLXTOCGKxmpH7zePpDwyauwlMxgUQKaVj9ms9FqpRA7NZMK2yP-aR08ofqEj7YSYFL3biXThBItxUWmGdLIYWUo2LCymINXFHvusw6detc8WLVKt0o~F02scd9gWFHwCbDSc1NfaQxdDxP8n4j0LxFkQGLhp00yJFHTYNdD-jhOEK1CubXfCDOI-6n34kfD5g46QRth1O8iWl7AKg__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)

15. [Survey on watermarking methods in the artificial intelligence domain and beyond](https://www.sciencedirect.com/science/article/pii/S0140366422000664): 利用AI设计水印算法

16. [人工智能模型水印研究综述](https://www.jsjkx.com/CN/article/openArticlePDF.jsp?id=20072)


# <span id="Preliminary">Preliminary</span> [^](#back)

## Threats for Intellectual Property of Deep Model 
### IP Plagiarism
- [model modifications]: [Fine-tuning](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf) (adi2018turning) |  [Transfer Learning](https://arxiv.org/pdf/1912.11370.pdf) (kolesnikov2020big) | [Mode Compression]()
- [model extratcion]: Some reference in **related work** of [Stealing Links from Graph Neural Networks](https://www.usenix.org/system/files/sec21summer_he.pdf)

  <!-- <details>
  <summary>inference results</summary>

    1. [Stealing machine learning models via prediction apis](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_tramer.pdf): protecting against an adversary with physical access to the host device of the policy is often impractical or disproportionately costly | [BibTex](): tramer2016stealing | Tramer et al, *25th USENIX* 2016

    2. [Model Extraction Warning in MLaaS Paradigm](https://dl.acm.org/doi/pdf/10.1145/3274694.3274740):  | [BibTex](): kesarwani2018model | Kesarwani et al, *Proceedings of the 34th Annual Computer Security Applications Conference(ACSAC)* 2018

    3. [Knockoff nets: Stealing functionality of black-box models]() | [BibTex]():  orekondy2019knockoff | *CVPR* 2019

    4. [High Accuracy and High Fidelity Extraction of Neural Networks](https://arxiv.org/pdf/1909.01838.pdf): distinguish between two types of model extraction-fidelity extraction and accuracy extraction | [BibTex](): jagielski2020high | Jagielski et al, *29th {USENIX} Security Symposium (S&P)* 2020

    5. [Stealing hyperparameters in machine learning](https://arxiv.org/pdf/1802.05351.pdf) | [BibTex]():  | Wang et al, *2018 IEEE Symposium on Security and Privacy (SP)*

    6. [CloudLeak: Large-scale deep learning models stealing through adversarial examples](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf) | Yu et al, *Proceedings of Network and Distributed Systems Security Symposium (NDSS). 2020.*

    7. [Towards Reverse-Engineering Black-Box Neural Networks](https://arxiv.org/pdf/1711.01768.pdf) | [BibTex](): oh2019towards | *ICLR 2018*
  </details> -->

  
  <!-- <details>
  <summary>different application </summary>

    1. [Stealing Deep Reinforcement Learning Models for Fun and Profit](https://arxiv.org/pdf/2006.05032.pdf): first model extraction attack against Deep Reinforcement Learning (DRL), which enables an external adversary to precisely recover a black-box DRL model only from its interaction with the environment | [Bibtex](): chen2020stealing | Chen et al, 2020.6

    2. [Good Artists Copy, Great Artists Steal: Model Extraction Attacks Against Image Translation Generative Adversarial Networks](https://arxiv.org/pdf/2104.12623.pdf): we show the first model extraction attack against real-world generative adversarial network (GAN) image translation models | [BibTex](): szyller2021good | Szyller et al, 2021.4

    3. [Killing Two Birds with One Stone: Stealing Model and Inferring Attribute from BERT-based APIs](https://arxiv.org/pdf/2105.10909.pdf): BERT | [BibTex](): lyu2021killing | Lyu et al, 2021.5

    4. [Stealing Links from Graph Neural Networks](https://www.usenix.org/system/files/sec21summer_he.pdf)

  </details> -->
### IP Security
  refer to [Sensitive-Sample Fingerprinting of Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.pdf)
  - complex cloud environment: [All your clouds are belong to us: security analysis of cloud management interfaces](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.232.497&rep=rep1&type=pdf)
- posioning attack &　backdoor attack:　refer to [poison ink](https://arxiv.org/pdf/2108.02488.pdf)


# <span id="Access-Control">Access Control</span> [^](#back)
## <span id="Encryption-Scheme">Encryption Scheme</span> [^](#back)

### Decryption before inference (TEE, device-specific)

1. [MLCapsule: Guarded Offline Deployment of Machine Learning as a Service](https://arxiv.org/pdf/1808.00590.pdf):  deployment mechanism based on Isolated Execution Environment (IEE), we couple the  <font color=red> secure offline deployment </font> with defenses against advanced attacks on machine learning models such as model stealing, reverse engineering, and membership inference. | [BibTex](): hanzlik2018mlcapsule | Hanzlik et al, *In Proceedings of ACM Conference (Conference’17). ACM* 2019

2. [Slalom: Fast, Verifiable and Private Execution of Neural Networks in Trusted Hardware](https://arxiv.org/pdf/1806.03287.pdf): It partitions DNN computations into nonlinear and linear operations. These two parts are then assigned to the TEE and the untrusted environment for execution, respectively | Tramer, Florian and Boneh, Dan | tramer2018slalom | 2018.6

3. [DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks](http://cseweb.ucsd.edu/~jzhao/files/DeepAttest-isca2019.pdf): the first on-device DNN attestation method that certifies the legitimacy of the DNN program mapped to the device; device-specific fingerprint | [BibTex](): chen2019deepattest | Chen et al, *ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)* 2019


### Decryption after inference 
(`Homomorphic Encryption HE`, does not support operations such as comparison and maximization. Therefore, HE cannot be directly applied to deep learning. Solution: approximation or leaving in plaintext)

[data, privacy concern, privacy-perserving]
1. [Machine Learning Classification over Encrypted Data](http://iot.stanford.edu/pubs/bost-learning-ndss15.pdf): privacy-preserving classiﬁers | [BibTex](): bost2015machine | Bost et al, *NDSS 2015*

2. [CryptoNets: Applying Neural Networks to Encrypted Data with High Throughput and Accuracy](http://proceedings.mlr.press/v48/gilad-bachrach16.pdf) | [BibTex](): dowlin2016cryptonets | Dowlin et al, *ICML 2016*

3. [Deep Learning as a Service Based on Encrypted Data](https://ieeexplore.ieee.org/abstract/document/9353769): we combine deep learning with homomorphic encryption algorithm and design a deep learning network model based on secure Multi-party computing (MPC); 用户不用拿到模型，云端只拿到加密的用户，在加密测试集上进行测试 | [BibTex](): hei2020deep | Hei et al, *International Conference on Networking and Network Applications (NaNA)* 2020

[model weights]
1. [Security for Distributed Deep Neural Networks: Towards Data Confidentiality & Intellectual Property Protection](https://arxiv.org/pdf/1907.04246.pdf): Making use of Fully Homomorphic Encryption (FHE), our approach enables the protection of Distributed Neural Networks, while processing encrypted data. | [BibTex](): gomez2019security | Gomez et al, 2019.7


## <span id="Product-Key-Scheme">Key Scheme</span> [^](#back)

### `Input layer`
1. [Protect Your Deep Neural Networks from Piracy](https://www.jianguoyun.com/p/DdrMupcQ0J2UCRjaou4D): using the key to enable correct image transformation of triggers; 对trigger进行加密 | [BibTex](): chen2018protect  | Chen et al, *IEEE International Workshop on Information Forensics and Security (WIFS)* 2018

(AprilPyone -- access control)

2. [A Protection Method of Trained CNN Model Using Feature Maps Transformed With Secret Key From Unauthorized Access](https://arxiv.org/pdf/2109.00224.pdf): up-to-data version, 比较完整的，涵盖了注释掉的两篇 | AprilPyone et al, 2021.9

<!-- 1. [Training DNN Model with Secret Key for Model Protection](https://www-isys.sd.tmu.ac.jp/local/2020/gcce2020_10_maung.pdf): main paper of AprilPyone, inpsired by perceptual image encryption ([sirichotedumrong2019pixel](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8931606),[chuman2018encryption](https://arxiv.org/pdf/1811.00236.pdf)) | [BibTex](): pyone2020training | AprilPyone et al, *2020 IEEE 9th Global Conference on Consumer Electronics (GCCE)*  -->

<!-- 2. [Transfer Learning-Based Model Protection With Secret Key](https://arxiv.org/pdf/2103.03525.pdf)：用两种fine-tuning：last layer or all layers | [BibTex](): aprilpyone2021transfer | AprilPyone et al, 2021.3 -->

  (AprilPyone -- semantic segmentation models)
  1. [Access Control Using Spatially Invariant Permutation of Feature Maps for Semantic Segmentation Models](https://arxiv.org/pdf/2109.01332.pdf): spatially invariant permutation with correct key | ito2021access, AprilPyone et al, 2021.9 

  (AprilPyone -- adversarial robustness)
  1. [Encryption inspired adversarial defense for visual classification]() | [BibTex](): maung2020encryption |  AprilPyone et al, *In 2020 IEEE International Conference on Image Processing (ICIP)* 

  2. [Block-wise Image Transformation with Secret Key for Adversarially Robust Defense](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9366496): propose a novel defensive transformation that enables us to maintain a high classification accuracy under the use of both clean images and adversarial examples for adversarially robust defense. The proposed transformation is a block-wise preprocessing technique with a secret key to input images [BibeTex](): aprilpyone2021block | AprilPyone et al, *IEEE Transactions on Information Forensics and Security (TIFS)* 2021

  (AprilPyone -- piracy)
  1. [Piracy-Resistant DNN Watermarking by Block-Wise Image Transformation with Secret Key](https://arxiv.org/pdf/2104.04241.pdf)：uses a secret key to verify ownership instead of trigger sets, 类似张鑫鹏的protocal， 连续变换，生成trigger sets | [BibTex](): AprilPyone2021privacy | AprilPyone et al, 2021.4 | [IH&MMSec'21 version](https://dl.acm.org/doi/pdf/10.1145/3437880.3460398)

3. [Non-Transferable Learning: A New Approach for Model Verification and Authorization](https://arxiv.org/pdf/2106.06916.pdf): propose the idea is feasible to both ownership verification (target-specified cases) and usage authorization (source-only NTL).; 反其道行之，只要加了扰动就下降，利用脆弱性，或者说是超强的转移性，exclusive | [BibTex](): wang2021nontransferable | Wang et al, *NeurIPS 2021 submission* [Mark]: for robust black-box watermarking | ICLR'22 | [blog](https://mp.weixin.qq.com/s/JgTQW9Szj40kMMsEwThCfA)


[prevent unauthorized training]

4. [A Novel Data Encryption Method Inspired by Adversarial Attacks](https://arxiv.org/pdf/2109.06634.pdf): 利用encoder生成对抗噪声转移数据分布，然后decoder再拉回去； 数据加密着，使用时解密；攻击者从加密的数据和相应的输出学不到relationship; inference stage | fernando2021novel, fernando et al, 2021.9
    
7. [Protect the Intellectual Property of Dataset against Unauthorized Use](https://arxiv.org/pdf/2109.07921.pdf): 将数据都加feature level的对噪声，然后用可逆隐写进行加密解密，可逆隐写讲干净图片利用LSB 藏在对抗样本里，攻击者只能拿到对抗样本数据集进行训  | xue2021protect, Xue et al, 2021.9

    simialr idea for data privacy protection -- [unlearnable_examples_making_personal_data_unexploitable](https://arxiv.org/pdf/2101.04898.pdf) | [BibTex](): huang2021unlearnable | Huang et al, *ICLR 2021* 

    [Learning to Confuse: Generating Training Time Adversarial Data with Auto-Encoder](https://arxiv.org/pdf/1905.09027.pdf): modifying training data with bounded perturbation, hoping to manipulate the behavior (both targeted or non-targeted) of any corresponding trained classifier during test time when facing clean samples. 可以用来做水印 | [Code](https://github.com/kingfengji/DeepConfuse) | [BibTex](): feng2019learning | Feng et al, *NeurIPS* 2019



### `Intermediate Layer`

[model architecture]
1. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex]():fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | Fan et al, *NeuraIPS* 2019, 2019.9

2. [Hardware-Assisted Intellectual Property Protection of Deep Learning Models](https://eprint.iacr.org/2020/1016.pdf): ensures that only an authorized end-user who possesses a trustworthy hardware device (with the secret key embedded on-chip) is able to run intended DL applications using the published model | [BibTex](): chakraborty2020hardware | Chakraborty et al, *57th ACM/IEEE Design Automation Conference (DAC)* 2020

[model weights]

3. [Deep-Lock : Secure Authorization for Deep Neural Networks](https://arxiv.org/pdf/2008.05966.pdf):  utilizes S-Boxes with good security properties to encrypt each parameter of a trained DNN model with secret keys generated from a master key via a key scheduling algorithm, same threat model with [chakraborty2020hardware]| [update](NN-Lock: A Lightweight Authorization to Prevent IP Threats of Deep Learning Models):不是就一次认证，每次输入都要带着认证 | [BibTex](): alam2020deep; alam2022nn | Alam et al, 2020.8

4. [Enabling Secure in-Memory Neural Network Computing by Sparse Fast Gradient Encryption](https://nicsefc.ee.tsinghua.edu.cn/media/publications/2019/ICCAD19_286.pdf): 加密尽可能少的权值使模型出错，  把对抗噪声加在权值上，解密时直接减去相应权值 , run-time encryption scheduling (layer-by-layer) to resist confidentiality attack | [BibTex](): cai2019enabling | Cai et al, *ICCAD* 2019

5. [AdvParams: An Active DNN Intellectual Property Protection Technique via Adversarial Perturbation Based Parameter Encryption](https://arxiv.org/pdf/2105.13697.pdf): 用JSMA找加密位置，更加准确，扰动更小? | [BibTex](): xue2021advparams | Xue et al, 2021.5


6. [Chaotic Weights- A Novel Approach to Protect Intellectual Property of Deep Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9171904): exchanging the weight `positions` to obtain a satisfying encryption effect, instead of using the conventional idea of encrypting the weight values; CV, NLP tasks; | [BibTex](): lin2020chaotic | Lin et al, *IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (2020)* 2020

5. [On the Importance of Encrypting Deep Features](https://arxiv.org/pdf/2108.07147.pdf): shuffle bits on tensor, direct way| [Code](https://github.com/nixingyang/ShuffleBits) | ni2021importance | Ni te al, 2021.8


[Encrpted Weights -- Hierarchical Service]
1. [Probabilistic Selective Encryption of Convolutional Neural Networks for Hierarchical Services](https://arxiv.org/pdf/2105.12344.pdf): Probabilistic Selection Strategy (PSS)， 如何优化可以借鉴; Distribution Preserving Random Mask (DPRM) | [Code]() | [BibTex](): tian2021probabilistic | Tian et al, *CVPR2021*

[Encrpted Architecture]
1. [DeepObfuscation: Securing the Structure of Convolutional Neural Networks via Knowledge Distillation](https://arxiv.org/pdf/1806.10313.pdf): . Our obfuscation approach is very effective to protect the critical structure of a deep learning model from being exposed to attackers; limitation: weights may be more important than the architecture; agaisnt transfer learning & incremental learning | [BibTex](): xu2018deepobfuscation | Xu et al, 2018.6

### `Output Layer`
1. [Active DNN IP Protection: A Novel User Fingerprint Management and DNN Authorization Control Technique](https://www.jianguoyun.com/p/DdZ92TMQ0J2UCRjt4O0D): using trigger sets as copyright management | [BibTex](): xue2020active | Xue et al, *Security and Privacy in Computing and Communications (TrustCom)* 2020

2. [ActiveGuard: An Active DNN IP Protection Technique via Adversarial Examples](https://www.jianguoyun.com/p/DdZ92TMQ0J2UCRjt4O0D): different compared with [xue2020active]: adversarial  example based | [update](https://practical-dl.github.io/long_paper/26.pdf) | [BibTex](): xue2021activeguard | Xue et al, 2021.3 

2. [Hierarchical Authorization of Convolutional Neural Networks for Multi-User](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9497716): we refer to differential privacy and use the Laplace mechanism to perturb the output of the model to vary degrees, 没有加解密过程，直接release | [BibTex](): luo2021hierarchical | Luo et al, *IEEE Signal Processing Letters 2021*

# <span id="Model-Retrieval">Model Retrieval</span> [^](#back)

1. [Grounding Representation Similarity with Statistical Testing](https://proceedings.neurips.cc/paper/2021/file/0c0bf917c7942b5a08df71f9da626f97-Paper.pdf): orthogonal procrustes for representing similarity  | ding2021grounding | Ding et al, NeurIPS 2021

2. [Similarity of Neural Network Representations Revisited](http://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf): 可以衡量不同模型结构学习到的特征表达，那到底是数据集影响还是结构影响呢？ analysis study | Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey | kornblith2019similarity | ICML'19

3. [ModelDiff: Testing-Based DNN Similarity Comparison for Model Reuse Detection](https://arxiv.org/pdf/2106.08890.pdf): Specifically, the behavioral pattern of a model is represented as a decision distance vector (DDV), in which each element is the distance between the model’s reactions to a pair of inputs 类似 twin trigger | [BibTex](): li2021modeldiff | Li et al, *In Proceedings of the 30th ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA ’21)* 

4. [Deep Neural Network Retrieval](https://dl.acm.org/doi/pdf/10.1145/3474085.3475505): reducing the computational cost of MLaaS providers. | zhong2021deep | Zhong et al, 

5. [Copy, Right? A Testing Framework for Copyright Protection of Deep Learning Models](https://arxiv.org/pdf/2112.05588.pdf): mulit-level such as porperty-level, neuron-level and layer-level | Chen, Jialuo and Wang, Jingyi and Peng, Tinglan and Sun, Youcheng and Cheng, Peng and Ji, Shouling and Ma, Xingjun and Li, Bo and Song, Dawn | chen2021copy | *S&P2022* 

# <span id="DNN-Watermarking-Mechanism">DNN Watermarking</span> [^](#back)

1. [Machine Learning Models that Remember Too Much](https://arxiv.org/pdf/1709.07886.pdf)：redundancy: embedding secret information into network parameters | [BibTex](): song2017machine  | Song et al, *Proceedings of the 2017 ACM SIGSAC Conference on computer and communications security* 2017

2. [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf?from=timeline&isappinstalled=0)：overfitting: The capability
of neural networks to “memorize” random noise | [BibTex](): zhang2016understanding | Zhang et al, 2016.11

# <span id="White-box-DNN-Watermarking">White-box DNN Watermarking</span> [^](#back)
## <span id="First-Attempt">First Attempt</span>
1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082)：第一篇模型水印工作 | [Code](https://github.com/yu4u/dnn-watermark) | [BibTex]): uchida2017embedding | Uchia et al, *ICMR* 2017.1

2. [Digital Watermarking for Deep Neural Networks](https://arxiv.org/pdf/1802.02601.pdf)：Extension of [1] | [BibTex](): nagai2018digital | Nagai et al, 2018.2

## <span id="Improvement">Improvement</span> [^](#back)
### <span id="Watermark-Carriers">Watermark Carriers</span> [^](#back)

1. [DeepSigns: An End-to-End Watermarking Framework for Protecting the Ownership of Deep Neural Networks](http://www.aceslab.org/sites/default/files/deepsigns.pdf)： pdf distribution of activation maps as cover; the activation of an intermediate layer is continuous-valued | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](): rouhani2019deepsigns | Rouhani et al, *ASPLOS* 2019

2. [Don’t Forget To Sign The Gradients! ](https://proceedings.mlsys.org/paper/2021/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf)： imposing a statistical bias on the expected gradients of the cost function with respect to the model’s input. **introduce some adaptive watermark attacks** [Pros](): The watermark key set for GradSigns is constructed from samples of training data without any modification or relabeling, which renders this attack (Namba) futile against our method  | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](): aramoon2021don | Aramoon et al, *Proceedings of Machine Learning and Systems* 2021

3. [ When NAS Meets Watermarking: Ownership Verification of DNN Models via Cache Side Channels ](https://arxiv.org/pdf/2102.03523.pdf)：dopts a conventional NAS method with mk
to produce the watermarked architecture and a verification key vk; the owner collects the inference execution trace (by side-channel), and identifies any potential watermark based on vk | [BibTex](): lou2021when | Lou et al, 2021.2

4. [Structural Watermarking to Deep Neural Networks via Network Channel Pruning](https://arxiv.org/pdf/2107.08688.pdf):  structural watermarking scheme that utilizes `channel pruning` to embed the watermark into the host DNN architecture instead of crafting the DNN parameters; bijective algorithm | [BibTex](): zhao2021structural | Zhao et al, 2021.7

5. [You are caught stealing my winning lottery ticket! Making a lottery ticket claim its ownership](https://proceedings.neurips.cc/paper/2021/file/0dfd8a39e2a5dd536c185e19a804a73b-Paper.pdf): 让模型尽可能地精简,把signature藏在sparse mask里 | [BibTex](): chen2021you | Chen et al, NeurIPS 2021.



### <span id="lvt">Loss Constrains | Verification Approach | Training Strategies</span> [^](#back)
 
[Stealthiness]
1. [Attacks on digital watermarks for deep neural networks](https://scholar.harvard.edu/files/tianhaowang/files/icassp.pdf)：weights variance or weights standard deviation, will increase noticeably and systematically during the process of watermark embedding algorithm by Uchida et al; using L2 regulatization to achieve stealthiness; w tend to mean=0, var=1 | [BibTex](): wang2019attacks | Wang et al, *ICASSP* 2019

2. [RIGA Covert and Robust White-Box Watermarking of Deep Neural Networks](https://arxiv.org/pdf/1910.14268.pdf)：improvement of [1] in stealthiness, constrain the weights distribution with advesarial training;  white-box watermark that does not impact accuracy; [Cons]() but cannot possibly protect against model stealing and  distillation attacks, since model stealing and distillation are black-box attacks and the black-box interface is unmodified by the white-box watermark. However, white-box watermarks still have important applications when the model needs to be highly accurate, or model stealing attacks are not feasible due to rate limitation or available computational resources. | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](): wang2019riga | Wang et al, *WWW* 2021

3. [Adam and the ants: On the influence of the optimization algorithm on the detectability of dnn watermarks](https://www.mdpi.com/1099-4300/22/12/1379/pdf)：improvement of [1] in stealthiness, adoption of the Adam optimiser introduces a dramatic variation on the histogram distribution of the weights after watermarking, constrain Adam optimiser is run on the projected weights using the projected gradients | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](): cortinas2020adam | Cortiñas-Lorenzo et al, *Entropy* 2020

[Capacity]
1. [RIGA Covert and Robust White-Box Watermarking of Deep Neural Networks](https://arxiv.org/pdf/1910.14268.pdf)：improvement of [1] in stealthiness, constrain the weights distribution with advesarial training;  white-box watermark that does not impact accuracy; [Cons]() but cannot possibly protect against model stealing and  distillation attacks, since model stealing and distillation are black-box attacks and the black-box interface is unmodified by the white-box watermark. However, white-box watermarks still have important applications when the model needs to be highly accurate, or model stealing attacks are not feasible due to rate limitation or available computational resources. | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](): wang2019riga | Wang et al, *WWW* 2021

2. [A Feature-Map-Based Large-Payload DNN Watermarking Algorithm](https://link.springer.com/content/pdf/10.1007%2F978-3-030-95398-0_10.pdf): simialr to deepsign, feature map as cover | Li, Yue and Abady, Lydia and Wang, Hongxia and Barni, Mauro | li2021feature | International Workshop on Digital Watermarking, 2021

[Fidelity]
1. [Spread-Transform Dither Modulation Watermarking of Deep Neural Network ](https://arxiv.org/pdf/2012.14171.pdf)：changing the activation method of [1], whcih increase the payload (capacity), couping the spread spectrum and dither modulation | [BibTex](): li2020spread | Li et al, 2020.12

2. [Watermarking in Deep Neural Networks via Error Back-propagation](https://www.ingentaconnect.com/contentone/ist/ei/2020/00002020/00000004/art00003?crawler=true&mimetype=application/pdf)：using an independent network (weights selected from the main network) to embed and extract watermark; provide some suggestions for watermarking; **introduce model isomorphism attack** | [BibTex](): wang2020watermarking | Wang et al, *Electronic Imaging* 2020.4

[Robustness]
1. [Delving in the loss landscape to embed robust watermarks into neural networks](https://www.jianguoyun.com/p/DfA64QMQ0J2UCRjlw-0D)：using partial weights to embed watermark information and keep it untrainable, optimize the non-chosen weights; denoise training strategies; robust to fine-tune and model parameter quantization  | [BibTex](): tartaglione2020delving | Tartaglione et al, *ICPR* 2020

2. [DeepWatermark: Embedding Watermark into DNN Model](http://www.apsipa.org/proceedings/2020/pdfs/0001340.pdf)：using dither modulation in FC layers  fine-tune the pre-trainde model; the amount of changes in weights can be measured (energy perspective )  | [BibTex](): kuribayashi2020deepwatermark | Kuribayashi et al, *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* 2020  (only overwriting attack) | *IH&MMSec 21 WS* [White-Box Watermarking Scheme for Fully-Connected Layers in Fine-Tuning Model](https://dl.acm.org/doi/pdf/10.1145/3437880.3460402)

3. [TATTOOED: A Robust Deep Neural Network Watermarking Scheme based on Spread-Spectrum Channel Coding](https://arxiv.org/pdf/2202.06091.pdf): 权值直接添加编码，改变后过一遍数据测准确率然后再调节超参数，寻找 optimum； robust to refit; 不同性质定义的很好 | Pagnotta, Giulio and Hitaj, Dorjan and Hitaj, Briland and Perez-Cruz, Fernando and Mancini, Luigi V | pagnotta2022tattooed | 2022.2

4. [Immunization of Pruning Attack in DNN Watermarking Using Constant Weight Code](https://arxiv.org/pdf/2107.02961.pdf): solve the vulnerability against pruning, viewpoint of communication | [BibTex](): kuribayashi2021immunization, kuribayashi et al, 2021.7

5. [Fostering The Robustness Of White-box Deep Neural Network Watermarks By Neuron Alignment](https://arxiv.org/pdf/2112.14108.pdf): using trigger to remember the order of  neurons, against the neuron permutation attack [lukas,sok] | li2021fostering | 2021.12

[security]
1. [Watermarking Neural Network with Compensation Mechanism](https://www.jianguoyun.com/p/DV0-NowQ0J2UCRjey-0D): using spread spectrum (capability) and a noise sequence for security; 补偿机制指对没有嵌入水印的权值再进行fine-tune; measure changes with norm (energy perspective), 多用方法在前面robustness中都提过，此处可以不再提这篇工作 | [BibTex](): feng2020watermarking | Feng et al, *International Conference on Knowledge Science, Engineering and Management* 2020 [Fidelity] | [Compensation Mechanism]

2. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex](): fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | Fan et al, *NeuraIPS* 2019, 2019.9

3. [Passport-aware Normalization for Deep Model Protection](https://proceedings.neurips.cc/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf): Improvemnet of [1] | [Code](https://github.com/ZJZAC/Passport-aware-Normalization) | [BibTex](): zhang2020passport | Zhang et al, *NeuraIPS* 2020, 2020.9

4. [Watermarking Deep Neural Networks with Greedy Residuals](http://proceedings.mlr.press/v139/liu21x.html): less is more; feasible to the deep model without normalization layer | [BibTex](): liu2021watermarking | Liu et al, *ICML 2021* 

5. [Collusion Resistant Watermarking for Deep Learning Models Protection](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9728937): 可不引用 ： ICACT‘22 | Tokyo



### <span id="MTL">Approaches Based on Muliti-task Learning</span> [^](#back)
 
1. [Secure Watermark for Deep Neural Networks with Multi-task Learning](https://arxiv.org/pdf/2103.10021.pdf):  The proposed scheme explicitly meets various security requirements by using corresponding regularizers; With a decentralized consensus protocol, the entire framework is secure against all possible attacks. ;We are looking forward to using cryptological protocols such as zero-knowledge proof to improve the ownership verification process so it is possible to use one secret key for multiple notarizations. 白盒水印藏在不同地方，互相不影响，即使被擦除也没事儿？ | [BibTex](): li2021secure | Li et al, 2021.3

2. [HufuNet: Embedding the Left Piece as Watermark and Keeping the Right Piece for Ownership Verification in Deep Neural Networks](https://arxiv.org/pdf/2103.13628.pdf)：Hufu(虎符), left piece for embedding watermark, right piece as local secret; introduce some attack: model pruning, model fine-tuning, kernels cutoff/supplement and crafting adversarial samples, structure adjustment or parameter adjustment; Table12 shows the number of backoors have influence on the performance; cosine similarity is robust even weights or sturctures are adjusted, can restore the original structures or parameters; satisfy Kerckhoff's principle | [Code](https://github.com/HufuNet/HufuNet) | [BibTex](): lv2021hufunet | Lv et al, 2021.3

3. [TrojanNet: Embedding Hidden Trojan Horse Models in Neural Networks](https://arxiv.org/pdf/2002.10078.pdf): We show that this opaqueness provides an opportunity for adversaries to embed unintended functionalities into the network in the form of Trojan horses; Our method utilizes excess model capacity to simultaneously learn a public and secret task in a single network | [Code](https://github.com/wrh14/trojannet) | [NeurIPS2021 submission](https://arxiv.org/pdf/2002.10078.pdf) | [BibTex](): guo2020trojannet | Guo et al, 2020.2

<!-- ## Black-box DNN Watermarking (Input-output Style) -->
# <span id="Black-box-DNN-Watermarking">Black-box DNN Watermarking</span> [^](#back)
## <span id="Unrelated-Trigger-&-Unrelated-Prediction">Unrelated Trigger & Unrelated Prediction</span> [^](#back)

1. [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf)：thefirst backdoor-based， abstract image; 补充材料： From Private to Public Verifiability, Zero-Knowledge Arguments. | [Code](https://github.com/adiyoss/WatermarkNN) | [BibTex](): adi2018turning | Adi et al, *27th {USENIX} Security Symposium* 2018

2. [Protecting Intellectual Property of Deep Neural Networks with Watermarking](https://www.researchgate.net/profile/Zhongshu-Gu/publication/325480419_Protecting_Intellectual_Property_of_Deep_Neural_Networks_with_Watermarking/links/5c1cfcd4a6fdccfc705f2cd4/Protecting-Intellectual-Property-of-Deep-Neural-Networks-with-Watermarking.pdf)：Three backdoor-based watermark schemes | [BibTex](): zhang2018protecting | Zhang et al, *Asia Conference on Computer and Communications Security* 2018

3. [KeyNet An Asymmetric Key-Style Framework for Watermarking Deep Learning Models](https://www.mdpi.com/2076-3417/11/3/999/htm): append a private model after pristine network, the additive model for verification; describe drawbacks of two type triggers, 分析了对不同攻击的原理性解释:ft, pruning, overwriting; 做了laundring的实验 | [BibTex](): jebreel2021keynet| Jebreel et al, *Applied Sciences * 2021

## <span id="Related-Trigger-&-Unrelated-Prediction">Related Trigger & Unrelated Prediction</span> [^](#back)

1. [Protecting Intellectual Property of Deep Neural Networks with Watermarking](https://www.researchgate.net/profile/Zhongshu-Gu/publication/325480419_Protecting_Intellectual_Property_of_Deep_Neural_Networks_with_Watermarking/links/5c1cfcd4a6fdccfc705f2cd4/Protecting-Intellectual-Property-of-Deep-Neural-Networks-with-Watermarking.pdf)：Three backdoor-based watermark schemes: specific test string, some pattern of noise | [BibTex](): zhang2018protecting | Zhang et al, *Asia Conference on Computer and Communications Security* 2018

2. [Watermarking Deep Neural Networks for Embedded Systems](http://web.cs.ucla.edu/~miodrag/papers/Guo_ICCAD_2018.pdf)：One clear drawback of their Adi is the difficulty to associate abstract images with the author’s identity. Their answer is to use a cryptographic commitment scheme, incurring a lot of overhead to the proof of authorship; using message mark as the watermark information; unlike cloud-based MLaaS that usually charge users based on the number of queries made, there is no cost associated with querying embedded systems; owners's signature | [BibTex](): guo2018watermarking | Guo et al, *IEEE/ACM International Conference on Computer-Aided Design (ICCAD)* 2018

3. [Evolutionary Trigger Set Generation for DNN Black-Box Watermarking](https://arxiv.org/pdf/1906.04411.pdf)：proposed an evolutionary algorithmbased method to generate and optimize the trigger pattern of the backdoor-based watermark to reduce the false alarm rate. | [Code](https://github.com/guojia-git/watermarking-cnn-classifiers) | [BibTex](): guo2019evolutionary | Guo et al, 2019.6

4. [Deep Serial Number: Computational Watermarking for DNN Intellectual Property Protection](https://arxiv.org/pdf/2011.08960.pdf): we introduce the first attempt to embed a serial number into DNNs,  DSN is implemented in the knowledge distillation framework, During the distillation process, each customer DNN is augmented with a unique serial number; gradient reversal layer (GRL)  [ganin2015unsupervised] | [BibTex](): tang2020deep | Tang et al, 2020.11

5. [How to prove your model belongs to you: a blind-watermark based framework to protect intellectual property of DNN](https://arxiv.org/pdf/1903.01743.pdf)：combine some ordinary data samples with an exclusive ‘logo’ and train the model to predict them into a specific label, embedding logo into the trigger image | [BibTex](): li2019prove | Li et al, *Proceedings of the 35th Annual Computer Security Applications Conference* 2019

6. [Protecting the intellectual property of deep neural networks with watermarking: The frequency domain approach](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9343235): explain the failure to forgery attack of zhang-noise method. | [BibTex](): li2021protecting | Li et al, *19th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)* 2021


### <span id="Adversarial-Examples">Adversarial Examples</span>
1. [Adversarial frontier stitching for remote neural network watermarking](https://arxiv.org/pdf/1711.01894.pdf)：propose a novel zero-bit watermarking algorithm that makes use of adversarial model examples,  slightly adjusts the decision boundary of the model so that a specific set of queries can verify the watermark information.  | [Code](https://github.com/dunky11/adversarial-frontier-stitching) | [BibTex](): merrer2020adversarial | Merrer et al, *Neural Computing and Applications 2020* 2017.11 | [Repo by Merrer: awesome-audit-algorithms](https://github.com/erwanlemerrer/awesome-audit-algorithms): A curated list of audit algorithms for getting insights from black-box algorithms.

2. [BlackMarks: Blackbox Multibit Watermarking for Deep Neural Networks](https://arxiv.org/pdf/1904.00344.pdf)： The first end-toend multi-bit watermarking framework ; Given the owner’s watermark signature (a binary string), a set of key image and label pairs are designed using targeted adversarial attacks; provide evaluation method | [BibTex](): chen2019blackmarks | Chen et al, 2019.4


[`watermark for adv`]
1. [A Watermarking-Based Framework for Protecting Deep Image Classifiers Against Adversarial Attacks](https://openaccess.thecvf.com/content/CVPR2021W/TCV/papers/Sun_A_Watermarking-Based_Framework_for_Protecting_Deep_Image_Classifiers_Against_Adversarial_CVPRW_2021_paper.pdf): watermark is robust to adversarial noise | [BibTex](): sun2021watermarking | Sun et al, *CVPR W 2021*

2. [Watermarking-based Defense against Adversarial Attacks on Deep Neural Networks](https://faculty.ist.psu.edu/wu/papers/IJCNN.pdf): we propose a new defense mechanism that creates a knowledge gap between attackers and defenders by imposing a designed watermarking system into standard deep neural networks; introduce `randomization` | [BibTex](): liwatermarking | Li et al, 2021.4


## <span id="Clean-Image-&-Unrelated-Prediction">Clean Image & Unrelated Prediction</span> [^](#back)

1. [Robust Watermarking of Neural Network with Exponential Weighting](https://arxiv.org/pdf/1901.06151.pdf)： increase the weight value (trained on clean data) exponentially by fine-tuning on combined data set; introduce query modification attack (detect and AE) | [BibTex](): namba2019robust |  et al, *Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security (AisaCCS)* 2019

2. [DeepTrigger: A Watermarking Scheme of Deep Learning Models Based on Chaotic Automatic Data Annotation](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9264250)：fraudulent ownership claim attacks, chaotic automatic data annotation; Anti-counterfeiting | [BibTex](): zhang2020deeptrigger | Zhang et al, * IEEE Access 8* 2020

## <span id="Other-Attempts-with-Unrelated-Prediction">Other Attempts with Unrelated Prediction</span> [^](#back)
### Fidelity
1. [Protecting IP of Deep Neural Networks with Watermarking: A New Label Helps](https://link.springer.com/content/pdf/10.1007%2F978-3-030-47436-2_35.pdf):  adding a new label will not twist the original decision boundary but can help the model learn the features of key samples better;  investigate the relationship between model accuracy, perturbation strength, and key samples’ length.; reports more robust than zhang's method in pruning and | [BibTex]():  zhong2020protecting | Zhong et al; *Pacific-Asia Conference on Knowledge Discovery and Data Mining* 2020

2. [Protecting the Intellectual Properties of Deep Neural Networks with an Additional Class and Steganographic Images](https://arxiv.org/pdf/2104.09203.pdf):  use a set of watermark key samples (from another distribution) to embed an additional class into the DNN; adopt the least significant bit (LSB) image steganography to embed users’ fingerprints for authentication and management of fingerprints, 引用里有code | [BibTex](): sun2021protecting | Sun et al, 2021.4

3. [Robust Watermarking for Deep Neural Networks via Bi-level Optimization](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Robust_Watermarking_for_Deep_Neural_Networks_via_Bi-Level_Optimization_ICCV_2021_paper.pdf): inner loop phase optimizes the example-level problem to generate robust exemplars, while the outer loop phase proposes a masked adaptive optimization to achieve the robustness of the projected DNN models | yang2021robust, Yang et al, *ICCV 2021*


### Capacity
1. [‘‘Identity Bracelets’’ for Deep Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9104681)：using MNIST (unrelated to original dataset) as trigger set; exploit the discarded capacity in the intermediate distribution of DL models’ output to embed the WM information; SN is a vector that contains n decimal units where n is the number of neurons in the output layer; 同样scale的trigger set, 分析了unrelated 和 related trigger 各自的 drawback; 提到dark knowledge？; extension of zhang; 给出了evasion attack 不作考虑的原因 | [BibTex](): xu2020identity  | [Initial Version: A novel method for identifying the deep neural network model with the Serial Number](https://arxiv.org/pdf/1911.08053.pdf) | Xu et al, *IEEE Access* 2020.8

2. [Visual Decoding of Hidden Watermark in Trained Deep Neural Network](https://ieeexplore.ieee.org/abstract/document/8695386)：The proposed method has a remarkable feature for watermark detection process, which can decode the embedded pattern cumulatively and visually. 关注提取端，进行label可视化成二位图片，增加关联性 | [BibTex](): sakazawa2019visual | Sakazawa et al, * IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)* 2019

### Robustness
[post-processing]
1. [Robust and Verifiable Information Embedding Attacks to Deep Neural Networks via Error-Correcting Codes](https://arxiv.org/pdf/2010.13751.pdf)： 使用纠错码对trigger进行annotation, 分析了和现有information embedding attack 以及 model watermarking的区别； 可以recover的不只是label, 也可以是训练数据， property， 类似inference attcak | Jia, Jinyuan and Wang, Binghui and Gong, Neil Zhenqiang | [BibTex](): jia2020robust | Jia et al, 2020.10

[removal attacks] 
1. [Re-markable: Stealing watermarked neural networks through synthesis](https://link.springer.com/content/pdf/10.1007%2F978-3-030-66626-2_3.pdf): using synthesized data (iid) to retrain the target model  | Chattopadhyay, Nandish and Viroy, Chua Sheng Yang and Chattopadhyay, Anupam | chattopadhyay2020re

2. [ROWBACK: RObust Watermarking for neural networks using BACKdoors](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9680232) using adv in every layer | Chattopadhyay, Nandish and Chattopadhyay, Anupam | chattopadhyay2021rowback | ICMLA 2022 


[pre-processing]
1. [Persistent Watermark For Image Classification Neural Networks By Penetrating The Autoencoder](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9506368): enhance the robustness against AE pre-processing | li2021persistent |  Li et al, *ICIP 2021*

[model extratcion] tramer2016stealing
1. [DAWN: Dynamic Adversarial Watermarking of Neural Networks](https://arxiv.org/pdf/1906.00830.pdf): dynamically changing the responses for a small subset of queries (e.g., <0.5%) from API clients | [BibTex](): szyller2019dawn | Szyller et al, 2019,6

2. [Cosine Model Watermarking Against Ensemble Distillation](https://arxiv.org/pdf/2203.02777.pdf): [to do]


2. [Entangled Watermarks as a Defense against Model Extraction](https://arxiv.org/pdf/2002.12200.pdf)：forcing the model to learn features which are jointly used to analyse both the normal and the triggers; using soft nearest neighbor loss (SNNL) to measure entanglement over labeled data; location的确定取决于梯度，还是很全面分析了一些adaptive attack 值得进一步得阅读； outlier dection可以参考 | [Code](https://github.com/cleverhans-lab/entangled-watermark) | Jia, Hengrui and Choquette-Choo, Christopher A and Chandrasekaran, Varun and Papernot, Nicolas | [BibTex](): jia2020entangled |  et al, *30th USENIX* 2020

3. [Was my Model Stolen? Feature Sharing for Robust and Transferable Watermarks](https://openreview.net/pdf?id=XHxRBwjpEQ): 互信息的概念， T-test定义可以借鉴；watermark的location指哪些层可以用来fine-tune；P7 take home； feature extractor is prone to use a part of neurons to identify watermark samples if we directly add watermark samples into the training set.  水印数据和原始数据同分布和非同分布都可以，Jia 是OOD？ 对entangled的改进 | Tang, Ruixiang and Jin, Hongye and Wigington, Curtis and Du, Mengnan and Jain, Rajiv and Hu, Xia | tang2021my | ICLR2020 submission

4. [Effectiveness of Distillation Attack and Countermeasure on DNN watermarking](https://arxiv.org/pdf/1906.06046.pdf)：Distilling attack; countermeasure: embedding the watermark into NN in an indiret way rather than directly overfitting the model on watermark, specifically, let the target model learn the general patterns of the trigger not regarding it as noise. evaluate both embedding and trigger watermarking | [Distillation](https://arxiv.org/pdf/1503.02531.pdf): yang2019effectiveness;  *NIPS 2014 Deep Learning Workshop* | [BibTex](): yang2019effectiveness  | Yang et al, 2019.6



### Security
1. [Secure neural network watermarking protocol against forging attack](https://www.jianguoyun.com/p/DVsuU1IQ0J2UCRic_-0D)：noise-like trigger; 引入单向哈希函数，使得用于证明所有权的触发集样本必须通过连续的哈希逐个形成，并且它们的标签也按照样本的哈希值指定; 对overwriting 有其他解释: 本文针对黑盒情况下的forging attack； 如果白盒情况下两个水印同时存在，只要能提供具有单一水印的模型即可，因此简单的再添加一个水印并不构成攻击威胁; [idea]() 模型水印从差的迁移性的角度去考虑，训练的时候见过的trigger能识别但是verification的时候不能识别 | [BibTex](): zhu2020secure | Zhu et al, *EURASIP Journal on Image and Video Processing* 2020.1

2. [Piracy Resistant Watermarks for Deep Neural Networks](https://arxiv.org/pdf/1910.01226.pdf): out-of-bound values; null embedding (land into sub-area/local minimum); wonder filter | [Video](https://www.youtube.com/watch?v=yb0_GwRvF4k&ab_channel=stanfordonline) | [BibTex](): li2019piracy | Li et al, 2019.10 | [Initial version](http://web.stanford.edu/class/ee380/Abstracts/191030-paper.pdf): Persistent and Unforgeable Watermarks for Deep Neural Networks | [BibTex](): li2019persistent | Li et al, 2019.10

3. [Preventing Watermark Forging Attacks in a MLaaS Environment](https://hal.archives-ouvertes.fr/hal-03220414/): | [BibTex](): sofiane2021preventing | Lounici et al. *SECRYPT 2021, 18th International Conference on Security and Cryptography*

4. [A Protocol for Secure Verification of Watermarks Embedded into Machine Learning Models](https://dl.acm.org/doi/pdf/10.1145/3437880.3460409): choose normal inputs and watermarked inputs randomly; he whole verification process is finally formulated as a problem of Private Set Intersection (PSI), and an adaptive protocol is also introduced accordingly | [BibTex](): Kapusta2021aprotocol | Kapusta et al, *IH&MMSec 21*

### Provability 
1. [Certified Watermarks for Neural Networks](https://openreview.net/forum?id=Im43P9kuaeP)：Using the randomized smoothing technique proposed in Chiang et al., we show that our watermark is guaranteed to be unremovable unless the model parameters are changed by more than a certain `2 threshold | [BibTex](): chiang2020watermarks | Bansal et al, 2018.2

### Hardware
1. [DeepHardMark: Towards Watermarking Neural Network Hardware](): injected model, trigger, target functional blocks together to trigger the special behavior. 多加了一个硬件约束 | AAAI2022 under-review | [toto do]()


## <span id="Attempts-with-Related-Prediction">Attempts with Related Prediction </span> [^](#back)
`[image processing]`
1. [Watermarking Neural Networks with Watermarked Images](https://ieeexplore.ieee.org/document/9222304)：Image Peocessing,  exclude surrogate model attack | [BibTex](): wu2020watermarking | Wu et al, *TCSVT* 2020

2. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VVOq5e67uCEJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1YmLg:AAGBfm0AAAAAYG5egLj8-8TdhW-OrFR5PtcTAgXDBsUU&scisig=AAGBfm0AAAAAYG5egJ2W418j7bkygIvLDr7B5IUgFq1r&scisf=4&ct=citation&cd=-1&hl=en): zhang2020model | Zhang et al, *AAAI* 2020.2

3. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](): zhang2021deep | Zhang al, *TPAMI* 2021.3

4. [Exploring Structure Consistency for Deep Model Watermarking](https://arxiv.org/pdf/2108.02360.pdf): improvement against model extration attack with pre-processing | zhang2021exploring | Zhang et al, 2021.8

<!-- `[deepfake]`
1. [DeepTag: Robust Image Tagging for DeepFake Provenance](https://arxiv.org/pdf/2009.09869.pdf): using watermarked image to resist the facial manipulation; but white-box method I think,suing employ mask to embed more information, or enhance the robustness | [BibTex](): wang2020deeptag | Wang et al, 2020.9 | accepted by *ACM MM'21* [FakeTagger](https://arxiv.org/pdf/2009.09869.pdf)

2. [FaceGuard: Proactive Deepfake Detection](https://arxiv.org/pdf/2109.05673.pdf)： 对一般处理鲁棒，对deepfake过程脆弱； 但其实就是鲁棒水印的方法，对于不同后处理的组合使用强化学习进行训练，防止过拟合；但任务定义`不太清楚` | [BibTex](): yang2021faceguard | Yang, Neil Zhenqiang Gong et al, 2021.9 -->

[`text`]
1. [Watermarking the outputs of structured prediction with an application in statistical machine translation](https://www.aclweb.org/anthology/D11-1126.pdf): proposed a method to watermark the outputs of machine learning models, especially machine translation, to be distinguished from the human-generated productions. | [BibTex](): venugopal2011watermarking | Venugopal et al, *Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing* 2011

2. [Adversarial Watermarking Transformer: Towards Tracing Text Provenance with Data Hiding](https://arxiv.org/pdf/2009.03015.pdf):  towards marking and tracing the provenance of machine-generated text ; While the main purpose of model watermarking is to prove ownership and protect against model stealing or extraction, our language watermarking scheme is designed to trace provenance and to prevent misuse. Thus, it should be consistently present in the output, not only a response to a trigger set. | [BibTex](): abdelnabi2020adversarial | Abdelnabi et al, 2020.9

3. [Tracing Text Provenance via Context-Aware Lexical Substitution]() | Yang et al, AAAI'22

4. [Protecting Intellectual Property of Language Generation APIs with Lexical Watermark](https://arxiv.org/pdf/2112.02701.pdf) model extraction attack | he2021protecting | He et al, AAAI'22

## <span id="Attempts-with-Clean-Prediction">Attempts with Clean Prediction </span> [^](#back)
1. [Defending against Model Stealing via Verifying Embedded External Features](https://openreview.net/pdf?id=g6zfnWUg8A1): We embed the external features by poisoning a few training samples via style transfer; train a meta-classifier, based on the gradient of predictions; white-box; against some model steal attack | [BibTex](): zhu2021defending | Zhu et al, *ICML 2021 workshop on A Blessing in Disguise: The Prospects and Perils of Adversarial Machine Learning*, *[AAAI'22](https://arxiv.org/pdf/2112.03476.pdf)* [Yiming]



# <span id="Applications">Applications</span> [^](#back)
## <span id="Image-Processing">Image Processing</span>
1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](): zhang2020model | Zhang et al, *AAAI* 2020.2

2. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](): zhang2021deep | Zhang al, *TPAMI* 2021.3

3. [Watermarking Neural Networks with Watermarked Images](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9222304)：Image Peocessing, similar to [1] but exclude surrogate model attack | [BibTex](): wu2020watermarking | Wu et al, *TCSVT* 2020

4. [Watermarking Deep Neural Networks in Image Processing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9093125)：Image Peocessing, using the unrelated trigger pair, target label replaced by target image; inspired by Adi and deepsigns | [BibTex](): quan2020watermarking | Quan et al, *TNNLS* 2020

5. [Protecting Deep Cerebrospinal Fluid Cell Image Processing Models with Backdoor and Semi-Distillation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9647115): pattern of predicted bounding box as watermark | Li, Fang–Qi, Shi–Lin Wang, and Zhen–Hai Wang. | li2021protecting1 | DICTA 2021

## <span id="Image-Generation">Image Generation</span> [shown in fingerprints]
1. [Protecting Intellectual Property of Generative Adversarial Networks from Ambiguity Attack](https://arxiv.org/pdf/2102.04362.pdf): using trigger noise to generate trigger pattern on the original image; using passport to implenment white-box verification | Ong, Ding Sheng and Chan, Chee Seng and Ng, Kam Woh and Fan, Lixin and Yang, Qiang | ong2021protecting, One et al, *CVPR 2021*
  <br/><img src='./IP-images/220204-6.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/>

2. [Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data](https://arxiv.org/pdf/2007.08457.pdf): We first embed artificial fingerprints into training data, then validate a surprising discovery on the transferability of such fingerprints from training data to generative models, which in turn appears in the generated deepfakes; proactive method for deepfake detection; leverage [4]; cannot scale up to a large number of fingerprints | [[Empirical Study]](https://www-inst.eecs.berkeley.edu/~cs194-26/fa20/upload/files/projFinalProposed/cs194-26-aek/CS294_26_Final_Project_Write_Up.pdf) [[ICC'21 oral]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_Artificial_Fingerprinting_for_Generative_Models_Rooting_Deepfake_Attribution_in_Training_ICCV_2021_paper.pdf) | [BibTex](): yu2020artificial | Yu et al, 2020.7
  <br/><img src='./IP-images/220202-1.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/>

3. [An Empirical Study of GAN Watermarking](https://inst.eecs.berkeley.edu/~cs194-26/fa20/upload/files/projFinalProposed/cs194-26-aek/CS294_26_Final_Project_Write_Up.pdf)  | thakkarempirical | Thakkar wt al, 2020.Fall 


<!-- ## <span id="Semantic-segmentation">Semantic segmentation</span>
1. [Access Control Using Spatially Invariant Permutation of Feature Maps for Semantic Segmentation Models](https://arxiv.org/pdf/2109.01332.pdf): correct permutation with correct key | ito2021access, AprilPyone et al, 2021.9 [To do]() -->


## <span id="Automatic-Speech-Recognition(ASR)">Automatic Speech Recognition (ASR)</span>
1. [Entangled Watermarks as a Defense against Model Extraction ](https://arxiv.org/pdf/2002.12200.pdf)：forcing the model to learn features which are jointly used to analyse both the normal and the triggers (related square); using soft nearest neighbor loss (SNNL) to measure entanglement over labeled data | [Code](https://github.com/cleverhans-lab/entangled-watermark) | [BibTex](): jia2020entangled | Jia et al, *30th USENIX* 2020

2. [SpecMark: A Spectral Watermarking Framework for IP Protection of Speech Recognition Systems](https://indico2.conference4me.psnc.pl/event/35/contributions/3413/attachments/489/514/Wed-1-8-8.pdf): Automatic Speech Recognition (ASR) | [BibTex]: chen2020specmark | Chen et al, *Interspeech* 2020

3. [Speech Pattern based Black-box Model Watermarking for Automatic Speech Recognition](https://arxiv.org/pdf/2110.09814.pdf) | [BibTex](): chen2021speech | Chen et al, 2021.10

4. [Watermarking of Deep Recurrent Neural Network Using Adversarial Examples to Protect Intellectual Property](https://www.tandfonline.com/doi/pdf/10.1080/08839514.2021.2008613): speech-to-text RNN model, based on adv example | Rathi, Pulkit and Bhadauria, Saumya and Rathi, Sugandha | rathi2021watermarking | Applied Artificial Intelligence, 2021

5. [Protecting the Intellectual Property of Speaker Recognition Model by Black-Box Watermarking in the Frequency Domain](https://www.mdpi.com/2073-8994/14/3/619/html): adding a trigger signal in the frequency domain; a new label is assigned | Yumin Wang and Hanzhou Wu | [to do]

## <span id="NLP">NLP</span> [[link]](https://blog.csdn.net/weixin_44517291/article/details/115508909)
### <span id="Sentence-Classification">`Sentence Classification`</span>
1. [Watermarking Neural Language Models based on Backdooring](https://github.com/TIANHAO-WANG/nlm-watermark/blob/master/nlpwatermark.pdf): sentiment analysis (sentence-iunputted calssification task) | Fu et al, 2020.12

2. [Robust Black-box Watermarking for Deep Neural Network using Inverse Document Frequency](https://arxiv.org/pdf/2103.05590.pdf): modified text as trigger;  divided into the following three categories: Watermarking the training data, network's parameters, model's output; Dataset: IMDB users' reviews sentiment analysis, HamSpam spam detraction | [BibTex](): yadollahi2021robust | Yadollahi et al, 2021.3

### <span id="Machine-Translation">`Machine-Translation`</span>
1. [Yes We can: Watermarking Machine Learning Models beyond Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9505220) attack can be referenced; watermark forging? | lounici2021yes | Lounici wt al, *2021 IEEE 34th Computer Security Foundations Symposium (CSF)*  | machine translation 

1. [Protecting Your NLG Models with Semantic and Robust Watermarks](https://openreview.net/pdf?id=VuW5ojKGI43): re | xiang2021protecting | Xiang et al, ICLR 2022 withdraw  [New Arxiv](https://arxiv.org/pdf/2112.05428.pdf)

### <span id="Language-Generation">`Language Generation`</span>
1. [Watermarking the outputs of structured prediction with an application in statistical machine translation](https://www.aclweb.org/anthology/D11-1126.pdf): proposed a method to watermark the outputs of machine learning models, especially machine translation, to be distinguished from the human-generated productions. | [BibTex](): venugopal2011watermarking | Venugopal et al, *Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing* 2011

2. [Adversarial Watermarking Transformer Towards Tracing Text Provenance with Data Hiding](https://arxiv.org/pdf/2009.03015.pdf): 和模型水印对于攻击的定义有所不同 towards marking and tracing the provenance of machine-generated text; our language watermarking scheme is designed to trace provenance and to prevent misuse. Thus, it should be consistently present in the output, not only a response to a trigger set. Transformer| [BibTex](): abdelnabi2020adversarial | Abdelnabi et al, 2020.9 *S&P21*

3. [Tracing Text Provenance via Context-Aware Lexical Substitution](https://arxiv.org/pdf/2112.07873.pdf) | Yang et al, AAAI'22

4. [Protecting Intellectual Property of Language Generation APIs with Lexical Watermark](https://arxiv.org/pdf/2112.02701.pdf) model extraction attack;machine translation, document summarizartion, image captioning | [BibTex](): he2021protecting | He et al, AAAI'22

## <span id="Image-Captioning">Image Captioning</span>
1. [Protect, Show, Attend and Tell: Empower Image Captioning Model with Ownership Protection](https://arxiv.org/pdf/2008.11009.pdf)：Image Caption | [BibTex](): lim2020protect  | Lim et al, 2020.8 (surrogate model attck) | [Pattern Recognition](https://reader.elsevier.com/reader/sd/pii/S0031320321004659?token=A0BD59797E19F9A8A58426F36C1075E497A757D5E5990907924D2B4FE9217F3C5382B02807A5C5308C81C90F14944567&originRegion=us-east-1&originCreation=20210905073628) *2021.8*

2. [Protecting Intellectual Property of Language Generation APIs with Lexical Watermark](https://arxiv.org/pdf/2112.02701.pdf) model extraction attack;machine translation, document summarizartion, image captioning | [BibTex](): he2021protecting | He et al, AAAI'22


## <span id="3D-&-Graph">3D & Graph</span>
1. [Passport-aware Normalization for Deep Model Protection](https://proceedings.neurips.cc/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf): Improvemnet of [1] | [Code](https://github.com/ZJZAC/Passport-aware-Normalization) | [BibTex](): zhang2020passport | Zhang et al, *NeuraIPS* 2020, 2020.9

1. [Watermarking Graph Neural Networks by Random Graphs](https://arxiv.org/pdf/2011.00512.pdf): Graph Neural Networks (GNN); focus on node classification task | [BibTex](): zhao2020watermarking | Zhao et al, *Interspeech* 2020

2. [Watermarking Graph Neural Networks based on Backdoor Attacks](https://arxiv.org/pdf/2110.11024.pdf): both graph and node classification tasks; backdoor-style; considering suspicious models with different architecture | [BibTex](): xu2021watermarking | Xu et al, 2021.10


## <span id="Federated-Learning">Federated Learning</span>
1. [WAFFLE: Watermarking in Federated Learning](https://arxiv.org/pdf/2011.00512.pdf): WAFFLE leverages capabilities of the aggregator to embed a backdoor-based watermark by re-training the global model with the watermark during each aggregation round. considering evasion attack by a detctor, watermark forging attack; client might be malicious, clients not involved in watermarking, and have no access to watermark set. | [BibTex](): atli2020waffle | Atli et al, 2020.8

<!-- 2. [Watermarking Federated Deep Neural Network Models](https://aaltodoc.aalto.fi/bitstream/handle/123456789/43561/master_Xia_Yuxi_2020.pdf?sequence=1): for degree of master, advisor: Buse Atli | [BibTex](): xia2020watermarking | Xia et al, 2020 -->

2. [FedIPR: Ownership Verification for Federated Deep Neural Network Models](http://federated-learning.org/fl-ijcai-2021/FTL-IJCAI21_paper_21.pdf): claim legitimate intellectual property rights (IPR) of FedDNN models, in case that models are illegally copied, re-distributed or misused. | fan2021fedipr, Fan et al, *FTL-IJCAI 2021*

3. [Towards Practical Watermark for Deep Neural Networks in Federated Learning](https://arxiv.org/pdf/2105.03167.pdf): we demonstrate a watermarking protocol for protecting deep neural networks in the setting of FL. [Merkle-Sign: Watermarking Framework for Deep Neural Networks in Federated Learning] | [BibTex](): li2021towards | Li et al, 2021.5

<!-- 5. [A novel approach to simultaneously improve privacy, efficiency and reliability of federated DNN learning](http://federated-learning.org/fl-ijcai-2021/FTL-IJCAI21_paper_31.pdf): demonstrate that embedding passports resist deep leakage attack and model inversion attack; adding passport can confuse the reconstruction | gunovel, Gu et al, *FTL-IJCAI 2021* -->

4. [Secure Federated Learning Model Verification: A Client-side Backdoor Triggered Watermarking Scheme](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9658998): Secure FL framework is developed to address data leakage issue when central node is not fully trustable; backdoor in encrypted way | Liu, Xiyao and Shao, Shuo and Yang, Yue and Wu, Kangming and Yang, Wenyuan and Fang, Hui | liu2021secure | SMC'21



## <span id="Deep-Reinforcement-Learning">Deep Reinforcement Learning</span>

1. [Sequential Triggers for Watermarking of Deep Reinforcement Learning Policies](https://arxiv.org/pdf/1906.01126.pdf): experimental evaluation of watermarking a DQN policy trained in the Cartpole environment | [BibTex](): behzadan2019sequential | Behzadan et al, 2019,6

1. [Yes We can: Watermarking Machine Learning Models beyond Classification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9505220) | lounici2021yes, Lounici wt al, *2021 IEEE 34th Computer Security Foundations Symposium (CSF)* 

2. [Temporal Watermarks for Deep Reinforcement Learning Models](https://personal.ntu.edu.sg/tianwei.zhang/paper/aamas2021.pdf): damage-free (related) states | [BibTex](): chen2021temporal | Chen et al, *International Conference on Autonomous Agents and Multiagent Systems* 2021


## <span id="Transformer">Transformer</span>
1. [Protecting Your Pretrained Transformer: Data and Model Co-designed Ownership Verification](): CVPR'22 under review [toto do]()

## <span id="Pretrained-Encoders">Pretrained Encoders</span>
1. [StolenEncoder: Stealing Pre-trained Encoders](https://arxiv.org/pdf/2201.05889.pdf): ImageNet encoder, CLIP encoder, and Clarifai’s General Embedding encoder | Yupei Liu, Jinyuan Jia, Hongbin Liu, Neil Zhenqiang Gong | liu2022stolenencoder | 2022.1

2. [Can’t Steal? Cont-Steal! Contrastive Stealing Attacks Against Image Encoders](https://arxiv.org/pdf/2201.07513.pdf) | Sha, Zeyang and He, Xinlei and Yu, Ning and Backes, Michael and Zhang, Yang | sha2022can | 2022.1

3. [Watermarking Pre-trained Encoders in Contrastive Learning](https://arxiv.org/pdf/2201.08217.pdf): [Selling](https://twimlai.com/solutions/features/model-marketplace/) pre-trained encoders;  introduce a task-agnostic loss function to effectively embed into the encoder a backdoor as the watermark. | Wu, Yutong and Qiu, Han and Zhang, Tianwei and Qiu, Meikang | wu2022watermarking | 2022.1
  <br/><img src='./IP-images/220122-1.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/>

4. [SSLGuard: A Watermarking Scheme for Self-supervised Learning Pre-trained Encoders](https://arxiv.org/pdf/2201.11692.pdf): uses optimized verification dataset a decoder to extract copyright information to the stolen encoder and a surrogate model is involved during the watermark embedding stage. | Cong, Tianshuo and He, Xinlei and Zhang, Yang | cong2022sslguard | 2022.1

## <span id="Dataset">Dataset</span>
1. [Radioactive data tracing through training](https://arxiv.org/pdf/2002.00937.pdf): craft a class-specific additive mark in the latent space before the classification layer | sablayrolles2020radioactive | Sablayrolles et al, *ICML* 2020

2. [On the Effectiveness of Dataset Watermarking in Adversarial Settings](https://arxiv.org/pdf/2202.12506.pdf): 对radioactive data的分析 [to do]

2. [Open-sourced Dataset Protection via Backdoor Watermarking](https://arxiv.org/pdf/2010.05821.pdf): use a hypothesis test guided method for dataset verification based on the posterior probability generated by the suspicious third-party model of the benign samples and their correspondingly watermarked samples  | [BibTex](): li2020open | Li ea el, *NeurIPS Workshop on Dataset Curation and Security* 2020 [Yiming]

3. [Anti-Neuron Watermarking: Protecting Personal Data Against Unauthorized Neural Model Training](https://arxiv.org/pdf/2109.09023.pdf): utilize linear color transformation as shift of the private dataset. | zou2021anti | Zou et al, 2021.9
 
4. [Dataset Watermarking](https://aaltodoc.aalto.fi/bitstream/handle/123456789/109322/master_Hoang_Minh_2021.pdf?sequence=1&isAllowed=y): 介绍了fingerprinting, dataset inference， dataset watermarking;Alto, examination for the degree of master | Hoang, Minh and others | hoang2021dataset |  2021.8 [emperical]()

5. [Data Protection in Big Data Analysis](https://uwspace.uwaterloo.ca/bitstream/handle/10012/17319/Shafieinejad_Masoumeh.pdf?sequence=1&isAllowed=y): 一些数据加密，和隐私保护；examination for Ph.D | Masoumeh Shafieinejad, 2021.8


# <span id="Idetification-Tracing">Identificiton Tracing</span> [^](#back)

**阐明fingerprints和fingerprinting的不同：一个类似相机噪声，设备指纹；一个是为了进行用户追踪的分配指纹，序列号**
- [Fingerprinting vs. Watermarking](https://www.plagiarismtoday.com/2007/10/09/watermarking-vs-fingerprinting-a-war-in-terminology/)

- [wikipedia](https://en.wikipedia.org/wiki/Fingerprint_(computing))

## <span id="Fingerprints">Fingerprints</span> [^](#back)

### <span id="Data">Boundary</span>
[adversarail example]
1. [AFA Adversarial fingerprinting authentication for deep neural networks](https://www.sciencedirect.com/science/article/abs/pii/S014036641931686X)：Use the adversarial examples as the model’s fingerprint； also mimic the `logits vector` of the target sample 𝑥𝑡; ghost model [li2020learning] (directly modify) substitute refrence models | [BibTex](): zhao2020afa | Zhao et al, * Computer Communications* 2020
  <br/><img src='./IP-images/220203-3.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

2. [Fingerprinting Deep Neural Networks - A DeepFool Approach](https://ieeexplore.ieee.org/document/9401119): In this paper, we utilize the geometry characteristics inherited in the DeepFool algorithm to extract data points near the classification boundary; execution time independent of dataset  | Wang, Si and Chang, Chip-Hong | wang2021fingerprinting | Wang et al, *IEEE International Symposium on Circuits and Systems (ISCAS)* 2021

3. [Deep neural network fingerprinting by conferrable adversarial examples](https://arxiv.org/pdf/1912.00888.pdf): conferrable adversarial examples that exclusively transfer with a target label from a source model to its surrogates, using refrence model | [BibTex](): lukas2019deep | Lukas et al, *ICLR* 2021
  <br/><img src='./IP-images/220203-4.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

4. [Characteristic Examples: High-Robustness, Low-Transferability Fingerprinting of Neural Networks](https://www.ijcai.org/proceedings/2021/0080.pdf):  we use random initialization instead of true data and therefore our method is data-free; using high frequency to constrain the transferablity | Wang, Siyue and Wang, Xiao and Chen, Pin-Yu and Zhao, Pu and Lin, Xue | wang2021characteristic | *IJCAI2021*
<br/><img src='./IP-images/220203-5.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

5. [Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations](https://arxiv.org/pdf/2202.08602.pdf): UAPs,  contrastive learning | Peng, Zirui and Li, Shaofeng and Chen, Guoxing and Zhang, Cheng and Zhu, Haojin and Xue, Minhui |　2022.2 [to do]




[boundary example]

1. [IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary](https://arxiv.org/pdf/1910.12903.pdf): data points near the classification boundary of the model owner’s classifier (either on or far away), 找对于两个预测都很近的点; only identify verification in exp | [BibTex](): cao2019ipguard | Cao, Xiaoyu and Jia, Jinyuan and Gong, Neil Zhenqiang | *AsiaCCS* 2021
  <br/><img src='./IP-images/220203-2.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

6. [Forensicability of Deep Neural Network Inference Pipelines](): identification of the execution environment (software & hardware) used to produce deep neural network predictions. Finally, we introduce boundary samples that amplify the numerical deviations in order to distinguish machines by their predicted label only. | [BibTex](): schlogl2021forensicability | Schlogl et al, *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)* 2021 

7. [iNNformant: Boundary Samples as Telltale Watermarks](https://informationsecurity.uibk.ac.at/pdfs/SKB2021_IH.pdf): Improvement of [schlogl2021forensicability]; This is relevant if, in the above example, the model owner wants to probe the inference pipeline inconspicuously in order to avoid that the licensee can process obvious boundary samples in a different pipeline (the legitimate one) than the bulk of organic samples. We propose to generate transparent boundary samples as perturbations of natural input samples and measure the distortion by the peak  signal-to-noise ratio (PSNR). | [BibTex](): schlogl2021innformant | Schlogl et al, * IH&MMSEC '21* 2021
<br/><img src='./IP-images/220204-5.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

4. [Integrity Fingerprinting of DNN with Double Black-box Design and Verification](https://arxiv.org/pdf/2203.10902.pdf): h captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model’s decision boundary in order to capture the inherent fingerprints of the model [to do]
<br/><img src='./IP-images/220329-1.png' align='left' style=' width:500px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

[more application]
1. [A Novel Verifiable Fingerprinting Scheme for Generative Adversarial Networks](https://arxiv.org/pdf/2106.11760.pdf): image 后面再加一个classifier, 使用adv，这样生成的adv会更不可见一点 | [BibTex](): li2021novel | Li et al, 2021.6

2. [UTAF: A Universal Approach to Task-Agnostic Model Fingerprinting](https://arxiv.org/pdf/2201.07391.pdf)  | Pan, Xudong and Zhang, Mi and Yan, Yifan | pan2022utaf | 2022.1
<br/><img src='./IP-images/220204-4.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

3. [TAFA: A Task-Agnostic Fingerprinting Algorithm for Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-88418-5_26):  on a variety of downstream tasks including classification, regression and generative  modeling, with no assumption on training data access. [rolnick2020reverse] | Pan, Xudong and Zhang, Mi and Lu, Yifan and Yang, Min | pan2021tafa |  *European Symposium on Research in Computer Security 2021 (B类)*
<br/><img src='./IP-images/220204-2.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

[interesting]
- [Boundary Defense Against Black-box Adversarial Attacks](https://arxiv.org/pdf/2201.13444.pdf): Our method detects the boundary samples as those with low classification confidence and adds white Gaussian noise to their logits.



### <span id="Inference">Inference</span> [^](#back)
[outputs]
1. [Do gans leave artificial fingerprints?](https://arxiv.org/pdf/1812.11842.pdf): visualize GAN fingerprints motivated by PRNU, extract noise residual (unrelated to the image semantics) and show their application to GAN source identification | [BibTex](): marra2019gans | Marra et al, *IEEE Conference on Multimedia Information Processing and Retrieval (MIPR)* 2019

2. [Leveraging frequency analysis for deep fake image recognition](http://proceedings.mlr.press/v119/frank20a/frank20a.pdf): DCT domain, these artifacts are consistent across different neural network architectures, data sets, and resolutions (不易区分相同结构？) | frank2020leveraging | Frank et al | ICML'20

3. [Attributing Fake Images to GANs: Learning and Analyzing GAN Fingerprints](https://arxiv.org/pdf/1811.08180.pdf): We replace their hand-crafted fingerprint (of [1]) formulation with a learning-based one, `decoupling` model fingerprint from image fingerprint, and show superior performances in a variety of experimental conditions. | [Supplementary Material](https://openaccess.thecvf.com/content_ICCV_2019/supplemental/Yu_Attributing_Fake_Images_ICCV_2019_supplemental.pdf) | [Code](https://github.com/ningyu1991/GANFingerprints) | [Ref Code](https://github.com/cleverhans-lab/deepfake_attribution) | [BibTex]: yu2019attributing | [Homepage](https://ningyu1991.github.io/) | Yu et al, *ICCV* 2019
<br/><img src='./IP-images/220204-3.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

4. [Learning to Disentangle GAN Fingerprint for Fake Image Attribution](https://arxiv.org/pdf/2106.08749.pdf): the extracted features could include many content-relevant components and generalize poorly on unseen images with different content | [BibTex](): c | Yang et al, 2021.6
  <br/><img src='./IP-images/220203-1.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

5. [Your Model Trains on My Data? Protecting Intellectual Property of Training Data via Membership Fingerprint Authentication](https://www.chenwang.net.cn/publications/MeFA-TIFS22.pdf)：使用MIA做 fingerprint [to do]





[pair outputs]
1. [Teacher Model Fingerprinting Attacks Against Transfer Learning](https://arxiv.org/pdf/2106.12478.pdf): [latent backdoor](https://dl.acm.org/doi/pdf/10.1145/3319535.3354209) 生成一个和prob样本输出类似的样本，如果模型对这两个成对的数据都是相似的response,证明用来原始的feature extractor，和我想的成对的trigger 异曲同工 | Chen, Yufei and Shen, Chao and Wang, Cong and Zhang, Yang | chen2021teacher, Chen et al, 2021.6

[inference time]
1. [Fingerprinting Multi-exit Deep Neural Network Models Via Inference Time](https://arxiv.org/pdf/2110.03175.pdf): we propose a novel approach to fingerprint multi-exit models via **inference time** rather than inference predictions. | Dong, Tian and Qiu, Han and Zhang, Tianwei and Li, Jiwei and Li, Hewu and Lu, Jialiang | dong2021fingerprinting | 2021.10 

<!-- [deployment enviroment]
1. [DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks](http://cseweb.ucsd.edu/~jzhao/files/DeepAttest-isca2019.pdf): the first on-device DNN attestation method that certifies the legitimacy of the DNN program mapped to the device; device-specific fingerprint | [BibTex](): chen2019deepattest | Chen et al, *ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)* 2019 -->

### <span id="Training">Training</span> [^](#back)
1. [Proof-of-Learning: Definitions and Practice](https://arxiv.org/pdf/2103.05633.pdf): 证明训练过程的完整性，要求：验证花费小于训练花费，训练花费小于伪造花费；通过特定初始化下，梯度更新的随机性，以及逆向highly costly, 来作为交互验证的信息。可以用来做模型版权和模型完整性认证(分布训练，确定client model 是否trusty) | [Code](https://github.com/cleverhans-lab/Proof-of-Learning) | [BibTex](): jia2021proof | Jia et al, *42nd S&P* 2021.3

2. [“Adversarial Examples” for Proof-of-Learning](https://arxiv.org/pdf/2108.09454.pdf): we show that PoL is vulnerable to “adversarial examples”! | Zhang, Rui and Liu, Jian and Ding, Yuan and Wu, Qingbiao and Ren, Kui | zhang2021adversarial | 2021.8

2. [Towards Smart Contracts for Verifying DNN Model Generation Process with the Blockchain](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9403138): we propose a smart contract that is based on the dispute resolution protocol for verifying DNN model generation process. | [BibTex](): seike2021towards | Seike et al, *IEEE 6th International Conference on Big Data Analytics (ICBDA)* 2021

**阐明fingerprints和fingerprinting的不同：一个类似相机噪声，设备指纹；一个是为了进行用户追踪的分配指纹，序列号**
## <span id="Fingerprinting">DNN Fingerprinting</span> [^](#back)
1. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID); The main difference between watermarking and fingerprinting is that the WM remains the same for all copies of the IP while the FP is unique for each copy. As such, FPs address the ambiguity of WMs and enables tracking of IP misuse conducted by a specific user. | [BibTex](): chen2019deepmarks | Chen et al, *ICMR* 2019

2. [A Deep Learning Framework Supporting Model Ownership Protection and Traitor Tracing](https://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=6917&context=sis_research): [Collusion-resistant fingerprinting for multimedia] | [BibTex](): xu2020deep | Xu et al, *2020 IEEE 26th International Conference on Parallel and Distributed Systems (ICPADS)*
  <br/><img src='./IP-images/220204-1.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

3. [Mitigating Adversarial Attacks by Distributing Different Copies to Different Users](https://arxiv.org/pdf/2111.15160.pdf)： induce different sets of dversarial samples in different copies in a more controllable manner; 为了防止相同分发模型直接进行对抗攻击；也可用于attack tracing; based on [attractors](https://arxiv.org/pdf/2003.02732.pdf) | [BibTex](): zhang2021mitigating | Zhang et al, 2021.11.30


4. [Responsible Disclosure of Generative Models Using Scalable Fingerprinting](https://arxiv.org/pdf/2012.08726.pdf): 使fingerprints和watermark都可用, after training one generic fingerprinting model, we can instantiate a large number of generators adhoc with different fingerprints; conditional GAN, FP code as condition | [BibTex](): yu2020responsible | Yu et al, 2020.12
  <br/><img src='./IP-images/220202-2.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/><br/>

5. [Decentralized Attribution of Generative Models](https://arxiv.org/pdf/2010.13974.pdf): To redmey the `non-scalibility` of [Yu'ICCV 19]; Each binary classifier is parameterized by a user-specific key and distinguishes its associated model distribution from the authentic data distribution. We develop sufficient conditions of the keys that guarantee an attributability lower bound.| [Code](https://github.com/ASU-Active-Perception-Group/decentralized_attribution_of_generative_models) | [BibTex](): kim2020decentralized | Kim, Changhoon and Ren, Yi and Yang, Yezhou | *ICLR* 2021

6. [Attributable Watermarking Of Speech Generative Models](https://arxiv.org/pdf/2202.08900.pdf): model attribution, i.e., the classification of synthetic contents by their source models via watermarks embedded in the contents; 和图片的GAN attribution一样，可以把model attribution 单独分一个section？ generative model attribution？ [to do]


[作为access control]
1. [Active DNN IP Protection: A Novel User Fingerprint Management and DNN Authorization Control Technique](https://www.jianguoyun.com/p/DdZ92TMQ0J2UCRjt4O0D): using trigger sets as copyright management | [BibTex](): xue2020active | Xue et al, *Security and Privacy in Computing and Communications (TrustCom)* 2020

2. [ActiveGuard: An Active DNN IP Protection Technique via Adversarial Examples](https://www.jianguoyun.com/p/DdZ92TMQ0J2UCRjt4O0D): different compared with [xue2020active]: adversarial  example based； 考虑了Users’ fingerprints allocation 的问题| [update](https://practical-dl.github.io/long_paper/26.pdf) | [BibTex](): xue2021activeguard | Xue et al, 2021.3 



# <span id="Integrity-verification">Integrity verification</span> [^](#back)
The user may want to be sure of the provenance fo the model in some security applications or senarios

[sensitive sample]
1. [Verideep: Verifying integrity of deep neural networks through sensitive-sample fingerprinting](https://arxiv.org/pdf/1808.03277.pdf): initial version of [he2019sensitive] | [BibTex](): he2018verideep | He et al, 2018.8

2. [Sensitive-Sample Fingerprinting of Deep Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.pdf): we define Sensitive-Sample fingerprints, which are a small set of human unnoticeable transformed inputs that make the model outputs sensitive to the model’s parameters. | [BibTex](): he2019sensitive | He et al, *CVPR* 2019

3. [Verification of integrity of deployed deep learning models using Bayesian optimization](https://www.sciencedirect.com/science/article/pii/S0950705122000703): handle both small and large weights pertubation | Kuttichira, Deepthi Praveenlal and Gupta, Sunil and Nguyen, Dang and Rana, Santu and Venkatesh, Svetha | kuttichira2022verification | Knowledge-Based Systems,2022

4. [Sensitive Samples Revisited: Detecting Neural Network Attacks Using Constraint Solvers](https://arxiv.org/pdf/2109.03966.pdf): He 的方法需要假设凸函数特性，对He的方法选择以一种新的优化求解方法；**其中的故事描述可借鉴，e.g., to be trusted** | Docena, Amel Nestor and Wahl, Thomas and Pearce, Trevor and Fei, Yunsi | docena2021sensitive  | 2021.9

5. [TamperNN: Efficient Tampering Detection of Deployed Neural Nets](https://arxiv.org/pdf/1903.00317.pdf): In the remote interaction setup we consider, the proposed strategy is to identify markers of the model input space that are likely to change class if the model is attacked, allowing a user to detect a possible tampering. | [BibTex](): merrer2019tampernn | Merrer et al, *IEEE 30th International Symposium on Software Reliability Engineering (ISSRE)* 2019

6. [Fragile Neural Network Watermarking with Trigger Image Set](https://link.springer.com/content/pdf/10.1007%2F978-3-030-82136-4_23.pdf): The watermarked model is sensitive to malicious fine tuning and will produce unstable classification results of the trigger images. | zhu2021fragile, Zhu et al, *International Conference on Knowledge Science, Engineering and Management 2021 (KSEM)*

[Fragility]
1. [MimosaNet: An Unrobust Neural Network Preventing Model Stealing](https://arxiv.org/pdf/1907.01650.pdf): . In this paper, we propose a method for creating an equivalent version of an already trained fully connected deep neural network that can prevent network stealing: namely, it produces the same responses and classification accuracy, but it is extremely sensitive to weight changes; focus on three consecutive FC layer | [BibTex](): szentannai2019mimosanet | Szentannai et al, 2019.7

2. [DeepiSign: Invisible Fragile Watermark to Protect the Integrity and Authenticity of CNN](https://arxiv.org/pdf/2101.04319.pdf): convert to DCT domain, choose the high frequency to adopt LSB for information hiding， To verify
the integrity and authenticity of the model | [BibTex](): abuadbba2021deepisign | Abuadbba et al, *SAC* 2021

3. [NeuNAC: A Novel Fragile Watermarking Algorithm for Integrity Protection of Neural Networks](https://reader.elsevier.com/reader/sd/pii/S0020025521006642?token=69900AD74DF75BCD44D9AF0ACB66CF250754FF4F8DB188F9A7A11CDB06EBFF2F114E90EE6440F4AB1629A0878D2C9455&originRegion=us-east-1&originCreation=20211129131257): white-box | [BibTex](): botta2021neunac | Botta et al, *Information Sciences (2021)* 

    [Reversible]
1. [Reversible Watermarking in Deep Convolutional Neural Networks for Integrity Authentication](https://arxiv.org/pdf/2101.04319.pdf): chose the least important weights as the cover, can reverse the original model performance, can authenticate the integrity | [Reversible data hiding](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.219.4172&rep=rep1&type=pdf) | [BibTex](): guan2020reversible | Guan et al, *ACM MM* 2020


[others]
1. [SafetyNets: verifiable execution of deep neural networks on an untrusted cloud](https://proceedings.neurips.cc/paper/2017/file/6048ff4e8cb07aa60b6777b6f7384d52-Paper.pdf): Specifically, SafetyNets develops and implements a  specialized interactive proof (IP) protocol for verifiable execution of a class of deep neural networks,  | [BibTex](): ghodsi2017safetynets | Ghodsi et al, *Proceedings of the 31st International Conference on Neural Information Processing Systems. 2017* [to do]

<!-- 3. [Multipurpose Watermarking Approach for Copyright and Integrity of Steganographic Autoencoder Models](https://downloads.hindawi.com/journals/scn/2021/9936661.pdf): store the hash value of each block in the redundant bits of the FC layer | gu2021multipurpose | Security and Communication Networks,2021 -->

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



# <span id="Evaluation">Evaluation</span>

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

##  
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

##  
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

## Evaluation
1. [Practical Evaluation of Neural Network Watermarking Approaches](https://www.mi.fu-berlin.de/inf/groups/ag-idm/theseses/2021-Tim-von-Kaenel.pdf): [To do]()

2. [SoK: How Robust is Deep Neural Network Image Classification Watermarking?](https://arxiv.org/pdf/2108.04974.pdf#page=16&zoom=100,416,614): | Lukas, et al, *S&P2022* | [[Toolbox](https://github.com/dnn-security/Watermark-Robustness-Toolbox)]

3. [Copy, Right? A Testing Framework for Copyright Protection of Deep Learning Models](https://arxiv.org/pdf/2112.05588.pdf) | Jialong Chen et al, *S&P2022* 

## Reference:
1. 数字水印技术及应用2004（孙圣和）1.7.1 评价问题
2. 数字水印技术及其应用2018（楼偶俊） 2.3 数字水印系统的性能评价
3. 数字水印技术及其应用2015（蒋天发）1.6 数字水印的性能评测方法
4. [Digital Rights Management The Problem of Expanding Ownership Rights](https://books.google.ca/books?id=IgSkAgAAQBAJ&lpg=PP1&ots=tA7ZrVoYx-&dq=Digital%20Rights%20Management%20The%20Problem%20of%20Expanding%20Ownership%20Rights&lr&pg=PA16#v=onepage&q=Digital%20Rights%20Management%20The%20Problem%20of%20Expanding%20Ownership%20Rights&f=false)

## <span id="Robustness-&-Security">Robustness & Security</span>

1. [Forgotten siblings: Unifying attacks on machine learning and digital watermarking](https://www.sec.cs.tu-bs.de/pubs/2018-eurosp.pdf): The two research communities have worked in parallel so far, unnoticeably developing similar attack and defense strategies. This paper is a first effort to bring these communities together. To this end, we present a unified notation of blackbox attacks against machine learning and watermarking. | [Cited by](): [Protecting artificial intelligence IPs: a survey of watermarking and fingerprinting for machine learning] | [BibTex](): quiring2018forgotten | Quiring et al, *IEEE European Symposium on Security and Privacy (EuroS&P)* 2018 

2. [Evaluating the Robustness of Trigger Set-Based Watermarks Embedded in Deep Neural Networks](https://arxiv.org/pdf/2106.10147.pdf): FT, model stealing, parameter pruning, evasion, 

## Model Modifications
### Fine-tuning 
1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082)：第一篇模型水印工作 | [Code](https://github.com/yu4u/dnn-watermark) | [BibTex]): uchida2017embedding | Uchia et al, *ICMR* 2017.1

### Model Pruning or Parameter Pruning
1. [DeepSigns: An End-to-End Watermarking Framework for Protecting the Ownership of Deep Neural Networks](http://www.aceslab.org/sites/default/files/deepsigns.pdf)：using activation map as cover | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](): rouhani2019deepsigns | Rouhani et al, *ASPLOS* 2019

### Model Compression
1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082)：第一篇模型水印工作 | [Code](https://github.com/yu4u/dnn-watermark) | [Pruning](https://arxiv.org/pdf/1510.00149)): han2015deep; *ICLR* 2016 | [BibTex]): uchida2017embedding | Uchia et al, *ICMR* 2017.1

### Model Retraining
1. [IPGuard: Protecting Intellectual Property of Deep Neural Networks via Fingerprinting the Classification Boundary](https://arxiv.org/pdf/1910.12903.pdf): Based on this observation, IPGuard extracts some data points near the classification boundary of the model owner’s classifier and uses them to fingerprint the classifier  | [BibTex](): cao2019ipguard | Cao et al, *AsiaCCS* 2021

2. [Robust Watermarking of Neural Network with Exponential Weighting](https://arxiv.org/pdf/1901.06151.pdf)：using original training data with wrong label as triggers; increase the weight value exponentially so that model modification cannot change the prediction behavior of samples (including key samples) before and after model modification; introduce query modification attack, namely, pre-processing to query | [BibTex](): namba2019robust |  et al, *Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security (AisaCCS)* 2019

## Removal Attack 
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

<!-- 13. [Watermarking in Deep Neural Networks via Error Back-propagation](https://www.ingentaconnect.com/contentone/ist/ei/2020/00002020/00000004/art00003?crawler=true&mimetype=application/pdf)：using an independent network (weights selected from the main network) to embed and extract watermark; provide some suggestions for watermarking; **introduce model isomorphism attack** | [BibTex](): wang2020watermarking | Wang et al, *Electronic Imaging* 2020.4 -->

14. [Detect and remove watermark in deep neural networks via generative adversarial networks](https://arxiv.org/pdf/2106.08104.pdf):  backdoorbased DNN watermarks are vulnerable to the proposed GANbased watermark removal attack, like Neural Cleanse, replacing the optimized method with GAN | [BibTex](): wang2021detect | Wang et al, 2021.6 | [To do]()

15. [Fine-tuning Is Not Enough: A Simple yet Effective Watermark Removal Attack for DNN Models](https://personal.ntu.edu.sg/tianwei.zhang/paper/ijcai21.pdf) | [BibTex](): guofine | *IJCAI 2021*

16. [Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks](https://arxiv.org/pdf/2101.05930.pdf) | [BibTex](): li2021neural | Li et al, *ICLR 2021*

## Collusion Attack
1. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID) | [BibTex](): chen2019deepmarks | Chen et al, *ICMR* 2019


## Evasion Attack  
Query-modification (like detection-based defense) 
1. [Evasion Attacks Against Watermarking Techniques found in MLaaS Systems](https://www.researchgate.net/profile/Dorjan-Hitaj/publication/334698259_Evasion_Attacks_Against_Watermarking_Techniques_found_in_MLaaS_Systems/links/5dd6a6e692851c1feda559db/Evasion-Attacks-Against-Watermarking-Techniques-found-in-MLaaS-Systems.pdf)：ensemble prediction based on voting-mechanism | [BibTex](): hitaj2019evasion | Hitaj et al, *Sixth International Conference on Software Defined Systems (SDS)* 2019 | [Initial Version: Have You Stolen My Model? Evasion Attacks Against Deep Neural Network Watermarking Techniques](https://arxiv.org/pdf/1809.00615.pdf)


2. [An Evasion Algorithm to Fool Fingerprint Detector for Deep Neural Networks](https://crad.ict.ac.cn/EN/article/downloadArticleFile.do?attachType=PDF&id=4415): ． 该逃避算法的核心是设计了一个指纹样本检测器— —— FingerprintＧGAN． 利用生成对抗网络(generative adversarial network,) 原理,学习正常样本在隐空间的特征表示及其分布,根据指纹 GAN 样本与正常样本在隐空间中特征表示的差异性,检测到指纹样本,并向目标模型所有者返回有别于预测的标签,使模型所有者的指纹比对方法失效. | [BibTex](): yaguan2021evasion | Qian et al, *Journal of Computer Research and Development* 2021

3. [Persistent Watermark For Image Classification Neural Networks By Penetrating The Autoencoder](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9506368): enhance the robustness against AE pre-processing | li2021persistent |  Li et al, *ICIP 2021*

## Overwriting Attack
energy perspective
<!-- 1. [Watermarking Neural Network with Compensation Mechanism](https://www.jianguoyun.com/p/DV0-NowQ0J2UCRjey-0D): using spread spectrum and a noise sequence for security; 补偿机制指对没有嵌入水印的权值再进行fine-tune; measure changes with norm (energy perspective) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:xed2zy5YT5YJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD-kI4:AAGBfm0AAAAAYHP4iI6opse7jxpYkvyx4yzXtNjTcNYl&scisig=AAGBfm0AAAAAYHP4iKhXdKnITn4E9R_eO2rFPPPjZQXs&scisf=4&ct=citation&cd=-1&hl=en): feng2020watermarking | Feng et al, *International Conference on Knowledge Science, Engineering and Management* 2020

2. [DeepWatermark: Embedding Watermark into DNN Model](http://www.apsipa.org/proceedings/2020/pdfs/0001340.pdf)：using dither modulation in FC layers  fine-tune the pre-trainde model; the amount of changes in weights can be measured (energy perspective )  | [BibTex](): kuribayashi2020deepwatermark | Kuribayashi et al, *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* 2020  (only overwriting attack) -->


## Ambiguity Attack
forgery attack; protocol attack; invisible attack
1. [Combatting ambiguity attacks via selective detection of embedded watermarks](https://www.researchgate.net/profile/Nasir-Memon/publication/3455345_Combatting_Ambiguity_Attacks_via_Selective_Detection_of_Embedded_Watermarks/links/02e7e529fec5813232000000/Combatting-Ambiguity-Attacks-via-Selective-Detection-of-Embedded-Watermarks.pdf): `common attack in media watermarking` | [BibTex](): sencar2007combatting | *IEEE Transactions on Information Forensics and Security (TIFS)* 2007

<!-- 1. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:BKAV-WKeJ1AJ:scholar.google.com/&output=citation&scisdr=CgWVvEwREJLC_OQ2dGI:AAGBfm0AAAAAYGcwbGKgqKY6a88Qf5KSWhJ1cZDTLhKp&scisig=AAGBfm0AAAAAYGcwbFH6YVqAHUeAAN6Prl_2T1s73g_a&scisf=4&ct=citation&cd=-1&hl=en):fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | [PAMI:DeepIPR: Deep Neural Network Ownership Verification with Passports](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9454280): fan2021deepip | Fan et al, *NeuraIPS* 2019, 2019.9

2. [Passport-aware Normalization for Deep Model Protection](https://proceedings.neurips.cc/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf): Improvemnet of [1] | [Code](https://github.com/ZJZAC/Passport-aware-Normalization) | [BibTex](): zhang2020passport | Zhang et al, *NeuraIPS* 2020, 2020.9

3. [Secure neural network watermarking protocol against forging attack](https://www.jianguoyun.com/p/DVsuU1IQ0J2UCRic_-0D)：引入单向哈希函数，使得用于证明所有权的触发集样本必须通过连续的哈希逐个形成，并且它们的标签也按照样本的哈希值指定。 | [BibTex](): zhu2020secure | Zhu et al, *EURASIP Journal on Image and Video Processing* 2020.1

5. [Preventing Watermark Forging Attacks in a MLaaS Environment](https://hal.archives-ouvertes.fr/hal-03220414/): | [BibTex](): sofiane2021preventing | Sofiane et al. *SECRYPT 2021, 18th International Conference on Security and Cryptography* -->

## Surrogate Model Attack / Model Stealing Attack
Shall distinguishing surrogate model attack and inference attack

1. [Stealing machine learning models via prediction apis](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_tramer.pdf): protecting against an adversary with physical access to the host device of the policy is often impractical or disproportionately costly | [BibTex](): tramer2016stealing | Tramer et al, *25th USENIX* 2016

2. [Knockoff nets: Stealing functionality of black-box models]() | [BibTex](): orekondy2019knockoff | *CVPR* 2019

3. [Stealing Deep Reinforcement Learning Models for Fun and Profit](https://arxiv.org/pdf/2006.05032.pdf): first model extraction attack against Deep Reinforcement Learning (DRL), which enables an external adversary to precisely recover a black-box DRL model only from its interaction with the environment | [Bibtex](): chen2020stealing | Chen et al, 2020.6

4. [Good Artists Copy, Great Artists Steal: Model Extraction Attacks Against Image Translation Generative Adversarial Networks](https://arxiv.org/pdf/2104.12623.pdf): we show the first model extraction attack against real-world generative adversarial network (GAN) image translation models | [BibTex](): szyller2021good | Szyller et al, 2021.4

<!-- 10. [Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers](https://arxiv.org/pdf/1306.4447.pdf): This kind of information leakage can be exploited, for example, by a vendor to build more effective classifiers or to simply acquire trade secrets from a competitor’s apparatus, potentially violating its intellectual property rights. 训练数据会泄露，可以用来做模型版权溯源| [BibTex](): ateniese2015hacking | Ateniese et al, *International Journal of Security and Networks* 2015 -->

5. [High Accuracy and High Fidelity Extraction of Neural Networks](https://arxiv.org/pdf/1909.01838.pdf): distinguish between two types of model extraction-fidelity extraction and accuracy extraction | [BibTex](): jagielski2020high | Jagielski et al, *29th {USENIX} Security Symposium (S&P)* 2020

6. [Model Extraction Warning in MLaaS Paradigm](https://dl.acm.org/doi/pdf/10.1145/3274694.3274740):  | [BibTex](): kesarwani2018model | Kesarwani et al, *Proceedings of the 34th Annual Computer Security Applications Conference(ACSAC)* 2018

7. [Stealing neural networks via timing side channels](https://arxiv.org/pdf/1812.11720.pdf): Here, an adversary can extract the Neural Network parameters, infer the regularization hyperparameter, identify if a data point was part of the training data, and generate effective transferable adversarial examples to evade classifiers; this paper is exploiting the timing side channels to infer the depth of the network; using reinforcement learning to reduce the search space | [BibTex](): duddu2018stealing | Duddu et al, 2018.12



[Countermeasures]
`model watermarking methods`
1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](): zhang2020model | Zhang et al, *AAAI* 2020.2

2. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](): zhang2021deep | Zhang al, *TPAMI* 2021.3

3. [DAWN: Dynamic Adversarial Watermarking of Neural Networks](https://arxiv.org/pdf/1906.00830.pdf): dynamically changing the responses for a small subset of queries (e.g., <0.5%) from API clients | [BibTex](): szyller2019dawn | Szyller et al, 2019,6


`other methods`
1. [PRADA: Protecting Against DNN Model Stealing Attacks](https://arxiv.org/pdf/2103.04980.pdf)：detect query patterns associated with some distillation attacks | [BibTex](): juuti2019prada | Juuti al, *IEEE European Symposium on Security and Privacy (EuroS&P)* 2019

2. [Hardness of Samples Is All You Need: Protecting Deep Learning Models Using Hardness of Samples](https://arxiv.org/pdf/2106.11424.pdf): outperforms PRADA by a large margin and has significantly less computational overhead; Hardness-Oriented Detection Approach (HODA) can detect JBDA, JBRAND, and Knockoff Net attacks with a high success rate by only watching 100 samples of attack. | [BibTex](): mahdi2021hardness | Sadeghzadeh et al, 2021.6

3. [Extraction of complex DNN models: Real threat or boogeyman?](https://arxiv.org/pdf/1910.05429.pdf)：we introduce a defense based on distinguishing queries used for Knockoff nets from benign queries. | [Slide](https://asokan.org/asokan/research/ModelStealing-master.pdf) | [BibTex](): atli2020extraction | Atli et al, *International Workshop on Engineering Dependable and Secure Machine Learning Systems. Springer, Cham* 2020

4. [Protecting Decision Boundary of Machine Learning Model With Differentially Private Perturbation](https://ieeexplore.ieee.org/abstract/document/9286504)：we propose boundary differential privacy (BDP) against surrogate model attacks by obfuscating the prediction responses with noises | [BibTex](): zheng2020protecting | Zheng et al, *IEEE Transactions on Dependable and Secure Computing* 2020

5. [Prediction poisoning: Towards defenses against dnn model stealing attacks](https://arxiv.org/pdf/1906.10908v2.pdf): In this paper, we propose the first defense which actively perturbs predictions targeted at poisoning the training objective of the attacker. | [BibTex](): orekondy2019prediction | Orekondy et al, *ICLR*2020

6. [MimosaNet: An Unrobust Neural Network Preventing Model Stealing](https://arxiv.org/pdf/1907.01650.pdf): . In this paper, we propose a method for creating an equivalent version of an already trained fully connected deep neural network that can prevent network stealing: namely, it produces the same responses and classification accuracy, but it is extremely sensitive to weight changes.  | [BibTex](): szentannai2019mimosanet | Szentannai et al, 2019.7

7. [Stateful Detection of Model Extraction Attacks](https://arxiv.org/pdf/2107.05166.pdf) detection-based approach | [To do]()


# <span id="Perspective">Perspective</span> [^](#back)

## <span id="Digital-Rights-Management(DRM)">Digital Rights Management (DRM)</span> [^](#back)
1. [Survey on the Technological Aspects of Digital Rights Management](https://link.springer.com/content/pdf/10.1007%2F978-3-540-30144-8_33.pdf): Digital Rights Management (DRM) has emerged as a multidisciplinary measure to protect the copyright of content owners and to facilitate the consumption of digital content. | [BibTex](): ku2004survey | Ku et al, *International Conference on Information Security* 2004

2. [Digital rights management](http://www.medien.ifi.lmu.de/lehre/ws0607/mmn/mmn2a.pdf): slides | [BibTex](): rosenblatt2002digital | Rosenblatt et al,  *New York* 2002 

2. [SoK: Machine Learning Governance](https://arxiv.org/pdf/2109.10870.pdf): 提出了机器学习管理的概念，其中一个环节就是要找到model owner 也就是identity | Chandrasekaran et al, 2021.9

## <span id="Hardware">Hardware</span> [^](#back)
1. [Machine Learning IP Protection](https://dl.acm.org/doi/pdf/10.1145/3240765.3270589): Major players in the semiconductor industry provide mechanisms on device to protect the IP at rest and during execution from being copied, altered, reverse engineered, and abused by attackers. **参考硬件领域的保护措施（静态动态）** | [BitTex](): cammarota2018machine | Cammarota et al, *Proceedings of the International Conference on Computer-Aided Design(ICCAD)* 2018

1. [SIGNED- A Challenge-Response Based Interrogation Scheme for Simultaneous Watermarking and Trojan Detection](https://arxiv.org/pdf/2010.05209.pdf)：半导体电路的版权保护，电路通断的选择是否可以运用到神经网络？ | [BibTex](): nair2020signed | Nair et al, 2020.10

2. [Scanning the Cycle: Timing-based Authentication on PLCs](https://arxiv.org/pdf/2102.08985.pdf): a novel
technique to authenticate PLCs is proposed that aims at raising the bar against powerful attackers while being compatible with real-time systems | [BibTex](): mujeeb2021scanning | Mujeeb et al, *AsiaCCS* 2021

4. [DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks](http://cseweb.ucsd.edu/~jzhao/files/DeepAttest-isca2019.pdf): the first on-device DNN attestation method that certifies the legitimacy of the DNN program mapped to the device; device-specific fingerprint | chen2019deepattest | Chen et al, *ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)* 2019

5. [Hardware-Assisted Intellectual Property Protection of Deep Learning Models](https://eprint.iacr.org/2020/1016.pdf): nsures that only an authorized end-user who possesses a trustworthy hardware device (with the secret key embedded on-chip) is able to run intended DL applications using the published model | [BibTex](): chakraborty2020hardware | Chakraborty et al, *57th ACM/IEEE Design Automation Conference (DAC)* 2020

7. [Preventing DNN Model IP Theft via Hardware Obfuscation](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9417217): This paper presents a novel solution to defend against DL IP theft in NPUs during model distribution and deployment/execution via lightweight, keyed model obfuscation scheme. | [BibTex](): goldstein2021preventing | Goldstein et al, *IEEE Journal on Emerging and Selected Topics in Circuits and Systems* 2021

8. [Can Evil IoT Twins Be Identified? Now Yes, a Hardware  Fingerprinting Methodology](https://arxiv.org/pdf/2106.08209.pdf): In this work, we propose a novel methodology,
GNN4IP, to assess similarities between circuits and detect IP piracy.

9. [GNN4IP: Graph Neural Network for Hardware Intellectual Property Piracy Detection](https://arxiv.org/pdf/2107.09130.pdf)

threats from side-channel attacks
1. [Stealing neural networks via timing side channels](https://arxiv.org/pdf/1812.11720.pdf): Here, an adversary can extract the Neural Network parameters, infer the regularization hyperparameter, identify if a data point was part of the training data, and generate effective transferable adversarial examples to evade classifiers; this paper is exploiting the timing side channels to infer the depth of the network; using reinforcement learning to reduce the search space | [BibTex](): duddu2018stealing | Duddu et al, 2018.12

### <span id="IC-designs">IC designs</span> [^](#back)
1. [Analysis of watermarking techniques for graph coloring problem](https://drum.lib.umd.edu/bitstream/handle/1903/9032/c003.pdf?sequence=1)  | [BibTex](): qu1998analysis | Qu et al, *IEEE/ACM international conference on Computer-aided design* 1998 

2. [Intellectual property protection in vlsi designs: Theory and practice]() | [BibTex](): qu2007intellectual | Qu et al, *Springer Science & Business Media* 2007

3. [Hardware IP Watermarking and Fingerprinting](http://web.cs.ucla.edu/~miodrag/papers/Chang_SecureSystemDesign_2016.pdf) | [BibTex](): chang2016hardware | Chang et al, * Secure System Design and Trustable Computing* 2016

### IP core watermarking
1. [IP-cores watermarking scheme at behavioral level using genetic algorithms](https://reader.elsevier.com/reader/sd/pii/S0952197621002347?token=1F75074366C580A6469A1F4C61817DDA2DAF0A1CADBDF5828E5F58F078C0AA2E33002FB14A5033BE41CBC2D6B02A9181&originRegion=eu-west-1&originCreation=20210728075104) [To do]()

2. [A Study of Device Fingerprinting Methods](https://link.springer.com/chapter/10.1007/978-981-33-4968-1_55) [To do]()

## Software Watermarking
1. [Software Watermarking: Models and Dynamic Embeddings](http://users.rowan.edu/~tang/courses/ref/watermarking/collberg.pdf) | [BibTex](): collberg1999software | Collberg et al, *Proceedings of the 26th ACM SIGPLAN-SIGACT symposium on Principles of programming languages* 1999

2. [A Graph Theoretic Approach to Software](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.5287&rep=rep1&type=pdf) | [BibTex](): venkatesan2001graph | Venkatesan et al, *In International Workshop on Information Hiding (pp. 157-168). Springer, Berlin, Heidelberg* 2001


### Software Analysis
1. [How are Deep Learning Models Similar?: An Empirical Study on Clone Analysis of Deep Learning Software](https://dl.acm.org/doi/pdf/10.1145/3387904.3389254): first study how the existing clone analysis techniques perform in the deep learning software. Secrion6.2(deep learning testing anf analysis) | [BibTex](): wu2020deep | Wu et al, *Proceedings of the 28th International Conference on Program Comprehension(ICPC)* 2020

2. [LogExtractor: Extracting digital evidence from android log messages via string and taint analysis](https://www.sciencedirect.com/science/article/pii/S2666281721001013): digital evidence on mobile devices plays a more and more important role in crime investigations. | Cheng et al, *Forensic Science International: Digital Investigation 2021*

3. [Systematic Training and Testing for Machine Learning Using Combinatorial Interaction Testing](https://arxiv.org/pdf/2201.12428.pdf): ` combinatorial interaction testing` for identifying faults in software testing; set difference combinatorial coverage (SDCC)


### Graph Watermarking

## <span id="Privacy-Risk(inference-attack)">Privacy Risk (inference attack)</span> [^](#back)
1. [Hacking Smart Machines with Smarter Ones: How to Extract Meaningful Data from Machine Learning Classifiers](https://arxiv.org/pdf/1306.4447.pdf): on the `statistical information` that can be unconsciously or maliciously revealed from them by training a meta-classifier;  训练数据会泄露，可以用来做模型版权溯源 化敌为友 | [BibTex](): ateniese2015hacking | Ateniese et al, *International Journal of Security and Networks* 2015

2. [EMA: Auditing Data Removal from Trained Models](https://arxiv.org/pdf/2109.03675.pdf)；`Auditing Data`; to measure if certain data are memorized by the trained model. 作为水印或指纹的评价测试工具？ | huang2021mathsf, | Huang, Yangsibo and Li, Xiaoxiao and Li, Kai | International Conference on Medical Image Computing and Computer-Assisted Intervention, 2021.9

3. [Dataset inference: Ownership resolution in machine learning](https://openreview.net/pdf/f677fca9fd0a50d90120a4a823fcbbe889d8ca28.pdf): 给了线性分类器下，DI和MI的理论表达式， DI>>MI; 通过距离boundary的远近来进行判断，而不是0/1判断类似MI；hypothesis testing to guarantee the integrity; 不同的model stealing attack可以参考（Sup B）; but shall disclose data | [Code](https://github.com/cleverhans-lab/dataset-inference) | Maini, Pratyush and Yaghini, Mohammad and Papernot, Nicolas | maini2021dataset | *International Conference on Learning Representations (ICLR)* 2021
- 模型训练倾向于使boundary 远离训练样本，这样分的更准确（置信度大？）；test集不影响模型boundary
<br/><img src='./IP-images/220131-1.png' align='left' style=' width:300px;height:100 px'/> <br/><br/><br/><br/><br/><br/>


1. [Robust Membership Encoding: Inference Attacks and Copyright Protection for Deep Learning](https://arxiv.org/pdf/1909.12982.pdf)： first paper？ | *AsiaCCS‘20*

1. [Privacy risk in machine learning: Analyzing the connection to overfitting](https://arxiv.org/pdf/1709.01604.pdf): This paper examines the effect that overfitting and influence have on the ability of an attacker to learn information about the training data from machine learning models, either through training set membership inference or attribute inference attacks; our formal analysis also shows that overfitting is not necessary for these attacks and begins to shed light on what other factors may be in play | [BibTex](): yeom2018privacy| Yeom et al, *31st Computer Security Foundations Symposium (CSF)* 2018

2. [Membership Leakage in Label-Only Exposures](https://arxiv.org/pdf/2007.15528.pdf): we propose decision-based membership inference attacks and demonstrate that label-only exposures are also vulnerable to membership leakage | [BibTex](): li2020membership | Li et al, *CCS* 2021

3. [GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative Models](https://arxiv.org/pdf/1909.03935v3.pdf):  we present the first taxonomy of membership inference attacks, encompassing not only existing attacks but also our novel ones | [BibTex](): chen2020gan | Chen et al, *CCS* 2020

4. [MLCapsule: Guarded Offline Deployment of Machine Learning as a Service](https://arxiv.org/pdf/1808.00590.pdf):  if the user’s input is sensitive, sending it to the server is undesirable and sometimes even legally not possible. Equally, the service provider does not want to share the model by sending it to the client for protecting its intellectual property and pay-per-query business model; Beyond protecting against direct model access, we couple the  <font color=red> secure offline deployment </font> with defenses against advanced attacks on machine learning models such as model stealing, reverse engineering, and membership inference. | [BibTex](): hanzlik2018mlcapsule | *In Proceedings of ACM Conference (Conference’17). ACM* 2019

5. [Automatic Fairness Testing of Neural Classifiers through Adversarial Sampling](https://arxiv.org/pdf/2107.08176.pdf): 网络本身的偏见是一种局限性，是不是可以用来作为水印，更细致化的归纳偏置？ inductive bias | zhang2021automatic | Zhang, Peixin and Wang, Jingyi and Sun, Jun and Wang, Xinyu and Dong, Guoliang and Wang, Xingen and Dai, Ting and Dong, Jin Song | TSE，2021

6. [Against Membership Inference Attack: Pruning is All You Need](https://www.ijcai.org/proceedings/2021/0432.pdf): some protection of MIA can also be leveraged as the attack as model IP prtection | wang2020against | Wang, Yijue and Wang, Chenghong and Wang, Zigeng and Zhou, Shanglin and Liu, Hang and Bi, Jinbo and Ding, Caiwen and Rajasekaran, Sanguthevar | 2020.8


[side channel as black-box verification]
7. [CSI NN: Reverse engineering of neural network architectures through  electromagnetic side channel](): side channel | batina2019csi | Batina, Lejla and Bhasin, Shivam and Jap, Dirmanto and Picek, Stjepan | USENIX’19

2. [DeepSniffer: A DNN model extraction framework based on learning architectural hints](https://sites.cs.ucsb.edu/~sherwood/pubs/ASPLOS-20-deepsniff.pdf) | Hu et al, *Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems.(APLOS) 2020*

3. [Cache telepathy: Leveraging shared resource attacks to learn DNN architectures](https://www.usenix.org/system/files/sec20spring_yan_prepub.pdf) | Yan et al, *29th {USENIX} Security Symposium ({USENIX} Security 20)*

4. [Stealing neural networks via timing side channels](https://arxiv.org/pdf/1812.11720.pdf): Here, an adversary can extract the Neural Network parameters, infer the regularization hyperparameter, identify if a data point was part of the training data, and generate effective transferable adversarial examples to evade classifiers; this paper is exploiting the timing side channels to infer the depth of the network; using reinforcement learning to reduce the search space | [BibTex](): duddu2018stealing | Duddu et al, 2018.12


  
8. [Honest-but-Curious Nets: Sensitive Attributes of Private Inputs can be Secretly Coded into the Entropy of Classifiers’ Outputs](https://arxiv.org/pdf/2105.12049.pdf): Our work highlights a vulnerability that can be exploited by malicious machine learning service providers to attack their user’s privacy in several seemingly safe scenarios; such as encrypted inferences, computations at the edge, or private knowledge distillation. 多标签数据 | 2021.5

9. [Stealing Machine Learning Models: Attacks and Countermeasures for Generative Adversarial Networks](https://dl.acm.org/doi/pdf/10.1145/3485832.3485838) | Hu et al, *ACSAC'21*

### [data privacy]
1. [unlearnable_examples_making_personal_data_unexploitable](https://arxiv.org/pdf/2101.04898.pdf) | [BibTex](): huang2021unlearnable | Huang et al, *ICLR 2021* 

[Tips]
- `Specific or General, which is better for DNN watermarking or other techniques`
- `Similar to backdooring, what are the remaining techniques for us to leverage with positive purpose?`
- `external features may be robust to extraction attack, but prone to inference attack?`
- `white-box watermarking method is inherently fragile to model extraction attack`
- `对抗样本生成次数（距离边界的垂直距离），对抗样本的数量（平行距离），黄金样本？ 经过简单fine-tune就能回到原始模型的性能？ 怎么定义transfer-learning 攻击`
- ` According to recent works about the unfairness of ML applications, the inherent bias existing in the training data will cause the ML models to output unfair results with respect to some records [52], [53]. By identifying the prediction unfairness, we may verify the IP of a suspect model’s training data with a smaller cost.`
