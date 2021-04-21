Paper & Code 
========================
**Works for deep model intellectual property (IP) protection.**


# Contents 
+ [Survey](#Survey)
+ [Mechanism](#Mechanism)
+ [Classic&nbsp;Methods](#Classic&nbsp;Methods)
    + [White-box](#White-box)
        + [Embedding-based(extracting-driven)](#Embedding-based(extracting-driven))
        + [Improment](#Improvement)
    + [Black-box&nbsp;with&nbsp;Queries](#Black-box&nbsp;with&nbsp;Queries)
        + [Triggers](#Triggers)
            + [Enhancement](#Enhancement)
        + [Adversatial&nbsp;Examples](#Adversatial&nbsp;Examples)
    + [Black-box&nbsp;w/o&nbsp;Queries;Data-based](#Black-box&nbsp;w/o&nbsp;Queries;Data-based)
+ [Security](#Security)
    + [Overwriting&nbsp;Attack](#Overwriting&nbsp;Attack)
    + [Ambiguty&nbsp;Attack](#Ambiguty&nbsp;Attack)
    + [Collusion&nbsp;Attack](#Collusion&nbsp;Attack)
    + [Surrogate&nbsp;Model&nbsp;Attack&nbsp;/&nbsp;Model&nbsp;Stealing&nbsp;Attack](#Surrogate&nbsp;Model&nbsp;Attack&nbsp;/&nbsp;Model&nbsp;Stealing&nbsp;Attack)
    + [Removal&nbsp;Attack](#Removal&nbsp;Attack)
    + [Evasion&nbsp;Attack](#Evasion&nbsp;Attack)

+ [Applications](#Applications)
+ [Model&nbsp;Authentication](#Model&nbsp;Authentication)
    + [Integrity&nbsp;Authentication](#Integrity&nbsp;Authentication)
    + [Usage&nbsp;Authentication](#Usage&nbsp;Authentication)
+ [Model&nbsp;Encryption](#Model&nbsp;Encryption)
+ [Evaluation](#Evaluation)
+ [Motivation](#Motivation)


# Survey 

1. [A Survey on Model Watermarking Neural Networks](https://arxiv.org/pdf/2009.12153.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:Q_T8Vs8S7NcJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EJrulZOIFq0:AAGBfm0AAAAAYGiNDq0Faj45gcWm6bI3BZjpmxj9-9zq&scisig=AAGBfm0AAAAAYGiNDrWHl-6F-MiZQ8Dtjobj-Z8ucamc&scisf=4&ct=citation&cd=-1&hl=en): boenisch2020survey | Franziska Boenisch, 2020.9

2. [DNN Intellectual Property Protection: Taxonomy, Methods, Attack Resistance, and Evaluations](https://arxiv.org/pdf/2011.13564.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:80FeKvJfLoYJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EOiKsJSOOVU:AAGBfm0AAAAAYGiLIVWJR2wtWt-MyMwWP0Pbz7lH5fGu&scisig=AAGBfm0AAAAAYGiLIZV_wIwpymdYQy6wQsiPvLSjTf1n&scisf=4&ct=citation&cd=-1&hl=en): xue2020dnn | Xue et al, 2020.11

3. [A survey of deep neural network watermarking techniques](https://arxiv.org/pdf/2103.09274.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:zaIcaXKRpAwJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EJrulZOIaoU:AAGBfm0AAAAAYGiNcoVfvQjAQ5qtiv3zuyiYNBlQtZed&scisig=AAGBfm0AAAAAYGiNcnLbdfL56osMvII4kcRsUhEGc6gu&scisf=4&ct=citation&cd=-1&hl=en): li2021survey | Li et al, 2021.3

4. [Protecting artificial intelligence IPs: a survey of watermarking and fingerprinting for machine learning](https://ietresearch.onlinelibrary.wiley.com/doi/epdf/10.1049/cit2.12029): The majority of previous works are focused on watermarking, while more advanced methods such as fingerprinting anf attestation are promising but not yet explored in depth. This study has been concluded by discussing possible researh directions in the area. | [BibTex](): regazzoni2021protecting | Regazzoni et al, *CAAI Transactions on Intelligence Technology* 2021


# Mechanism
1. [Machine Learning Models that Remember Too Much](https://arxiv.org/pdf/1709.07886.pdf)：redundancy: embedding secret information into network parameters | [BibTex](): song2017machine  | Song et al, *Proceedings of the 2017 ACM SIGSAC Conference on computer and communications security* 2017

1. [Understanding deep learning requires rethinking generalization](https://arxiv.org/pdf/1611.03530.pdf?from=timeline&isappinstalled=0)：overfitting: The capability
of neural networks to “memorize” random noise | [BibTex](): zhang2016understanding | Zhang et al, 2016.11

3. [Stealing machine learning models via prediction apis](https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_tramer.pdf): protecting against an adversary with physical access to the host device of the policy is often impractical or disproportionately costly | [BibTex](): tramer2016stealing | Tramer et al, *25th USENIX* 2016

4. [Knockoff nets: Stealing functionality of black-box models]() | [BibTex](): orekondy2019knockoff | *CVPR* 2019

# Classic&nbsp;Methods
## White-box
### Embedding-based(extracting-driven) 
### Classic

1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082)：第一篇模型水印工作 | [Code](https://github.com/yu4u/dnn-watermark) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:UU2mQ9z-ZvgJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EJrulZOIru8:AAGBfm0AAAAAYGiNtu9S2kCupEfs3KQHz8WTFWGBZZY6&scisig=AAGBfm0AAAAAYGiNthdhBu-s0qWdmukpr6j0pFnKnZ7e&scisf=4&ct=citation&cd=-1&hl=en): uchida2017embedding | Uchia et al, *ICMR* 2017.1

2. [Digital Watermarking for Deep Neural Networks](https://arxiv.org/pdf/1802.02601.pdf)：Extension of [1] | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:vmWYEIokp0wJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EJrulZOL_nM:AAGBfm0AAAAAYGiO5nPSKZUBqialwhY9KcN5ci-bHfvY&scisig=AAGBfm0AAAAAYGiO5lQR2pB3gxvKT5mGKJ3RWmfTAqK3&scisf=4&ct=citation&cd=-1&hl=en): nagai2018digital | Nagai et al, 2018.2


### Cover Types

1. [DeepSigns: An End-to-End Watermarking Framework for Protecting the Ownership of Deep Neural Networks](http://www.aceslab.org/sites/default/files/deepsigns.pdf)：using activation map as cover | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:6Y77twqBDEQJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1ggQI:AAGBfm0AAAAAYG5mmQJvx7qBEQT-gLijQP39bYy4riZr&scisig=AAGBfm0AAAAAYG5mmSpS1eqz2rAs9IzEW85Tt05vePj_&scisf=4&ct=citation&cd=-1&hl=en): rouhani2019deepsigns | Rouhani et al, *ASPLOS* 2019

4. [Don’t Forget To Sign The Gradients! ](https://proceedings.mlsys.org/paper/2021/file/45fbc6d3e05ebd93369ce542e8f2322d-Paper.pdf)：using gradients of some selected neurons as cover, and achieve black-box verification by zeroth-order gradient estimation; **introduce some adversarial watermark attacks**  | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:6Y77twqBDEQJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1ggQI:AAGBfm0AAAAAYG5mmQJvx7qBEQT-gLijQP39bYy4riZr&scisig=AAGBfm0AAAAAYG5mmSpS1eqz2rAs9IzEW85Tt05vePj_&scisf=4&ct=citation&cd=-1&hl=en): aramoon2021don | Aramoon et al, *Proceedings of Machine Learning and Systems* 2021


12. [ When NAS Meets Watermarking: Ownership Verification of DNN Models via Cache Side Channels ](https://arxiv.org/pdf/2102.03523.pdf)：using architecture as watermark | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:vP3iGmOxK_wJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCo2eI:AAGBfm0AAAAAYHOuweKzzDuUHnNorUcpJ4vk3Y6emJwc&scisig=AAGBfm0AAAAAYHOuwVbxoNSwS5Ozc6X2krHowojwjgQ7&scisf=4&ct=citation&cd=-1&hl=en): lou2021when | Lou et al, 2021.2

### Improvement on Stealthiness

1. [Attacks on digital watermarks for deep neural networks](https://scholar.harvard.edu/files/tianhaowang/files/icassp.pdf)：weights variance or weights standard deviation, will increase noticeably and systematically during the process of watermark embedding algorithm by Uchida et al; using L2 regulatization to achieve stealthiness; w tend to mean=0, var=1 | [BibTex](): wang2019attacks | Wang et al, *ICASSP* 2019


11. [RIGA Covert and Robust White-Box Watermarking of Deep Neural Networks](https://arxiv.org/pdf/1910.14268.pdf)：improvement of [1] in stealthiness, constrain the weights distribution with advesarial training | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:cS8_mQMHQYYJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1otdg:AAGBfm0AAAAAYG5urdhpKjalH8nqYP7SNLkBjTymKK4_&scisig=AAGBfm0AAAAAYG5urdf92Jw__aZTcaPD-r4tpJurQpsx&scisf=4&ct=citation&cd=-1&hl=en): wang2019riga | Wang et al, *WWW* 2021

12. [Adam and the ants: On the influence of the optimization algorithm on the detectability of dnn watermarks](https://www.mdpi.com/1099-4300/22/12/1379/pdf)：improvement of [1] in stealthiness, adoption of the Adam optimiser introduces a dramatic variation on
the histogram distribution of the weights after watermarking, constrain Adam optimiser is
run on the projected weights using the projected gradients | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:cS8_mQMHQYYJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1otdg:AAGBfm0AAAAAYG5urdhpKjalH8nqYP7SNLkBjTymKK4_&scisig=AAGBfm0AAAAAYG5urdf92Jw__aZTcaPD-r4tpJurQpsx&scisf=4&ct=citation&cd=-1&hl=en): cortinas2020adam | Cortiñas-Lorenzo et al, *Entropy* 2020

### Learning from tradtional watermarking

1. [DeepWatermark: Embedding Watermark into DNN Model](http://www.apsipa.org/proceedings/2020/pdfs/0001340.pdf)：using dither modulation in FC layers  fine-tune the pre-trainde model; the amount of changes in weights can be measured (energy perspective )  | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:lOUUCIYAZlQJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD2tFU:AAGBfm0AAAAAYHPwrFVFWPMAEfxbgngLdZrnTjmviyTG&scisig=AAGBfm0AAAAAYHPwrN0IxuIMpGcX6opjL56pCnY0EFHK&scisf=4&ct=citation&cd=-1&hl=en): kuribayashi2020deepwatermark | Kuribayashi et al, *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* 2020  (only overwriting attack)


7. [ Spread-Transform Dither Modulation Watermarking of Deep Neural Network ](https://arxiv.org/pdf/2012.14171.pdf)：changing the activation method of [1], whcih increase the payload (capacity), couping the spread spectrum and dither modulation | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VX0Iu7_rx18J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCpoTs:AAGBfm0AAAAAYHOvuTur8wBmFO7SEImIcTc1vaqgaWZO&scisig=AAGBfm0AAAAAYHOvuUsZDhKHFcZW6NYdvN-oeaeRyD_w&scisf=4&ct=citation&cd=-1&hl=en): li2020spread | Li et al, 2020.12


5. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:and7Xl29vpgJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3B9Rc:AAGBfm0AAAAAYG7H7Rd27DlL3WE79fbcPDcHgVpDQKuZ&scisig=AAGBfm0AAAAAYG7H7bejDswez8m_t6Y9zhsbsPAMtZ2c&scisf=4&ct=citation&cd=-1&hl=en): chen2019deepmarks | Chen et al, *ICMR* 2019

### Different trainig strategies

1. [Watermarking Neural Network with Compensation Mechanism](https://www.jianguoyun.com/p/DV0-NowQ0J2UCRjey-0D): using spread spectrum and a noise sequence for security; 补偿机制指对没有嵌入水印的权值再进行fine-tune; measure changes with norm (energy perspective) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:xed2zy5YT5YJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD-kI4:AAGBfm0AAAAAYHP4iI6opse7jxpYkvyx4yzXtNjTcNYl&scisig=AAGBfm0AAAAAYHP4iKhXdKnITn4E9R_eO2rFPPPjZQXs&scisf=4&ct=citation&cd=-1&hl=en): feng2020watermarking | Feng et al, *International Conference on Knowledge Science, Engineering and Management* 2020


2. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:BKAV-WKeJ1AJ:scholar.google.com/&output=citation&scisdr=CgWVvEwREJLC_OQ2dGI:AAGBfm0AAAAAYGcwbGKgqKY6a88Qf5KSWhJ1cZDTLhKp&scisig=AAGBfm0AAAAAYGcwbFH6YVqAHUeAAN6Prl_2T1s73g_a&scisf=4&ct=citation&cd=-1&hl=en):fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | Fan et al, *NeuraIPS* 2019, 2019.9

3. [Delving in the loss landscape to embed robust watermarks into neural networks](https://www.jianguoyun.com/p/DfA64QMQ0J2UCRjlw-0D)：using partial weights to embed watermark information and keep it untrainable | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VX0Iu7_rx18J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCpoTs:AAGBfm0AAAAAYHOvuTur8wBmFO7SEImIcTc1vaqgaWZO&scisig=AAGBfm0AAAAAYHOvuUsZDhKHFcZW6NYdvN-oeaeRyD_w&scisf=4&ct=citation&cd=-1&hl=en): tartaglione2020delving | Tartaglione et al, *ICPR* 2020

4. [HufuNet: Embedding the Left Piece as Watermark and Keeping the Right Piece for Ownership Verification in Deep Neural Networks](https://arxiv.org/pdf/2103.13628.pdf)：Hufu(虎符), left piece for embedding watermark, right piece as local secret; introduce some attack: model pruning, model fine-tuning, kernels cutoff/supplement and crafting adversarial samples, structure adjustment or parameter adjustment; Table12 shows the number of backoors have influence on the performance; cosine similarity is robust even weights or sturctures are adjusted, can restore the original structures or parameters | [Code](https://github.com/HufuNet/HufuNet) | [BibTex](): lv2021hufunet | Lv et al, 2021.3

5. [Watermarking in Deep Neural Networks via Error Back-propagation](https://www.ingentaconnect.com/contentone/ist/ei/2020/00002020/00000004/art00003?crawler=true&mimetype=application/pdf)：using an independent network (weights selected from the main network) to embed and extract watermark; provide some suggestions for watermarking; introduce model isomorphism attack | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:X8GMvXP87u4J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1xu3Y:AAGBfm0AAAAAYG53o3ZipYJL8UKjVx8Q2MnKPBS9qrmo&scisig=AAGBfm0AAAAAYG53o6WiowiFbXNm-pzzlMJ-xVqvw_3j&scisf=4&ct=citation&cd=-1&hl=en): wang2020watermarking | Wang et al, *Electronic Imaging* 2020.4






## Black-box&nbsp;with&nbsp;Queries
### Triggers

1. [Turning Your Weakness Into a Strength: Watermarking Deep Neural Networks by Backdooring](https://www.usenix.org/system/files/conference/usenixsecurity18/sec18-adi.pdf)：thefirst backdoor-based， abstract image; 补充材料： From Private to Public Verifiability, Zero-Knowledge Arguments. | [Code](https://github.com/adiyoss/WatermarkNN) | [BibTex](): adi2018turning | Adi et al, *27th {USENIX} Security Symposium* 2018

2. []()： | [BibTex]():  |  et al, 2018.2


1. []()： | [BibTex]():  |  et al, 2018.2
1. []()： | [BibTex]():  |  et al, 2018.2
1. []()： | [BibTex]():  |  et al, 2018.2
1. []()： | [BibTex]():  |  et al, 2018.2
1. []()： | [BibTex]():  |  et al, 2018.2




1. [‘‘Identity Bracelets’’ for Deep Neural Networks](https://arxiv.org/pdf/1911.08053.pdf)：using MNIST (unrelated to original dataset) as trigger set | [BibTex](): xu2020identity  | [Initial Version: A novel method for identifying the deep neural network model with the Serial Number](https://arxiv.org/pdf/1911.08053.pdf) | Xu et al, *IEEE Access* 2020.8


#### Enhancement

2. [Robust Watermarking of Neural Network with Exponential Weighting](https://arxiv.org/pdf/1901.06151.pdf)：using original training data with wrong label as triggers; increase the weight value exponentially so that model modification cannot change the prediction behavior of samples (including key samples) before and after model modification; introduce query modification attack, namely, pre-processing to query | [BibTex](): namba2019robust |  et al, *Proceedings of the 2019 ACM Asia Conference on Computer and Communications Security (AisaCCS)* 2019

3. [Secure neural network watermarking protocol against forging attack](https://www.jianguoyun.com/p/DVsuU1IQ0J2UCRic_-0D)：引入单向哈希函数，使得用于证明所有权的触发集样本必须通过连续的哈希逐个形成，并且它们的标签也按照样本的哈希值指定。 | [BibTex](): zhu2020secure | Zhu et al, *EURASIP Journal on Image and Video Processing* 2020.1


4. [Effectiveness of Distillation Attack and Countermeasure on DNN watermarking](https://arxiv.org/pdf/1906.06046.pdf)：Distilling the model's knowledge to another model of smaller size from scratch destroys all the watermarks because it has a fresh model architecture and training process; countermeasure: embedding the watermark into NN in an indiret way rather than directly overfitting the model on watermark, specifically, let the target model learn the general patterns of the trigger not regarding it as noise. | [BibTex](): yang2019effectiveness  | Yang et al, 2019.6

5. [Entangled Watermarks as a Defense against Model Extraction ](https://arxiv.org/pdf/2002.12200.pdf)：forcing the model to learn features which are jointly used to analyse both the normal and the triggers; using soft nearest neighbor loss (SNNL) to measure entanglement over labeled data | [BibTex](): jia2020entangled |  et al, *30th USENIX* 2020


6. [Protecting the Intellectual Properties of Deep Neural Networks with an Additional Class and Steganographic Images](https://arxiv.org/pdf/2104.09203.pdf):  use a set of watermark key samples to embed an additional class into the DNN; adopt the least significant bit (LSB) image steganography to embed users’ fingerprints for authentication and management of fingerprints | [BibTex](): sun2021protecting | Sun et al, 2021.4


7. [Piracy Resistant Watermarks for Deep Neural Networks](https://arxiv.org/pdf/1910.01226.pdf): out-of-bound values; null embedding; wonder filter | [Video](https://www.youtube.com/watch?v=yb0_GwRvF4k&ab_channel=stanfordonline) | [BibTex](): li2019piracy | Li et al, 2019.10 | [Initial version](http://web.stanford.edu/class/ee380/Abstracts/191030-paper.pdf): Persistent and Unforgeable Watermarks for Deep Neural Networks | [BibTex](): li2019persistent | Li et al, 2019.10

### Adversatial&nbsp;Examples



## Black-box&nbsp;w/o&nbsp;Queries;Data-based



# Security

## Overwriting&nbsp;Attack
1. [Watermarking Neural Network with Compensation Mechanism](https://www.jianguoyun.com/p/DV0-NowQ0J2UCRjey-0D): using spread spectrum and a noise sequence for security; 补偿机制指对没有嵌入水印的权值再进行fine-tune; measure changes with norm (energy perspective) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:xed2zy5YT5YJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD-kI4:AAGBfm0AAAAAYHP4iI6opse7jxpYkvyx4yzXtNjTcNYl&scisig=AAGBfm0AAAAAYHP4iKhXdKnITn4E9R_eO2rFPPPjZQXs&scisf=4&ct=citation&cd=-1&hl=en): feng2020watermarking | Feng et al, *International Conference on Knowledge Science, Engineering and Management* 2020


2. [DeepWatermark: Embedding Watermark into DNN Model](http://www.apsipa.org/proceedings/2020/pdfs/0001340.pdf)：using dither modulation in FC layers  fine-tune the pre-trainde model; the amount of changes in weights can be measured (energy perspective )  | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:lOUUCIYAZlQJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD2tFU:AAGBfm0AAAAAYHPwrFVFWPMAEfxbgngLdZrnTjmviyTG&scisig=AAGBfm0AAAAAYHPwrN0IxuIMpGcX6opjL56pCnY0EFHK&scisf=4&ct=citation&cd=-1&hl=en): kuribayashi2020deepwatermark | Kuribayashi et al, *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* 2020  (only overwriting attack)


## Ambiguty&nbsp;Attack
forgery attack; protocol attack; invisible attack

1. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:BKAV-WKeJ1AJ:scholar.google.com/&output=citation&scisdr=CgWVvEwREJLC_OQ2dGI:AAGBfm0AAAAAYGcwbGKgqKY6a88Qf5KSWhJ1cZDTLhKp&scisig=AAGBfm0AAAAAYGcwbFH6YVqAHUeAAN6Prl_2T1s73g_a&scisf=4&ct=citation&cd=-1&hl=en):fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | Fan et al, *NeuraIPS* 2019, 2019.9

2. [Passport-aware Normalization for Deep Model Protection](https://proceedings.neurips.cc/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf): Improvemnet of [1] | [Code](https://github.com/ZJZAC/Passport-aware-Normalization) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:5QgO8-Ei59MJ:scholar.google.com/&output=citation&scisdr=CgWVvEwREJLC_OQ0VCc:AAGBfm0AAAAAYGcyTCc8vZsk-pFuOhQTxIcQCkrbyaKh&scisig=AAGBfm0AAAAAYGcyTD4o2PSHMinhgmKjFreZiaaIMuHC&scisf=4&ct=citation&cd=-1&hl=zh-CN) | Zhang et al, *NeuraIPS* 2020, 2020.9


1. [Secure neural network watermarking protocol against forging attack](https://www.jianguoyun.com/p/DVsuU1IQ0J2UCRic_-0D)：引入单向哈希函数，使得用于证明所有权的触发集样本必须通过连续的哈希逐个形成，并且它们的标签也按照样本的哈希值指定。 | [BibTex](): zhu2020secure | Zhu et al, *EURASIP Journal on Image and Video Processing* 2020.1


## Collusion&nbsp;Attack

1. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:and7Xl29vpgJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3B9Rc:AAGBfm0AAAAAYG7H7Rd27DlL3WE79fbcPDcHgVpDQKuZ&scisig=AAGBfm0AAAAAYG7H7bejDswez8m_t6Y9zhsbsPAMtZ2c&scisf=4&ct=citation&cd=-1&hl=en): chen2019deepmarks | Chen et al, *ICMR* 2019


## Surrogate&nbsp;Model&nbsp;Attack&nbsp;/&nbsp;Model&nbsp;Stealing&nbsp;Attack

1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VVOq5e67uCEJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1YmLg:AAGBfm0AAAAAYG5egLj8-8TdhW-OrFR5PtcTAgXDBsUU&scisig=AAGBfm0AAAAAYG5egJ2W418j7bkygIvLDr7B5IUgFq1r&scisf=4&ct=citation&cd=-1&hl=en): zhang2020model | Zhang et al, *AAAI* 2020.2

2. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:_r5iMZdEAsAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1bYNo:AAGBfm0AAAAAYG5deNoV3ooCjF9U9Rk5ckk8f8_ZS956&scisig=AAGBfm0AAAAAYG5deM2L5_2I2AvaWBetKSrL4CFclBGM&scisf=4&ct=citation&cd=-1&hl=en):    zhang2021deep | Zhang al, *TPAMI* 2021.3

3. [PRADA: Protecting Against DNN Model Stealing Attacks](https://arxiv.org/pdf/2103.04980.pdf)：detect query patterns associated with some distillation attacks | [BibTex]():    juuti2019prada | Juuti al, *IEEE European Symposium on Security and Privacy (EuroS&P)* 2019


4. [Extraction of complex DNN models: Real threat or boogeyman?](https://arxiv.org/pdf/1910.05429.pdf)：we introduce a defense based on distinguishing queries used for Knockoff nets from benign queries. | [Slide](https://asokan.org/asokan/research/ModelStealing-master.pdf) | [BibTex](): atli2020extraction | Atli et al, *International Workshop on Engineering Dependable and Secure Machine Learning Systems. Springer, Cham* 2020

5. [DAWN: Dynamic Adversarial Watermarking of Neural Networks](https://arxiv.org/pdf/1906.00830.pdf): dynamically changing the responses for a small subset of queries (e.g., <0.5%) from API clients | [BibTex](): szyller2019dawn | Szyller et al, 2019,6

## Removal&nbsp;Attack
1. [Effectiveness of Distillation Attack and Countermeasure on DNN watermarking](https://arxiv.org/pdf/1906.06046.pdf)：Distilling the model's knowledge to another model of smaller size from scratch destroys all the watermarks because it has a fresh model architecture and training process; countermeasure: embedding the watermark into NN in an indiret way rather than directly overfitting the model on watermark, specifically, let the target model learn the general patterns of the trigger not regarding it as noise. | [BibTex](): yang2019effectiveness  | Yang et al, 2019.6


2. [Attacks on digital watermarks for deep neural networks](https://scholar.harvard.edu/files/tianhaowang/files/icassp.pdf)：weights variance or weights standard deviation, will increase noticeably and systematically during the process of watermark embedding algorithm by Uchida et al; using L2 regulatization to achieve stealthiness; w tend to mean=0, var=1 | [BibTex](): wang2019attacks | Wang et al, *ICASSP* 2019

3. [On the Robustness of the Backdoor-based Watermarking in Deep Neural Networks](https://arxiv.org/pdf/1906.07745.pdf): white-box: just surrogate model attack with limited data; black-box: L2 regularization to prevent over-fitting to backdoor noise and compensate with fine-tuning; property inference attack: detect whether the backdoor-based watermark is embedded in the model | [BibTex](): shafieinejad2019robustness | Shafieinejad et al, 2019.6


4. [Leveraging unlabeled data for watermark removal of deep neural networks](https://ruoxijia.info/wp-content/uploads/2020/03/watermark_removal_icml19_workshop.pdf)：carefully-designed fine-tuning method; Leveraging auxiliary unlabeled data significantly decreases the amount of labeled training data needed for effective watermark removal, even if the unlabeled data samples are not drawn from the same distribution as the benign data for model evaluation | [BibTex](): chen2019leveraging | Chen et al, *ICML workshop on Security and Privacy of Machine Learning* 2019

5. [REFIT: A Unified Watermark Removal Framework For Deep Learning Systems With Limited Data](https://arxiv.org/pdf/1911.07205.pdf)：) an adaption of the elastic weight consolidation (EWC) algorithm, which is originally proposed for mitigating the catastrophic forgetting phenomenon;  unlabeled data augmentation (AU), where we leverage auxiliary unlabeled data from other sources | [Code](https://github.com/sunblaze-ucb/REFIT) | [BibTex](): chen2019refit | Chen et al, *ASIA CCS* 2021

6. [Removing Backdoor-Based Watermarks in Neural Networks with Limited Data](https://arxiv.org/pdf/2008.00407.pdf)：we benchmark the robustness of watermarking; propose "WILD" (data augmentation and alignment of deature distribution) with the limited access to training data| [BibTex](): liu2020removing | Liu et al, *ICASSP* 2019

7. [The Hidden Vulnerability of Watermarking for Deep Neural Networks](https://arxiv.org/pdf/2009.08697.pdf): First, we propose a novel preprocessing function, which embeds imperceptible patterns and performs spatial-level transformations over the input. Then, conduct fine-tuning strategy using unlabelled and out-ofdistribution samples. | [BibTex](): guo2020hidden | Guo et al, 2020.9 | PST is analogical to [Backdoor attack in the physical world](https://arxiv.org/pdf/2104.02361.pdf) | [BibTex](): li2021backdoor | Li et al, *ICLR 2021 Workshop on Robust and Reliable Machine Learning in the Real World*

8. [Re-markable: Stealing Watermarked Neural Networks Through Synthesis](https://www.jianguoyun.com/p/Da714ncQ0J2UCRiBwe8D)：using DCGAN to synthesize own training data, and using transfer learning to execute removal;analyze the failure of evasion attack, e.g., [Hitaj](https://www.researchgate.net/profile/Dorjan-Hitaj/publication/334698259_Evasion_Attacks_Against_Watermarking_Techniques_found_in_MLaaS_Systems/links/5dd6a6e692851c1feda559db/Evasion-Attacks-Against-Watermarking-Techniques-found-in-MLaaS-Systems.pdf) ; introduce the MLaaS | [BibTex](): chattopadhyay2020re | Chattopadhyay et al, *International Conference on Security, Privacy, and Applied Cryptography Engineering* 2020






# Evasion&nbsp;Attack

1. [Evasion Attacks Against Watermarking Techniques found in MLaaS Systems](https://www.researchgate.net/profile/Dorjan-Hitaj/publication/334698259_Evasion_Attacks_Against_Watermarking_Techniques_found_in_MLaaS_Systems/links/5dd6a6e692851c1feda559db/Evasion-Attacks-Against-Watermarking-Techniques-found-in-MLaaS-Systems.pdf)： | [BibTex](): hitaj2019evasion | Hitaj et al, *Sixth International Conference on Software Defined Systems (SDS)* 2019


# Applications

## Image Processing
1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VVOq5e67uCEJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1YmLg:AAGBfm0AAAAAYG5egLj8-8TdhW-OrFR5PtcTAgXDBsUU&scisig=AAGBfm0AAAAAYG5egJ2W418j7bkygIvLDr7B5IUgFq1r&scisf=4&ct=citation&cd=-1&hl=en): zhang2020model | Zhang et al, *AAAI* 2020.2


2. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:_r5iMZdEAsAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1bYNo:AAGBfm0AAAAAYG5deNoV3ooCjF9U9Rk5ckk8f8_ZS956&scisig=AAGBfm0AAAAAYG5deM2L5_2I2AvaWBetKSrL4CFclBGM&scisf=4&ct=citation&cd=-1&hl=en):    zhang2021deep | Zhang al, *TPAMI* 2021.3

3. [Watermarking Deep Neural Networks in Image Processing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9093125)：Image Peocessing, similar to [1] but exclude surrogate model attack | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:EsQcYz3vGkcJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwuwVdtk:AAGBfm0AAAAAYG8TbtkJQ70EfBd6_y4SUgSsCqJiBYKM&scisig=AAGBfm0AAAAAYG8TbkOlrwiGNIYMKOVYBzGpP7VC11zM&scisf=4&ct=citation&cd=-1&hl=en): quan2020watermarking | Quan et al, *TNNLS* 2020

3. [Watermarking Neural Networks with Watermarked Images ](https://www.jianguoyun.com/p/DWcYeY8Q0J2UCRiaue4D))：Image Peocessing, similar to [1] but exclude surrogate model attack | [BibTex](): wu2020watermarking | Wu et al, *TCSVT* 2020


## Image Caption 
1. [Protect, Show, Attend and Tell: Empower Image Captioning Model with Ownership Protection](https://arxiv.org/pdf/2008.11009.pdf)：Image Caption | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:Hq9e_KZON_EJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3PMvw:AAGBfm0AAAAAYG7JKvyqTMhTSiim967PfzvghJH_-Afl&scisig=AAGBfm0AAAAAYG7JKpGZo5fN_dho1v9lJBI2VxMu3iAH&scisf=4&ct=citation&cd=-1&hl=en): lim2020protect  | Lim et al, 2020.8
(surrogate model attck)

## Automatic Speech Recognition (ASR)
1. [SpecMark: A Spectral Watermarking Framework for IP Protection of Speech Recognition Systems](https://indico2.conference4me.psnc.pl/event/35/contributions/3413/attachments/489/514/Wed-1-8-8.pdf): Automatic Speech Recognition (ASR) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:1mZFIe2pNnAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3Pais:AAGBfm0AAAAAYG7JcivwfswRKTpDRKVkYNWU4P_fbXQ3&scisig=AAGBfm0AAAAAYG7Jcpms1fjVvGSPVAa8en4_OwmscaUY&scisf=4&ct=citation&cd=-1&hl=en): chen2020specmark | Chen et al, *Interspeech* 2020

## NLP
1. [Watermarking Neural Language Models based on Backdooring](https://github.com/TIANHAO-WANG/nlm-watermark/blob/master/nlpwatermark.pdf): NLP | Fu et al, 2020.12

## GNN
1. [Watermarking Graph Neural Networks by Random Graphs](https://arxiv.org/pdf/2011.00512.pdf): Graph Neural Networks (GNN) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:JufA1FwKhYAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwuwS9bg:AAGBfm0AAAAAYG8U7biP-vr3I4mzYcQD9Ym4MEdwLjlL&scisig=AAGBfm0AAAAAYG8U7TJdjXkL2ClDHPdjSxMsnJ9CjcBY&scisf=4&ct=citation&cd=-1&hl=en): zhao2020watermarking | Zhao et al, *Interspeech* 2020

## Federated Learning
1. [WAFFLE: Watermarking in Federated Learning](https://arxiv.org/pdf/2011.00512.pdf): WAFFLE leverages capabilities of the aggregator to embed a backdoor-based watermark by re-training the global model with the watermark during each aggregation round. | [BibTex](): atli2020waffle | Atli et al, 2020.8

2. [Watermarking Federated Deep Neural Network Models](https://aaltodoc.aalto.fi/bitstream/handle/123456789/43561/master_Xia_Yuxi_2020.pdf?sequence=1): for degree of master, advisor: Buse Atli | [BibTex](): xia2020watermarking | Xia et al, 2020



## Deep Reinforcement Learning
1. [Sequential Triggers for Watermarking of Deep Reinforcement Learning Policies
](https://arxiv.org/pdf/1906.01126.pdf): experimental evaluation of watermarking a DQN policy trained in the Cartpole environment | [BibTex](): behzadan2019sequential | Behzadan et al, 2019,6


2. [Temporal Watermarks for Deep Reinforcement Learning Models](https://personal.ntu.edu.sg/tianwei.zhang/paper/aamas2021.pdf): Deep Reinforcement Learning (DRL) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:pafSRYDd6L8J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCqb8c:AAGBfm0AAAAAYHOsd8cEKOGOFCslTLOkJ-G7iKF_eCee&scisig=AAGBfm0AAAAAYHOsd6tpE4fU7r41NEcQsfHCyeNPpHaJ&scisf=4&ct=citation&cd=-1&hl=en): chen2021temporal | Chen et al, *International Conference on Autonomous Agents and Multiagent Systems* 2021




## Generative Models
1. [Black-Box Watermarking for Generative Adversarial Networks](https://arxiv.org/pdf/2007.08457.pdf): We first embed artificial fingerprints into training data, then validate a surprising discovery on the transferability of such fingerprints from training data to generative models, which in turn appears in the generated deepfakes | [Empirical Study](https://www-inst.eecs.berkeley.edu/~cs194-26/fa20/upload/files/projFinalProposed/cs194-26-aek/CS294_26_Final_Project_Write_Up.pdf) | [BibTex](): yu2020artificial | Yu et al, 2020.7

2. [Decentralized Attribution of Generative Models](https://arxiv.org/pdf/2010.13974.pdf): Each binary classifier is parameterized by a user-specific key and distinguishes its associated model distribution from the authentic data distribution. We develop sufficient conditions of the keys that guarantee an attributability lower bound.| [Code](https://github.com/ASU-Active-Perception-Group/decentralized_attribution_of_generative_models) | [BibTex](): kim2020decentralized | Kim et al, *ICLR* 2021


# Model&nbsp;Authentication

## Integrity&nbsp;Authentication

1. [Verideep: Verifying integrity of deep neural networks through sensitive-sample fingerprinting](https://arxiv.org/pdf/1808.03277.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:n0yJ9gaj-D8J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu5ZWJk:AAGBfm0AAAAAYG1fQJm71SY1B6hD70mtxJmC4eMJoPTd&scisig=AAGBfm0AAAAAYG1fQA9lDg03bZxSS_EeCYZlws6vGDZz&scisf=4&ct=citation&cd=-1&hl=en): he2018verideep | He et al, 2018.8


3. [Reversible Watermarking in Deep Convolutional Neural Networks for Integrity Authentication](https://arxiv.org/pdf/2101.04319.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:mejXG863kv0J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu5nWns:AAGBfm0AAAAAYG1hQntz72yIxy1aeAbsu0_JJOcDNwo2&scisig=AAGBfm0AAAAAYG1hQkNzQU_uymoeimEEDB7wDmn9CvtT&scisf=4&ct=citation&cd=-1&hl=en): guan2020reversible | Guan et al, *ACM MM* 2020

4. [DeepiSign: Invisible Fragile Watermark to Protect the Integrity and Authenticity of CNN](https://arxiv.org/pdf/2101.04319.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:YS9AeQMRxtsJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu5ZoVQ:AAGBfm0AAAAAYG1fuVSHK4rIk_Zub8cG9ahS_C59dRnm&scisig=AAGBfm0AAAAAYG1fufkzApCPLjNJxAEOfzaBYseV0xFC&scisf=4&ct=citation&cd=-1&hl=en): abuadbba2021deepisign | Abuadbba et al, *SAC* 2021

## Usage&nbsp;Authentication

1. [Active DNN IP Protection: A Novel User Fingerprint Management and DNN Authorization Control Technique](https://www.jianguoyun.com/p/DdZ92TMQ0J2UCRjt4O0D): using trigger sets as copyright management | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:yiGHPi-hXbcJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvYEo2U:AAGBfm0AAAAAYHUCu2Wlc6SEUsMhD_JPaVOW0ec4m2ZY&scisig=AAGBfm0AAAAAYHUCu6eNZBLGrhyfBN5MvFU0_LfduVtC&scisf=4&ct=citation&cd=-1&hl=en): xue2020active | Xue et al, *Security and Privacy in Computing and Communications (TrustCom)* 2020

2. [ActiveGuard: An Active DNN IP Protection Technique via Adversarial Examples 2103.01527](https://www.jianguoyun.com/p/DdZ92TMQ0J2UCRjt4O0D): extension version of [2] | [BibTex](): xue2021activeguard | Xue et al, 2021.3

1. [Protecting the Intellectual Properties of Deep Neural Networks with an Additional Class and Steganographic Images](https://arxiv.org/pdf/2104.09203.pdf):  use a set of watermark key samples to embed an additional class into the DNN; adopt the least significant bit (LSB) image steganography to embed users’ fingerprints for authentication and management of fingerprints | [BibTex](): sun2021protecting | Sun et al, 2021.4


3. [DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks](http://cseweb.ucsd.edu/~jzhao/files/DeepAttest-isca2019.pdf): the first on-device DNN attestation method
that certifies the legitimacy of the DNN program mapped to the device; device-specific fingerprint | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:eZpX8EPeuCsJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvc4Wto:AAGBfm0AAAAAYHQ-Qtr3ntZVqjetGSROZcyDovSrlk7q&scisig=AAGBfm0AAAAAYHQ-QujApYeVnUD8NdhVflqQc31eSj4o&scisf=4&ct=citation&cd=-1&hl=en): chen2019deepattest | Chen et al, *ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)* 2019

4. [Protect Your Deep Neural Networks from Piracy](https://www.jianguoyun.com/p/DdrMupcQ0J2UCRjaou4D): using the key to enable correct image transformation of triggers; 对trigger进行加密 | [BibTex](): chen2018protect  | Chen et al, *IEEE International Workshop on Information Forensics and Security (WIFS)* 2018



5. [Transfer Learning-Based Model Protection With Secret Key](https://arxiv.org/pdf/2103.03525.pdf)：using the key to enable correct image transformation of triggers; 对trigger进行加密 | AprilPyone et al, 2021.3 | [Related Paper：Piracy-Resistant DNN Watermarking by Block-Wise Image Transformation with Secret Key](https://arxiv.org/pdf/2104.04241.pdf): AprilPyone et al, 2021.4 | [Related Paper: Block-wise Image Transformation with Secret Key for Adversarially Robust Defense](https://arxiv.org/pdf/2010.00801.pdf) AprilPyone et al, 2020.10 | [Related Paper: Training DNN Model with Secret Key for Model Protection](https://arxiv.org/pdf/2010.00801.pdf) AprilPyone et al, 2020.8 

6. [Piracy-Resistant DNN Watermarking by Block-Wise Image Transformation with Secret Key](https://arxiv.org/pdf/2104.04241.pdf)：using the key to enable correct image transformation of triggers; 对trigger进行加密 | [Related Paper: Block-wise Image Transformation with Secret Key for Adversarially Robust Defense](https://arxiv.org/pdf/2010.00801.pdf) | AprilPyone et al, 2021.4


7. [Deep Serial Number: Computational Watermarking for DNN Intellectual Property Protection](https://arxiv.org/pdf/2011.08960.pdf): we introduce the first attempt to embed a serial number into DNNs,  DSN is implemented in the knowledge distillation framework, During the distillation process, each customer DNN is augmented with a unique serial number, | [BibTex](): tang2020deep | Tang et al, 2020.11

# Model&nbsp;Encryption





# Evaluation
##Fidelity

1.Accuracy

##Interity

Watermark Bit Error (BER)

## Reference:
1. 数字水印技术及应用2004（孙圣和）1.7.1 评价问题
2. 数字水印技术及其应用2018（楼偶俊） 2.3 数字水印系统的性能评价
3. 数字水印技术及其应用2015（蒋天发）1.6 数字水印的性能评测方法
4. [Digital Rights Management The Problem of Expanding Ownership Rights](https://books.google.ca/books?id=IgSkAgAAQBAJ&lpg=PP1&ots=tA7ZrVoYx-&dq=Digital%20Rights%20Management%20The%20Problem%20of%20Expanding%20Ownership%20Rights&lr&pg=PA16#v=onepage&q=Digital%20Rights%20Management%20The%20Problem%20of%20Expanding%20Ownership%20Rights&f=false)


# Motivation

1. [SIGNED- A Challenge-Response Based Interrogation Scheme for Simultaneous Watermarking and Trojan Detection](https://arxiv.org/pdf/2010.05209.pdf)：半导体电路的版权保护，电路通断的选择是否可以运用到神经网络？ | [BibTex](): nair2020signed | Nair et al, 2020.10

## Software Watermarking
1. [Software Watermarking: Models and Dynamic Embeddings](http://users.rowan.edu/~tang/courses/ref/watermarking/collberg.pdf) | [BibTex](): collberg1999software | Collberg et al, *Proceedings of the 26th ACM SIGPLAN-SIGACT symposium on Principles of programming languages* 1999

1. [A Graph Theoretic Approach to Software](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.8.5287&rep=rep1&type=pdf) | [BibTex](): venkatesan2001graph | Venkatesan et al, *In International Workshop on Information Hiding (pp. 157-168). Springer, Berlin, Heidelberg* 2001

