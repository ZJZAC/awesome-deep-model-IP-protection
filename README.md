Paper & Code 
========================
**Works for deep model intellectual property (IP) protection.**


# Contents 
+ [Survey](#Survey)
+ [Classic&nbsp;Methods](#Classic&nbsp;Methods)
    + [Embedding-based](#Embedding-based)
    + [Trigger-based](#Trigger-based)
+ [Security](#Security)
+ [Applications](#Applications)
+ [Model&nbsp;Authentication](#Model&nbsp;Authentication)
+ [Model&nbsp;Encryption](#Model&nbsp;Encryption)
+ [Evaluation](#Evaluation)

# Survey 

1. [A Survey on Model Watermarking Neural Networks](https://arxiv.org/pdf/2009.12153.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:Q_T8Vs8S7NcJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EJrulZOIFq0:AAGBfm0AAAAAYGiNDq0Faj45gcWm6bI3BZjpmxj9-9zq&scisig=AAGBfm0AAAAAYGiNDrWHl-6F-MiZQ8Dtjobj-Z8ucamc&scisf=4&ct=citation&cd=-1&hl=en): boenisch2020survey | Franziska Boenisch, 2020.9

2. [DNN Intellectual Property Protection: Taxonomy, Methods, Attack Resistance, and Evaluations](https://arxiv.org/pdf/2011.13564.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:80FeKvJfLoYJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EOiKsJSOOVU:AAGBfm0AAAAAYGiLIVWJR2wtWt-MyMwWP0Pbz7lH5fGu&scisig=AAGBfm0AAAAAYGiLIZV_wIwpymdYQy6wQsiPvLSjTf1n&scisf=4&ct=citation&cd=-1&hl=en): xue2020dnn | Xue et al, 2020.11

3. [A survey of deep neural network watermarking techniques](https://arxiv.org/pdf/2103.09274.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:zaIcaXKRpAwJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EJrulZOIaoU:AAGBfm0AAAAAYGiNcoVfvQjAQ5qtiv3zuyiYNBlQtZed&scisig=AAGBfm0AAAAAYGiNcnLbdfL56osMvII4kcRsUhEGc6gu&scisf=4&ct=citation&cd=-1&hl=en): li2021survey | Li et al, 2021.3

# Classic&nbsp;Methods
## Embedding-based
1. [Embedding Watermarks into Deep Neural Networks](https://arxiv.org/pdf/1701.04082)：第一篇模型水印工作 | [Code](https://github.com/yu4u/dnn-watermark) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:UU2mQ9z-ZvgJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EJrulZOIru8:AAGBfm0AAAAAYGiNtu9S2kCupEfs3KQHz8WTFWGBZZY6&scisig=AAGBfm0AAAAAYGiNthdhBu-s0qWdmukpr6j0pFnKnZ7e&scisf=4&ct=citation&cd=-1&hl=en): uchida2017embedding | Uchia et al, 2017.1

2. [Digital Watermarking for Deep Neural Networks](https://arxiv.org/pdf/1802.02601.pdf)：Extension of [1] | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:vmWYEIokp0wJ:scholar.google.com/&output=citation&scisdr=CgXai5N8EJrulZOL_nM:AAGBfm0AAAAAYGiO5nPSKZUBqialwhY9KcN5ci-bHfvY&scisig=AAGBfm0AAAAAYGiO5lQR2pB3gxvKT5mGKJ3RWmfTAqK3&scisf=4&ct=citation&cd=-1&hl=en): nagai2018digital | Nagai et al, 2018.2


3. [DeepSigns: An End-to-End Watermarking Framework for Protecting the Ownership of Deep Neural Networks](http://www.aceslab.org/sites/default/files/deepsigns.pdf)：using activation map as cover | [code](https://github.com/Bitadr/DeepSigns) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:6Y77twqBDEQJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1ggQI:AAGBfm0AAAAAYG5mmQJvx7qBEQT-gLijQP39bYy4riZr&scisig=AAGBfm0AAAAAYG5mmSpS1eqz2rAs9IzEW85Tt05vePj_&scisf=4&ct=citation&cd=-1&hl=en): rouhani2019deepsigns | Rouhani et al, *ASPLOS* 2019

4. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:and7Xl29vpgJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3B9Rc:AAGBfm0AAAAAYG7H7Rd27DlL3WE79fbcPDcHgVpDQKuZ&scisig=AAGBfm0AAAAAYG7H7bejDswez8m_t6Y9zhsbsPAMtZ2c&scisf=4&ct=citation&cd=-1&hl=en): chen2019deepmarks | Chen et al, *ICMR* 2019


5. [Watermarking in Deep Neural Networks via Error Back-propagation](https://www.ingentaconnect.com/contentone/ist/ei/2020/00002020/00000004/art00003?crawler=true&mimetype=application/pdf)：using an independent network (weights selected from the main network) to embed and extract watermark | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:X8GMvXP87u4J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1xu3Y:AAGBfm0AAAAAYG53o3ZipYJL8UKjVx8Q2MnKPBS9qrmo&scisig=AAGBfm0AAAAAYG53o6WiowiFbXNm-pzzlMJ-xVqvw_3j&scisf=4&ct=citation&cd=-1&hl=en): wang2020watermarking | Wang et al, *Electronic Imaging* 2020.4

6. [DeepWatermark: Embedding Watermark into DNN Model](http://www.apsipa.org/proceedings/2020/pdfs/0001340.pdf)：using dither modulation in FC layers  fine-tune the pre-trainde model; the amount of changes in weights can be measured (energy perspective )  | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:lOUUCIYAZlQJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD2tFU:AAGBfm0AAAAAYHPwrFVFWPMAEfxbgngLdZrnTjmviyTG&scisig=AAGBfm0AAAAAYHPwrN0IxuIMpGcX6opjL56pCnY0EFHK&scisf=4&ct=citation&cd=-1&hl=en): kuribayashi2020deepwatermark | Kuribayashi et al, *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* 2020  (only overwriting attack)


7. [ Spread-Transform Dither Modulation Watermarking of Deep Neural Network ](https://arxiv.org/pdf/2012.14171.pdf)：changing the activation method of [1], whcih increase the payload (capacity), couping the spread spectrum and dither modulation | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VX0Iu7_rx18J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCpoTs:AAGBfm0AAAAAYHOvuTur8wBmFO7SEImIcTc1vaqgaWZO&scisig=AAGBfm0AAAAAYHOvuUsZDhKHFcZW6NYdvN-oeaeRyD_w&scisf=4&ct=citation&cd=-1&hl=en): li2020spread | Li et al, 2020.12

8. [Watermarking Neural Network with Compensation Mechanism](https://www.jianguoyun.com/p/DV0-NowQ0J2UCRjey-0D): using spread spectrum and a noise sequence for security; 补偿机制指对没有嵌入水印的权值再进行fine-tune; measure changes with norm (energy perspective) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:xed2zy5YT5YJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD-kI4:AAGBfm0AAAAAYHP4iI6opse7jxpYkvyx4yzXtNjTcNYl&scisig=AAGBfm0AAAAAYHP4iKhXdKnITn4E9R_eO2rFPPPjZQXs&scisf=4&ct=citation&cd=-1&hl=en): feng2020watermarking | Feng et al, *International Conference on Knowledge Science, Engineering and Management* 2020

9. [ Delving in the loss landscape to embed robust watermarks into neural networks ](https://www.jianguoyun.com/p/DfA64QMQ0J2UCRjlw-0D)：using patrial weights to embed watermark information and keep it untrainable | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VX0Iu7_rx18J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCpoTs:AAGBfm0AAAAAYHOvuTur8wBmFO7SEImIcTc1vaqgaWZO&scisig=AAGBfm0AAAAAYHOvuUsZDhKHFcZW6NYdvN-oeaeRyD_w&scisf=4&ct=citation&cd=-1&hl=en): tartaglione2020delving | Tartaglione et al, *ICPR*2020

10. [RIGA Covert and Robust White-Box Watermarking of Deep Neural Networks](https://arxiv.org/pdf/1910.14268.pdf)：improvement of [1] in stealthiness, constrain the weights distribution with advesarial training | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:cS8_mQMHQYYJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1otdg:AAGBfm0AAAAAYG5urdhpKjalH8nqYP7SNLkBjTymKK4_&scisig=AAGBfm0AAAAAYG5urdf92Jw__aZTcaPD-r4tpJurQpsx&scisf=4&ct=citation&cd=-1&hl=en): wang2019riga | Wang et al, *WWW* 2021

11. [Adam and the ants: On the influence of the optimization algorithm on the detectability of dnn watermarks](https://www.mdpi.com/1099-4300/22/12/1379/pdf)：improvement of [1] in stealthiness, adoption of the Adam optimiser introduces a dramatic variation on
the histogram distribution of the weights after watermarking, constrain sAdam optimiser is
run on the projected weights using the projected gradients | [code](https://github.com/TIANHAO-WANG/riga) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:cS8_mQMHQYYJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1otdg:AAGBfm0AAAAAYG5urdhpKjalH8nqYP7SNLkBjTymKK4_&scisig=AAGBfm0AAAAAYG5urdf92Jw__aZTcaPD-r4tpJurQpsx&scisf=4&ct=citation&cd=-1&hl=en): cortinas2020adam | Cortiñas-Lorenzo et al, *Entropy* 2020

12. [ When NAS Meets Watermarking: Ownership Verification of DNN Models via Cache Side Channels ](https://arxiv.org/pdf/2102.03523.pdf)：using architecture as watermark | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:vP3iGmOxK_wJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCo2eI:AAGBfm0AAAAAYHOuweKzzDuUHnNorUcpJ4vk3Y6emJwc&scisig=AAGBfm0AAAAAYHOuwVbxoNSwS5Ozc6X2krHowojwjgQ7&scisf=4&ct=citation&cd=-1&hl=en): lou2021when | Lou et al, 2021.2

## Trigger-based

1. []()： | [BibTex]():  |  et al, 2018.2

Adam and the ants: On the influence of the optimization algorithm on the detectability of dnn watermarks

1. [Piracy-Resistant DNN Watermarking by Block-Wise Image Transformation with Secret Key](https://arxiv.org/pdf/2104.04241.pdf)：using the key to enable correct image transformation of triggers; 对trigger进行加密 | [Related Paper: Block-wise Image Transformation with Secret Key for Adversarially Robust Defense](https://arxiv.org/pdf/2010.00801.pdf) | AprilPyone et al, 2021.4

# Secutiry

## Overwriting Attack
1. [Watermarking Neural Network with Compensation Mechanism](https://www.jianguoyun.com/p/DV0-NowQ0J2UCRjey-0D): using spread spectrum and a noise sequence for security; 补偿机制指对没有嵌入水印的权值再进行fine-tune; measure changes with norm (energy perspective) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:xed2zy5YT5YJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD-kI4:AAGBfm0AAAAAYHP4iI6opse7jxpYkvyx4yzXtNjTcNYl&scisig=AAGBfm0AAAAAYHP4iKhXdKnITn4E9R_eO2rFPPPjZQXs&scisf=4&ct=citation&cd=-1&hl=en): feng2020watermarking | Feng et al, *International Conference on Knowledge Science, Engineering and Management* 2020


2. [DeepWatermark: Embedding Watermark into DNN Model](http://www.apsipa.org/proceedings/2020/pdfs/0001340.pdf)：using dither modulation in FC layers  fine-tune the pre-trainde model; the amount of changes in weights can be measured (energy perspective )  | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:lOUUCIYAZlQJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvD2tFU:AAGBfm0AAAAAYHPwrFVFWPMAEfxbgngLdZrnTjmviyTG&scisig=AAGBfm0AAAAAYHPwrN0IxuIMpGcX6opjL56pCnY0EFHK&scisf=4&ct=citation&cd=-1&hl=en): kuribayashi2020deepwatermark | Kuribayashi et al, *Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)* 2020  (only overwriting attack)


## Ambiguty Attack
1. [Rethinking Deep Neural Network Ownership Verification: Embedding Passports to Defeat Ambiguity Attacks](https://openreview.net/pdf?id=BJlfKVBeUr) | [Code](https://github.com/kamwoh/DeepIPR) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:BKAV-WKeJ1AJ:scholar.google.com/&output=citation&scisdr=CgWVvEwREJLC_OQ2dGI:AAGBfm0AAAAAYGcwbGKgqKY6a88Qf5KSWhJ1cZDTLhKp&scisig=AAGBfm0AAAAAYGcwbFH6YVqAHUeAAN6Prl_2T1s73g_a&scisf=4&ct=citation&cd=-1&hl=en):fan2019rethinking | [Extension](https://arxiv.org/pdf/1909.07830.pdf) | Fan et al, *NeuraIPS* 2019, 2019.9

2. [Passport-aware Normalization for Deep Model Protection](https://proceedings.neurips.cc/paper/2020/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf): Improvemnet of [1] | [Code](https://github.com/ZJZAC/Passport-aware-Normalization) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:5QgO8-Ei59MJ:scholar.google.com/&output=citation&scisdr=CgWVvEwREJLC_OQ0VCc:AAGBfm0AAAAAYGcyTCc8vZsk-pFuOhQTxIcQCkrbyaKh&scisig=AAGBfm0AAAAAYGcyTD4o2PSHMinhgmKjFreZiaaIMuHC&scisf=4&ct=citation&cd=-1&hl=zh-CN) | Zhang et al, *NeuraIPS* 2020, 2020.9

## Collusion Attack

1. [DeepMarks: A Secure Fingerprinting Framework for Digital Rights Management of Deep Learning Models](http://www.aceslab.org/sites/default/files/DeepMarks_ICMR.pdf): focusing on the watermark bit, which using Anti Collusion Codes (ACC), e.g., Balanced Incomplete Block Design (BIBID) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:and7Xl29vpgJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3B9Rc:AAGBfm0AAAAAYG7H7Rd27DlL3WE79fbcPDcHgVpDQKuZ&scisig=AAGBfm0AAAAAYG7H7bejDswez8m_t6Y9zhsbsPAMtZ2c&scisf=4&ct=citation&cd=-1&hl=en): chen2019deepmarks | Chen et al, *ICMR* 2019


## Surrogate Model Attack

1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VVOq5e67uCEJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1YmLg:AAGBfm0AAAAAYG5egLj8-8TdhW-OrFR5PtcTAgXDBsUU&scisig=AAGBfm0AAAAAYG5egJ2W418j7bkygIvLDr7B5IUgFq1r&scisf=4&ct=citation&cd=-1&hl=en): zhang2020model | Zhang et al, *AAAI* 2020.2

2. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:_r5iMZdEAsAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1bYNo:AAGBfm0AAAAAYG5deNoV3ooCjF9U9Rk5ckk8f8_ZS956&scisig=AAGBfm0AAAAAYG5deM2L5_2I2AvaWBetKSrL4CFclBGM&scisf=4&ct=citation&cd=-1&hl=en):    zhang2021deep | Zhang al, *TPAMI* 2021.3


# Applications

1. [Model Watermarking for Image Processing Networks](https://arxiv.org/pdf/2002.11088.pdf)：Image Peocessing | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:VVOq5e67uCEJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1YmLg:AAGBfm0AAAAAYG5egLj8-8TdhW-OrFR5PtcTAgXDBsUU&scisig=AAGBfm0AAAAAYG5egJ2W418j7bkygIvLDr7B5IUgFq1r&scisf=4&ct=citation&cd=-1&hl=en): zhang2020model | Zhang et al, *AAAI* 2020.2


2. [Deep Model Intellectual Property Protection via Deep Watermarking](https://arxiv.org/pdf/2103.04980.pdf)：Image Peocessing | [code](https://github.com/ZJZAC/Deep-Model-Watermarking) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:_r5iMZdEAsAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu1bYNo:AAGBfm0AAAAAYG5deNoV3ooCjF9U9Rk5ckk8f8_ZS956&scisig=AAGBfm0AAAAAYG5deM2L5_2I2AvaWBetKSrL4CFclBGM&scisf=4&ct=citation&cd=-1&hl=en):    zhang2021deep | Zhang al, *TPAMI* 2021.3

3. [Watermarking Deep Neural Networks in Image Processing](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9093125)：Image Peocessing, similar to [1] but exclude surrogate model attack | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:EsQcYz3vGkcJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwuwVdtk:AAGBfm0AAAAAYG8TbtkJQ70EfBd6_y4SUgSsCqJiBYKM&scisig=AAGBfm0AAAAAYG8TbkOlrwiGNIYMKOVYBzGpP7VC11zM&scisf=4&ct=citation&cd=-1&hl=en): quan2020watermarking | Quan et al, *TNNLS* 2020


4. [Protect, Show, Attend and Tell: Empower Image Captioning Model with Ownership Protection](https://arxiv.org/pdf/2008.11009.pdf)：Image Caption | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:Hq9e_KZON_EJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3PMvw:AAGBfm0AAAAAYG7JKvyqTMhTSiim967PfzvghJH_-Afl&scisig=AAGBfm0AAAAAYG7JKpGZo5fN_dho1v9lJBI2VxMu3iAH&scisf=4&ct=citation&cd=-1&hl=en): lim2020protect  | Lim et al, 2020.8
(surrogate model attck)


5. [SpecMark: A Spectral Watermarking Framework for IP Protection of Speech Recognition Systems](https://indico2.conference4me.psnc.pl/event/35/contributions/3413/attachments/489/514/Wed-1-8-8.pdf): Automatic Speech Recognition (ASR) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:1mZFIe2pNnAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu3Pais:AAGBfm0AAAAAYG7JcivwfswRKTpDRKVkYNWU4P_fbXQ3&scisig=AAGBfm0AAAAAYG7Jcpms1fjVvGSPVAa8en4_OwmscaUY&scisf=4&ct=citation&cd=-1&hl=en): chen2020specmark | Chen et al, *Interspeech* 2020


6. [Watermarking Neural Language Models based on Backdooring](https://github.com/TIANHAO-WANG/nlm-watermark/blob/master/nlpwatermark.pdf): NLP | Fu et al, 2020.12


7. [Watermarking Graph Neural Networks by Random Graphs](https://arxiv.org/pdf/2011.00512.pdf): Graph Neural Networks (GNN) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:JufA1FwKhYAJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwuwS9bg:AAGBfm0AAAAAYG8U7biP-vr3I4mzYcQD9Ym4MEdwLjlL&scisig=AAGBfm0AAAAAYG8U7TJdjXkL2ClDHPdjSxMsnJ9CjcBY&scisf=4&ct=citation&cd=-1&hl=en): zhao2020watermarking | Zhao et al, *Interspeech* 2020

8. [WAFFLE: Watermarking in Federated Learning](https://arxiv.org/pdf/2011.00512.pdf): Graph Neural Networks (GNN) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:G2k0R8lOVD0J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwuwTIRM:AAGBfm0AAAAAYG8VORNOe-DSrTPF0BkGNKsB_4A3A2zx&scisig=AAGBfm0AAAAAYG8VOeDKjpqOWWl15F21yg6_IN3G8-H1&scisf=4&ct=citation&cd=-1&hl=en): atli2020waffle | Zhao et al, *Interspeech* 2020

8. [Temporal Watermarks for Deep Reinforcement Learning Models](https://personal.ntu.edu.sg/tianwei.zhang/paper/aamas2021.pdf): Deep Reinforcement Learning (DRL) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:pafSRYDd6L8J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvCqb8c:AAGBfm0AAAAAYHOsd8cEKOGOFCslTLOkJ-G7iKF_eCee&scisig=AAGBfm0AAAAAYHOsd6tpE4fU7r41NEcQsfHCyeNPpHaJ&scisf=4&ct=citation&cd=-1&hl=en): chen2021temporal | Chen et al, *International Conference on Autonomous Agents and Multiagent Systems* 2021


# Model&nbsp;Authentication

1. [Verideep: Verifying integrity of deep neural networks through sensitive-sample fingerprinting](https://arxiv.org/pdf/1808.03277.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:n0yJ9gaj-D8J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu5ZWJk:AAGBfm0AAAAAYG1fQJm71SY1B6hD70mtxJmC4eMJoPTd&scisig=AAGBfm0AAAAAYG1fQA9lDg03bZxSS_EeCYZlws6vGDZz&scisf=4&ct=citation&cd=-1&hl=en): he2018verideep | He et al, 2018.8

2. [DeepAttest: An End-to-End Attestation Framework for Deep Neural Networks](http://cseweb.ucsd.edu/~jzhao/files/DeepAttest-isca2019.pdf): the first on-device DNN attestation method
that certifies the legitimacy of the DNN program mapped to the device; device-specific fingerprint | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:eZpX8EPeuCsJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwvc4Wto:AAGBfm0AAAAAYHQ-Qtr3ntZVqjetGSROZcyDovSrlk7q&scisig=AAGBfm0AAAAAYHQ-QujApYeVnUD8NdhVflqQc31eSj4o&scisf=4&ct=citation&cd=-1&hl=en): chen2019deepattest | Chen et al, *ACM/IEEE 46th Annual International Symposium on Computer Architecture (ISCA)* 2019

3. [Reversible Watermarking in Deep Convolutional Neural Networks for Integrity Authentication](https://arxiv.org/pdf/2101.04319.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:mejXG863kv0J:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu5nWns:AAGBfm0AAAAAYG1hQntz72yIxy1aeAbsu0_JJOcDNwo2&scisig=AAGBfm0AAAAAYG1hQkNzQU_uymoeimEEDB7wDmn9CvtT&scisf=4&ct=citation&cd=-1&hl=en): guan2020reversible | Guan et al, *ACM MM* 2020

4. [DeepiSign: Invisible Fragile Watermark to Protect the Integrity and Authenticity of CNN](https://arxiv.org/pdf/2101.04319.pdf) | [BibTex](https://scholar.googleusercontent.com/scholar.bib?q=info:YS9AeQMRxtsJ:scholar.google.com/&output=citation&scisdr=CgVHdjFVEIucwu5ZoVQ:AAGBfm0AAAAAYG1fuVSHK4rIk_Zub8cG9ahS_C59dRnm&scisig=AAGBfm0AAAAAYG1fufkzApCPLjNJxAEOfzaBYseV0xFC&scisf=4&ct=citation&cd=-1&hl=en): abuadbba2021deepisign | Abuadbba et al, *SAC* 2021



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
