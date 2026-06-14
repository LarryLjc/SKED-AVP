# SKED-AVP

## Prediction of Antiviral Peptides

The peptides possessing the potential to inhibit viral infection are considered antiviral peptides (AVPs). Usually, AVPs exert antiviral effects by blocking viral entry, replication, or virus–host interactions. Although experimental approaches can identify AVPs, they are generally time-consuming, laborious, and costly. Therefore, accurate computational tools are urgently needed to accelerate the discovery of antiviral peptides.

In this study, we propose a multi-feature fusion model named **SKED-AVP** for antiviral peptide prediction. SKED-AVP combines **4-mer Word2Vec biological word embeddings** with **Ankh3-XL protein language model representations** to capture both local physicochemical patterns and long-range protein semantic information. Moreover, selective kernel attention and efficient channel attention are introduced to extract multi-scale sequence motifs and recalibrate informative feature channels. Finally, SKED-AVP achieves **94.62% ACC, 91.65% SN, 97.59% SP, 0.8940 MCC, and 0.9772 AUC** on the independent test set. Meanwhile, the model also achieves strong performance on an external AVP dataset and can be generalized to anticancer peptide and DPP-IV inhibitory peptide prediction tasks.

<br>

<h2>Overall Framework</h2>

<img src="https://github.com/user-attachments/assets/de0ee6ac-c035-4751-83fe-59ff863582f0" alt="The overall framework of SKED-AVP" width="100%">

<p align="center">
  <b>Figure 1.</b> The overall framework of the SKED-AVP model.
</p>

<br>
