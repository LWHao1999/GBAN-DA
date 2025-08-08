# GBAN-DA: Domain Adversarial Gated Bilinear Attention Networks for Cross-Domain Drug Target Interaction Prediction

## Requirements
python==3.8.20
torch>=1.7.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
rdkit~=2021.03.2

## Datasets
The datasets folder contains all experimental data used in GBAN-DA: BindingDB [1], BioSNAP [2] and Human [3]. In datasets/bindingdb and datasets/biosnap folders, we have full data with two random and clustering-based splits for both in-domain and cross-domain experiments. In datasets/human folder, there is full data with random split for the in-domain experiment, and with cold split to alleviate ligand bias.

## Usage
For the in-domain experiments with vanilla GBAN-DA, you can directly run the following command. ${dataset} could either be bindingdb, biosnap and human. ${split_task} could be random and cold.

You can run this line of code command.  

" $ python main.py --cfg "configs/GBAN.yaml" --data ${dataset} --split ${split_task}" "

For the cross-domain experiments with vanilla GBAN-DA, you can directly run the following command. ${dataset} could beither bindingdb, biosnap.

You can run this line of code command. 

" $ python main.py --cfg "configs/GBAN-DA_Non_DA.yaml" --data ${dataset} --split "cluster" "

For the cross-domain experiments with CDAN GBAN-DA, you can directly run the following command. ${dataset} could beither bindingdb, biosnap.

You can run this line of code command. 

" $ python main.py --cfg "configs/GBAN_DA.yaml" --data ${dataset} --split "cluster" "


