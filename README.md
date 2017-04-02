# Identification of Coding Regions of the Genome

Team Memebers:
1. Megan Greening
2. Karthik Handady
3. Suresh Kumar

For our project we will aim to identify functional regions of the genome. Genomes can essentially be split into two parts, noncoding regions that do not encode protein sequences and coding regions that do encode protein sequences. A significant part of genomic research is currently focused on the coding regions of genomes. Since there is a large amount of genetic data it makes sense to automate the process for identifying coding regions in a genome. Due to the fact that there is a vagueness in the rules in determining what regions are coding regions it makes sense to use a machine learning approach.

 We will use publicly available datasets from GenBank and GeneBench. These data sets have annotated genomes which identify which regions contain genes and which do not contain genes. We will use annotated genomes both for training, testing, and validation. For accuracy we will measure genes missed, specificity, and sensitivity. There are datasets that exist for several different species. We will begin by exploring the human genome and, time permitting, we may test our algorithm on the datasets for other organisms to see if it works as well as it does on the human genome.
 
	We are especially interested in deep learning and neural networks. Our baseline will be a neural network with a single feature. We do not expect the baseline neural network to perform very well but it will serve as a good indicator for any improvements we make. We will add more complex features after that and compare our final network with our initial work. Identifying functional regions of the genome is a pattern recognition problem, which lends itself nicely to neural network processes. A sliding window technique is used in some gene identification programs where the part of the genome in the window is scored by the neural network and at the end the areas with the highest scores contain genes. We will likely use the sliding window technique and if time permits may try other methods to see if there is an ideal way to use neural networks for gene identification.
	
	For our project we expect initial results and a baseline by April 7th. We will continue to improve the algorithm and add some material to the write up. We will have finished the data and baseline milestone paper on April 13th. We expect to finalize the algorithm and get final results by April 27th. Then we will work on finishing up the write up by May 1st and start working on the video presentation. We will finish putting together and editing the video presentation on May 4th. 
