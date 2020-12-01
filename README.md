## Bias detection using Deep Supervised Contrastive Learning (Goodfellas)
#### [[Project Website]](https://ghafeleb.github.io/goodfellas/)


[Sina Aghaei](mailto:saghaei@usc.edu)<sup>1</sup>, [Zahra Abrishami](mailto:zabrisha@usc.edu)<sup>2</sup>, 
[Ali Ghafelehbash](mailto:ghafeleb@usc.edu)<sup>1</sup>, [Bahareh Harandizadeh](mailto:harandiz@usc.edu)<sup>2</sup>, [Negar Mokhberian](mailto:nmokhber@usc.edu)<sup>2</sup>

<sup>1</sup>Department of Industrial and Systems Engineering, University of Southern California, Los Angeles, CA 90008<br/>
<sup>2</sup>Department of Computer Science, University of Southern California, Los Angeles, CA 90008


## Abstract
In this paper, we propose an end-to-end model to detect ideological bias in news articles. We propose a deep supervised contrastive learning model to learn new representations for text with the goal of separating the embeddings from different classes to expose the bias in the textual data and exploit this bias to identify political orientation of different news articles.

## Introduction

In any nation, media can be a dividing issue as it might reflect topics differently based on the political views or ideological lines. Understanding this implicit bias is becoming more and more critical, specifically by growing numbers of social media and news agencies platforms. Recently the number of research in this domain also is increasing, from detection and mitigation of gender bias[[1]](#1), to polarization detection in political views[[4]](#4) also ideological bias of News[[6]](#6). We think the analysis of bias in the news could be very helpful for the readers as make them more responsible about what they hear or read. 

In this work we want to understand how different are news articles on the same subject but from different political parties (left and right) from each other. We want to detect the potential political bias within news articles. We, as human, can easily identify the different political orientation of articles from opposite parties. For example, how different conservative news agencies such as ''Fox News'' approach a subject like Covid-19 compares to a liberal news agency such as ''CNN''. The question is that can machines detect this political bias as well?
%
A proxy for this goal could be a classifier which tries to classify news articles depending on their political party. Existing approaches such as[[6]](#6) tackle this problem using a classifier on the space of the words embedding. The problem with this approach is that it is not end to end, i.e., the embedding are not trained with the purpose of getting a good classification result. As we can see in figure 1 (right), with general purpose word embedding models such as BERT[[5]](#5), classifying embedded articles might not be straightforward. Having a new representation such as the one shown in figure 1 (left) where it maximizes the distance between embedding from different classes could make the classification task much easier, as in the latent space, the bias is exposed.


<p float="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc593fd4f927164324048'><img src='https://www.linkpicture.com/q/embedding_1.png' type='image' width='250'></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/embedding.PNG" width="450" />  -->
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59442d9bef2013716327'><img src='https://www.linkpicture.com/q/bias_1.png' type='image' width='250'></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/bias.PNG" width="450" /> -->
</p>
<p align="center">
<b>Figure 1:</b> An ideal latent space (left) where the articles from opposite classes are far from each other which helps to expose the political bias (right) and improves the performance of the classification task.
</p>




To achieve such representation for news articles we propose a modification to the deep contrastive Learning model for unsupervised textual representation introduced in [[3]](#3). In [[3]](#3), they have a unsupervised contrastive loss which for any given textual segment (aka anchor span) it minimizes the distance between its embedding and the embeddings of other textual segments randomly sampled from nearby in the same document (aka positive spans). It also maximizes the distance between the given anchor from other spans which are not in its neighborhood (aka negative spans). In their model, the positive and negative spans are not chosen according to the label of the documents. We propose to alter their objective to a supervised contrastive loss so that the negative spans are sampled from articles with opposite label. The motivation is to maximize the distance between articles from different classes.



## Problem Formulation
We consider a setting where we have various documents (articles) from two different parties called *liberal* (label being 0) and *conservative* (label being 1). All the documents are about a similar topic, <span style="color:red">Covid-19</span>.

We sample a batch of <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;N" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;N" title="\small N" /></a> documents from the *liberal* party (class label being 0) and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;N" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;N" title="\small N" /></a> documents from the *conservative* party (class label being 1). For each document from class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k&space;\in&space;\{0,1\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;k&space;\in&space;\{0,1\}" title="\small k \in \{0,1\}" /></a> we sample <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;A" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;A" title="\small A" /></a> anchor spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s^k_i,~&space;i&space;\in&space;\{1,\dots,AN\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;s^k_i,~&space;i&space;\in&space;\{1,\dots,AN\}" title="\small s^k_i,~ i \in \{1,\dots,AN\}" /></a> and per anchor we sample <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;P" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;P" title="\small P" /></a> positive spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s^k_{i&plus;pAN},~&space;p&space;\in&space;\{1,\dots,P\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;s^k_{i&plus;pAN},~&space;p&space;\in&space;\{1,\dots,P\}" title="\small s^k_{i+pAN},~ p \in \{1,\dots,P\}" /></a> following the procedure introduced in [[3]](#3).

Given an input span, <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s^k_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;s^k_i" title="\small s^k_i" /></a>, a ''transformer-based language models'' encoder <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;f" title="\small f" /></a>, maps each token in the input span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s^k_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;s^k_i" title="\small s^k_i" /></a> to a word embedding.
    
Similar to [[3]](#3), a pooler <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g(.)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;g(.)" title="\small g(.)" /></a>, maps the encoded anchor spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f(s^k_i)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;f(s^k_i)" title="\small f(s^k_i)" /></a> to a fixed length embedding <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g(f(s^k_i))" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;g(f(s^k_i))" title="\small g(f(s^k_i))" /></a>. 
    

We take the average of the positive spans per anchor as follows:

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;e^k_{i&plus;AN}&space;=&space;\frac{1}{P}\sum_{p=1}^{P}g(f(s^k_{i&plus;pAN})" target="_blank"><img src="https://latex.codecogs.com/png.latex?\LARGE&space;e^k_{i&plus;AN}&space;=&space;\frac{1}{P}\sum_{p=1}^{P}g(f(s^k_{i&plus;pAN})" title="\LARGE e^k_{i+AN} = \frac{1}{P}\sum_{p=1}^{P}g(f(s^k_{i+pAN})" width="200" /></a>
</p>


<!--    
<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/AvgPosSpan.PNG" width="450" /> 
</p>
 -->
Now we have <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;2(AN)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;2(AN)" title="\small 2(AN)" /></a> datapoints per party and in total <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;4(AN)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;4(AN)" title="\small 4(AN)" /></a> datapoints per batch. 

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;\mathcal&space;L_{\text{contrastive}}&space;=&space;\sum_{k&space;\in&space;\{0,1\}}\mathcal&space;L^k" target="_blank"><img src="https://latex.codecogs.com/png.latex?\LARGE&space;\mathcal&space;L_{\text{contrastive}}&space;=&space;\sum_{k&space;\in&space;\{0,1\}}\mathcal&space;L^k" title="\LARGE \mathcal L_{\text{contrastive}} = \sum_{k \in \{0,1\}}\mathcal L^k" width="200" /></a>
</p>

<br>
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;\mathcal&space;L^k&space;=&space;\sum_{i=1}^{AN}l^k(i,i&plus;AN)&space;&plus;&space;l^k(i&plus;AN,i)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\LARGE&space;\mathcal&space;L^k&space;=&space;\sum_{i=1}^{AN}l^k(i,i&plus;AN)&space;&plus;&space;l^k(i&plus;AN,i)" title="\LARGE \mathcal L^k = \sum_{i=1}^{AN}l^k(i,i+AN) + l^k(i+AN,i)" width="200" /></a>
</p>

where

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\LARGE&space;l^k(i,j)&space;=&space;-\log&space;\frac{exp(sim(e^k_i,e^k_j)/\tau)}{\sum_{m=1}^{2AN}exp(sim(e^k_i,e^{1-k}_m)/\tau)}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\LARGE&space;l^k(i,j)&space;=&space;-\log&space;\frac{exp(sim(e^k_i,e^k_j)/\tau)}{\sum_{m=1}^{2AN}exp(sim(e^k_i,e^{1-k}_m)/\tau)}" title="\LARGE l^k(i,j) = -\log \frac{exp(sim(e^k_i,e^k_j)/\tau)}{\sum_{m=1}^{2AN}exp(sim(e^k_i,e^{1-k}_m)/\tau)}" width="200" /></a>
</p>


Loss function (3) enforces anchor <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;e^k_i" title="\small e^k_i" /></a> to be as closes as possible to its corresponding positive span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_{i&plus;AN}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;e^k_{i&plus;AN}" title="\small e^k_{i+AN}" /></a> (which is referred to as easy positive) and at the same time to be as far as possible from all spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^{1-k}_m" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;e^{1-k}_m" title="\small e^{1-k}_m" /></a> from the opposite party, i.e., for any given anchor from class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;k" title="\small k" /></a>, the corresponding set of negative spans only include the spans from opposite class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;1-k" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;1-k" title="\small 1-k" /></a> (which are referred to as easy negative). Figure 2 visualizes a simplified overview of our model.

<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59607f139d808105913'><img src='https://www.linkpicture.com/q/model_1.png' type='image' width="600"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/model.PNG" width="450" />  -->
</p>
<p align="center">
<b>Figure 2:</b> Overview of the supervised contrastive objective. In this figure, we show a simplified example where in each batch we sample 1 document <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;d^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;d^k" title="\small d^k" /></a> per class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;k" title="\small k" /></a> and we sample 1 anchor span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_i" title="\small e^k_i" /></a> per document and 1 positive span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_j" title="\small e^k_j" /></a> per anchor. All the spans are fed through the same encoder <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;f" title="\small f" /></a> and pooler <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;g" title="\small g" /></a> to produce the corresponding embedding vectors <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_i" title="\small e^k_i" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_j" title="\small e^k_j" /></a>. The model is trained to minimize the distance between each anchor <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_i" title="\small e^k_i" /></a> and its corresponding positive <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_j" title="\small e^k_j" /></a> and maximize the distance between anchor <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_i" title="\small e^k_i" /></a> and all other spans from class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;1-k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;1-k" title="\small 1-k" /></a>. 
</p>


## Data
For our experiments we use [AYLIEN’s Coronavirus news dataset](https://aylien.com/blog/coronavirus-news-dashboard) (Global COVID related news since Jan 2020). This dataset contains numerous news articles from different news sources with different political orientations. For simplicity we only focus on news articles from two news sources Huffington Post, which is considered as liberal (class 0), and Breitbart which is considered as conservative (class 1).

In the figure 3 we show the first few lines of the dataset. We assign Huffington's articles class 0 and Breitbart's articles class 1. Another important observation from the data is the distribution of the length (number of words) of the articles which is shown in figure 3. This is important to the step where we sample the anchor-positive pairs from the data. 


<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59610575c3646721762'><img src='https://www.linkpicture.com/q/data_head_1.png' type='image' width="600"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/data_head.png" width="900" />  -->
</p>
<p  align="center">
<b>Figure 3:</b> Overview of the dataset.
</p>

<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59612ed841794267532'><img src='https://www.linkpicture.com/q/length_distribution_1.png' type='image' width="250"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/length_distribution.png" width="450" />  -->
</p>
<p  align="center">
<b>Figure 3:</b> Overview of the dataset.
</p>

Another step that we do is topic modeling to make sure all the articles are about the same subject ''covid19''. We use Latent Dirichlet Allocation (LDA) for this step. The topics we found are as follows:

- Huffpost people new time home like 19 covid pandemic health help year just
- Trump president donald people house states white pandemic news state virus americans health going huffpost
- Minister china chinese cases italy wuhan government confirmed border reported countries virus prime authorities deaths
- Hanks rita kimmel jimmy wilson cordero aniston kloots fallon elvis song tom actor conan corden
- Newstex al views content et https advice www accuracy commentary authoritative guarantees distributors huffington conferring

It seems that the first topic is about ''covid19'', the second topic is about ''white house announcements'', the third one is about ''global news'', the fourth one is about ''enterntainment'' and the last one is not related to our work. To only keep covid19 related articles we kepth those having at least one of the following keywords, ''covid,covid19,pandemic,vaccine,virus,corona,face covering''.
At the end we are left with 7226 articles from Breitbart (class 1) and 6300 articles from Huffington Post (class 0).

## Experiments
In this section, as one of our baseline methods we train the <b>DeCLUTR</b> model introduced by [[3]](#3) on our covid19 data explained in section "Data", the overview of their model is given in figure 3. 


<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc596bc0dbb01300739986'><img src='https://www.linkpicture.com/q/DeCUTR_1.png' type='image' width="600"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/DeCUTR.PNG" width="900" />  -->
</p>
<p  align="center">
<b>Figure 4:</b> Overview of the self-supervised contrastive objective. For each document d in a minibatch of size <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;N" title="\small N" /></a>, we sample A anchor spans per document and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;P" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;P" title="\small P" /></a> positive spans per anchor. For simplicity, we illustrate the case where <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;A&space;=&space;P&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;A&space;=&space;P&space;=&space;1" title="\small A = P = 1" /></a> and denote the anchor-positive span pair as <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;s_i" title="\small s_i" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;s_j" title="\small s_j" /></a>. Both spans are fed through the same encoder <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;f" title="\small f" /></a> and pooler <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;g" title="\small g" /></a> to produce the corresponding embeddings <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e_i&space;=&space;g(f(s_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e_i&space;=&space;g(f(s_i))" title="\small e_i = g(f(s_i))" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e_j&space;=&space;g(f(s_j))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e_j&space;=&space;g(f(s_j))" title="\small e_j = g(f(s_j))" /></a>. The encoder and pooler are trained to minimize the distance between embeddings via a contrastive prediction task (where the other embeddings in a minibatch are treated as negatives, omitted here for simplicity).
</p>

In the implementation of <b>DeCLUTR</b>, in the process of sampling the anchor-positive spans, they randomly choose the length of each span, with the minimum length <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;l_{\min}=32" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;l_{\min}=32" title="\small l_{\min}=32" /></a> and maximum length <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;l_{\max}=512" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;l_{\max}=512" title="\small l_{\max}=512" /></a>. Furthermore they exclude all articles with less than (<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\text{num-anchor}*l_{\max}*2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\text{num-anchor}*l_{\max}*2" title="\small \text{num-anchor}*l_{\max}*2" /></a>) words, where num-anchor is the number of anchors sampled per article (For details of the sampling process please refer to the main text of [[3]](#3). According to the distriubtion of length of the articles in our dataset given in figure 4, in order to be able to use most of our data, we set the minimum length to <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\text{num-anchor}*l_{\max}*2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\text{num-anchor}*l_{\max}*2" title="\small \text{num-anchor}*l_{\max}*2" /></a> and maximum length to <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;l_{\max}=100" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;l_{\max}=100" title="\small l_{\max}=100" /></a> and we sample one anchor per document, i.e., num-anchor=1. Having all this, articles with less than 200 words (1345 of them) would be put aside which we use them as our test set and use the remaining articles (12181 of them) as our training set.
<br>
We train the <b>DeCLUTR</b> model with the unsupervised contrastive loss on the training data. We then get the embedding of the test articles under the trained model. The visualization of the embeddings is given in figure 6 (left). The embedding space is 768 dimensional. We applied Principal component analysis (PCA) to get the visualization. As we can see the embeddings are not well-separated from each other.
As our next step, we fit a binary classification model on these embeddings to see how well it can separate the articles from opposite classes. To do so, we fit a logistic regression model on 75% of the test set. The accuracy of the trained binary classifier on the remaining 25% of the data is 85.45%.  

<p float="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59700f367b1173027377'><img src='https://www.linkpicture.com/q/declutr_pca.jpg' type='image' width="250"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/declutr_pca.jpg" width="450" />  -->
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59770e9fc8358745015'><img src='https://www.linkpicture.com/q/fineBERT_pca.png' type='image' width="250"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/fineBERT_pca.png" width="450" /> -->
</p>
<p  align="center">
  <b>Figure 4:</b> The visualization of the embeddings from <b>DeCLUTR</b> (left) and <b>FineBERT</b> (right) of test data in two dimension.
</p>


As our second baseline method, we fine tune BERT [[5]](#5) by adding a classification layer and minimizing the the classification loss. We refer to this approach as <b>FineBERT</b>. A visualization of  the model <b>FineBERT</b> is given in figure 7.

Similar to <b>DeCLUTR</b>, we visualize the embeddings given by <b>FineBERT</b> in two dimension shown in figure 6 (right). Similar to the case <b>DeCLUTR</b> we fit a logistic regression model on 75% of the test set. The accuracy of the trained binary classifier on the remaining 25% of the data is 55.78%. <b>FineBERT</b> is performing worse than <b>DeCLUTR</b> which shows the power of the self-supervised contrastive loss.

So far we have implemented the baseline methods <b>DeCLUTR</b> and <b>FineBERT</b> and we can see that there is room for improvement. The out of sample accuracy of the downstream classification task is not good and we believe that our proposed model <b>GoodFellas</b> would improve upon that.

<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc5916db37a31261281140'><img src='https://www.linkpicture.com/q/FineTunedBERT.png' type='image' width="600" ></a>
</p>
<p  align="center">
<b>Figure 7</b>
</p>
<br>

## References
<a id="1">[1]</a> 
Lucas Dixon, John Li, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. Measuring and mitigating unintended bias in text classification. In proceedings of the 2018AAAI/ACM Conference on AI, Ethics, and Society, pages 67–73, 2018.

<a id="2">[2]</a> 
Matt Gardner, Joel Grus, Mark Neumann, Oyvind Tafjord, Pradeep Dasigi, Nelson F.Liu, Matthew Peters, Michael Schmitz, and Luke S. Zettlemoyer.  Allennlp:  A deepsemantic natural language processing platform. 2017

<a id="3">[3]</a> 
John M Giorgi, Osvald Nitski, Gary D Bader, and Bo Wang. Declutr: Deep contrastivelearning for unsupervised textual representations.arXiv preprint arXiv:2006.03659, 2020.

<a id="4">[4]</a> 
Jon Green, Jared Edgerton, Daniel Naftel, Kelsey Shoub, and Skyler J. Cranmer. Elusiveconsensus:  Polarization in elite communication on the COVID-19 pandemic. Science Advances, 6(28):eabc2717, July 2020.

<a id="5">[5]</a> 
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In advances in neural information processing systems, pages 3111–3119, 2013.

<a id="6">[6]</a> 
Negar Mokhberian, Andrés Abeliuk, Patrick Cummings, and Kristina Lerman. Moralframing and ideological bias of news.arXiv preprint arXiv:2009.12979, 2020.

