## Bias Detection using Deep Supervised Contrastive Learning (Goodfellas)



[Sina Aghaei](mailto:saghaei@usc.edu)<sup>1</sup>, [Zahra Abrishami](mailto:zabrisha@usc.edu)<sup>2</sup>, 
[Ali Ghafelehbashi](mailto:ghafeleb@usc.edu)<sup>1</sup>, [Bahareh Harandizadeh](mailto:harandiz@usc.edu)<sup>2</sup>, [Negar Mokhberian](mailto:nmokhber@usc.edu)<sup>2</sup>

<sup>1</sup>Department of Industrial and Systems Engineering, University of Southern California, Los Angeles, CA 90008<br/>
<sup>2</sup>Department of Computer Science, University of Southern California, Los Angeles, CA 90008


## Abstract
In this paper, we propose an end-to-end model to detect ideological bias in news articles. We propose a deep supervised contrastive learning model to learn new representations for text with the goal of separating the embeddings from different classes to expose the bias in the textual data and exploit this bias to identify political orientation of different news articles.

## Introduction

In any nation, media can be a dividing issue as it might reflect topics differently based on the political views or ideological lines. Understanding this implicit bias is becoming more and more critical, specifically by growing numbers of social media and news agencies platforms. Recently the number of research in this domain also is increasing, from detection and mitigation of gender bias [[1]](#1), to polarization detection in political views [[4]](#4) also ideological bias of News [[6]](#6). We think the analysis of bias in the news could be very helpful for the readers as make them more responsible about what they hear or read. 

In this work we want to understand how different are news articles on the same subject but from different political parties (left and right) from each other. We want to detect the potential political bias within news articles. We, as human, can easily identify the different political orientation of articles from opposite parties. For example, how different conservative news agencies such as ''Fox News'' approach a subject like Covid-19 compares to a liberal news agency such as ''CNN''. The question is that can machines detect this political bias as well?

A proxy for this goal could be a classifier which tries to classify news articles depending on their political party. Existing approaches such as [[6]](#6) tackle this problem using a classifier on the space of the words embedding. The problem with this approach is that it is not end to end, i.e., the embedding are not trained with the purpose of getting a good classification result. As we can see in figure 1 (right), with general purpose word embedding models such as BERT [[5]](#5), classifying embedded articles might not be straightforward. Having a new representation such as the one shown in figure 1 (left) where it maximizes the distance between embedding from different classes could make the classification task much easier, as in the latent space, the bias is exposed.
<br />

<p align='center'>
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc593fd4f927164324048'><img src='https://www.linkpicture.com/q/embedding_1.png' type='image' width='400' align="center"></a>
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59442d9bef2013716327'><img src='https://www.linkpicture.com/q/bias_1.png' type='image' width='400' align="center"></a>
</p>


<br />

<p align="center">
  <br />
<b>Figure 1:</b> An ideal latent space (left) where the articles from opposite classes are far from each other which helps to expose the political bias (right) and improves the performance of the classification task.
</p>




To achieve such representation for news articles we propose a modification to the deep contrastive Learning model for unsupervised textual representation introduced in [[3]](#3). In [[3]](#3), they have a unsupervised contrastive loss which for any given textual segment (aka anchor span) it minimizes the distance between its embedding and the embeddings of other textual segments randomly sampled from nearby in the same document (aka positive spans). It also maximizes the distance between the given anchor from other spans which are not in its neighborhood (aka negative spans). In their model, the positive and negative spans are not chosen according to the label of the documents. We propose to alter their objective to a supervised contrastive loss so that the negative spans are sampled from articles with opposite label. The motivation is to maximize the distance between articles from different classes.



## Problem Formulation
We consider a setting where we have various documents (articles) from two different parties called *liberal* (label being 0) and *conservative* (label being 1). All the documents are about a similar topic, <span style="color:red">Covid-19</span>.

- We sample a batch of <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;N" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;N" title="\small N" /></a> documents where the class of each document is specified by the class vector <a href="https://www.codecogs.com/eqnedit.php?latex=Y_{\text{batch}}&space;\in&space;\{0,1\}^N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Y_{\text{batch}}&space;\in&space;\{0,1\}^N" title="Y_{\text{batch}} \in \{0,1\}^N" /></a>. For each document from class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k&space;\in&space;\{0,1\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;k&space;\in&space;\{0,1\}" title="\small k \in \{0,1\}" /></a> we sample <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;A" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;A" title="\small A" /></a> anchor spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s_i,~&space;i&space;\in&space;\{1,\dots,AN\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;s_i,~&space;i&space;\in&space;\{1,\dots,AN\}" title="\small s_i,~ i \in \{1,\dots,AN\}" /></a> and per anchor we sample <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;P" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;P" title="\small P" /></a> positive spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s_{i&plus;pAN},~&space;p&space;\in&space;\{1,\dots,P\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;s_{i&plus;pAN},~&space;p&space;\in&space;\{1,\dots,P\}" title="\small s_{i+pAN},~ p \in \{1,\dots,P\}" /></a> following the procedure introduced in [[3]](#3).

- Given an input span, <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;s_i" title="\small s_i" /></a>, a ''transformer-based language models'' encoder <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;f" title="\small f" /></a>, maps each token in the input span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;s_i" title="\small s_i" /></a> to a word embedding.
    
- Similar to [[3]](#3), a pooler <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g(.)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\small&space;g(.)" title="\small g(.)" /></a>, maps the encoded anchor spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f(s_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;f(s_i)" title="\small f(s_i)" /></a> to a fixed length embedding <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g(f(s_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;g(f(s_i))" title="\small g(f(s_i))" /></a>. 
    

- We take the average of the positive spans per anchor as follows:

<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;e_{i&plus;AN}&space;=&space;\frac{1}{P}\sum_{p=1}^{P}g(f(s_{i&plus;pAN}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;e_{i&plus;AN}&space;=&space;\frac{1}{P}\sum_{p=1}^{P}g(f(s_{i&plus;pAN}))" title="\large e_{i+AN} = \frac{1}{P}\sum_{p=1}^{P}g(f(s_{i+pAN}))" /></a>
</p>


<!--    
<p align="center">
  <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/AvgPosSpan.PNG" width="450" /> 
</p>
 -->
Now we have <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;2(AN)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;2(AN)" title="\small 2(AN)" /></a> datapoints with class vector <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;Y&space;\in&space;\{0,1\}^{2(AN)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;Y&space;\in&space;\{0,1\}^{2(AN)}" title="\small Y \in \{0,1\}^{2(AN)}" /></a>. We define our supervised contrastive loss function as follows

<br>
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\mathcal&space;L_{\text{contrastive}}&space;=&space;\sum_{i=1}^{AN}l(i,i&plus;AN)&space;&plus;&space;l(i&plus;AN,i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\mathcal&space;L_{\text{contrastive}}&space;=&space;\sum_{i=1}^{AN}l(i,i&plus;AN)&space;&plus;&space;l(i&plus;AN,i)" title="\large \mathcal L_{\text{contrastive}} = \sum_{i=1}^{AN}l(i,i+AN) + l(i+AN,i)" /></a>
</p>
<br>

where 

<br>
<p align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;l(i,j)&space;=&space;-\log&space;\frac{exp(sim(e_i,e_j)/\tau)}{\sum_{m=1}^{2AN}exp(sim(e_i,e_m)\mathbb&space;I(Y_i&space;\neq&space;Y_m)/\tau)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;l(i,j)&space;=&space;-\log&space;\frac{exp(sim(e_i,e_j)/\tau)}{\sum_{m=1}^{2AN}exp(sim(e_i,e_m)\mathbb&space;I(Y_i&space;\neq&space;Y_m)/\tau)}" title="\large l(i,j) = -\log \frac{exp(sim(e_i,e_j)/\tau)}{\sum_{m=1}^{2AN}exp(sim(e_i,e_m)\mathbb I(Y_i \neq Y_m)/\tau)}" /></a>
  </p>
<br>


where <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\mathbb&space;I(.)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\mathbb&space;I(.)" title="\small \mathbb I(.)" /></a> is an indicator function. Loss function (2) enforces anchor <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e_i" title="\small e_i" /></a> to be as closes as possible to its corresponding positive span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e_{i&plus;AN}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e_{i&plus;AN}" title="\small e_{i+AN}" /></a> (which is referred to as easy positive) and at the same time to be as far as possible from all spans <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e_m" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e_m" title="\small e_m" /></a> from the opposite party, i.e., <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\mathbb&space;I(Y_i&space;\neq&space;Y_m)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\mathbb&space;I(Y_i&space;\neq&space;Y_m)" title="\small \mathbb I(Y_i \neq Y_m)" /></a> (which are referred to as easy negative). Figure 2 visualizes a simplified overview of our model. 


<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59607f139d808105913'><img src='https://www.linkpicture.com/q/model_1.png' type='image' width="800"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/model.PNG" width="450" />  -->
</p>
<p align="center">
<b>Figure 2:</b> Overview of the supervised contrastive objective. In this figure, we show a simplified example where in each batch we sample 1 document <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;d^k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;d^k" title="\small d^k" /></a> per class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;k" title="\small k" /></a> and we sample 1 anchor span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_i" title="\small e^k_i" /></a> per document and 1 positive span <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_j" title="\small e^k_j" /></a> per anchor. All the spans are fed through the same encoder <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;f" title="\small f" /></a> and pooler <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;g" title="\small g" /></a> to produce the corresponding embedding vectors <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_i" title="\small e^k_i" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_j" title="\small e^k_j" /></a>. The model is trained to minimize the distance between each anchor <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_i" title="\small e^k_i" /></a> and its corresponding positive <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_j" title="\small e^k_j" /></a> and maximize the distance between anchor <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e^k_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e^k_i" title="\small e^k_i" /></a> and all other spans from class <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;1-k" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;1-k" title="\small 1-k" /></a>. 
</p>


## Data
For our experiments we use [AYLIEN’s Coronavirus news dataset](https://aylien.com/blog/coronavirus-news-dashboard) (Global COVID related news since Jan 2020). This dataset contains numerous news articles from different news sources with different political orientations. For simplicity we only focus on news articles from two news sources Huffington Post, which is considered as liberal (class 0), and Breitbart which is considered as conservative (class 1).

In the figure 3 we show the first few lines of the dataset. We assign Huffington's articles class 0 and Breitbart's articles class 1. Another important observation from the data is the distribution of the length (number of words) of the articles which is shown in figure 3. This is important to the step where we sample the anchor-positive pairs from the data. 


<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59610575c3646721762'><img src='https://www.linkpicture.com/q/data_head_1.png' type='image' width="800"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/data_head.png" width="900" />  -->
</p>
<p  align="center">
<b>Figure 3:</b> Overview of the dataset.
</p>

<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc59612ed841794267532'><img src='https://www.linkpicture.com/q/length_distribution_1.png' type='image' width="400"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/length_distribution.png" width="450" />  -->
</p>
<p  align="center">
<b>Figure 4:</b> Overview of the dataset.
</p>

Another step that we do is topic modeling to make sure all the articles are about the same subject ''covid19''. We use Latent Dirichlet Allocation (LDA) for this step. The topics we found are as follows:

- Huffpost people new time home like 19 covid pandemic health help year just
- Trump president donald people house states white pandemic news state virus americans health going huffpost
- Minister china chinese cases italy wuhan government confirmed border reported countries virus prime authorities deaths
- Hanks rita kimmel jimmy wilson cordero aniston kloots fallon elvis song tom actor conan corden
- Newstex al views content et https advice www accuracy commentary authoritative guarantees distributors huffington conferring

It seems that the first topic is about ''covid19'', the second topic is about ''white house announcements'', the third one is about ''global news'', the fourth one is about ''enterntainment'' and the last one is not related to our work. To only keep covid19 related articles we kepth those having at least one of the following keywords, ''covid,covid19,pandemic,vaccine,virus,corona,face covering''.
At the end we are left with 7226 articles from Breitbart (class 1) and 6300 articles from Huffington Post (class 0).

## Code
In this section we discuss the implementation detail of our model. Our code is mainly based on [[3]](#3)'s code located in this [GitHub](https://github.com/JohnGiorgi/DeCLUTR) repository and also the <b>pytorch-metric-learning</b> package located in this [GitHub](https://github.com/KevinMusgrave/pytorch-metric-learning) repository.

The loss function that we are utilizing in our model is a normalized-temperature cross-entropy loss (NTXentLoss) in the following form

<p  align="center">
  <a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\mathcal&space;L_q&space;=&space;-\log\frac{\exp(k.q_{&plus;}/\tau)}{\sum_{i}^{K}\exp(k.q_{i}/\tau)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\mathcal&space;L_q&space;=&space;-\log\frac{\exp(k.q_{&plus;}/\tau)}{\sum_{i}^{K}\exp(k.q_{i}/\tau)}" title="\large \mathcal L_q = -\log\frac{\exp(k.q_{+}/\tau)}{\sum_{i}^{K}\exp(k.q_{i}/\tau)}" /></a>
</p>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;q" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;q" title="\small q" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k_&plus;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;k_&plus;" title="\small k_+" /></a> are anchor-positive pairs and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;q" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;q" title="\small q" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k_-" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;k_-" title="\small k_-" /></a> are anchor-negative pairs. The implementation of <b>NTXentLoss</b> is given in the <b>pytorch-metric-learning</b> package. 
When we call <b>NTXentLoss</b> it calls function <b>compute_loss</b> from class <b>GenericPairLoss</b> which requires the embeddings tensor along with a vector of labels where any two embeddings with similar labels are considered as anchor-positives and any two embedding with different label are considered as negative pairs. 
From here we call another function <b>get_all_pairs_indices</b> from file <b>loss_and_miner_utils.py</b> which creates a 2-dimensional matrix named (Matches) wherein an entry has a value of 1 iff the embeddings corresponding to its column and row are positive pairs. Similarly it creates a matrix named (Diffs)  for the negative pairs wherein any entry at the intersection of a negative pair would have a value of one. Having these two matrices we can implement the loss function given in (3).
In the <b>DeCLUTR</b>'s code, along with the embeddings of all spans within a batch, we pass a vector of labels wherein any anchor-positive pairs would share the same unique label. In this way we can identify the anchor-positive pairs. On the other hand, for a given anchor <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;q" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;q" title="\small q" /></a>, all other embedding other than its own <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k_&plus;" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;k_&plus;" title="\small k_+" /></a> are considered as negative pairs <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;k_-" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;k_-" title="\small k_-" /></a>.
In our model, we want to choose the negative pairs in a smarter way such that any anchor-negative pairs belong to different classes. To do so, along with the embeddings and labels mentioned above, we pass another vector <b>class</b> where it specifies the class of each embedding. We modify the function <b>get_all_pairs_indices</b> such that  using the class vector, it creates an auxiliary matrix such that an entry has a value of 1 iff the embeddings corresponding to its column and row belongs to different classes. We then multiply the negative matrix (<b>Diffs</b>)  defined above by this auxiliary matrix. By doing this any two embeddings from different classes would be negative pairs and the rest is exactly as the <b>DeCLUTR</b>'s model.

Furthermore, we need to update <b>DatasetReader.py</b> such that it specifies the class of each document within a batch.
We then should update the <b>DeCLUTR</b> class in <b>Model.py</b> to incorporate the class vector we get from <b>DatasetReader</b> into the forward pass.


## Experiments
In this section, as one of our baseline methods we train the <b>DeCLUTR</b> model introduced by [[3]](#3) on our covid19 data explained in section "Data", the overview of their model is given in figure 3. 


<p align="center">
  <a href='https://www.linkpicture.com/view.php?img=LPic5fc596bc0dbb01300739986'><img src='https://www.linkpicture.com/q/DeCUTR_1.png' type='image' width="800"></a>
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/DeCUTR.PNG" width="900" />  -->
</p>
<p  align="center">
<b>Figure 5:</b> Overview of the self-supervised contrastive objective. For each document d in a minibatch of size <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;N" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;N" title="\small N" /></a>, we sample A anchor spans per document and <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;P" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;P" title="\small P" /></a> positive spans per anchor. For simplicity, we illustrate the case where <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;A&space;=&space;P&space;=&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;A&space;=&space;P&space;=&space;1" title="\small A = P = 1" /></a> and denote the anchor-positive span pair as <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;s_i" title="\small s_i" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;s_j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;s_j" title="\small s_j" /></a>. Both spans are fed through the same encoder <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;f" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;f" title="\small f" /></a> and pooler <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;g" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;g" title="\small g" /></a> to produce the corresponding embeddings <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e_i&space;=&space;g(f(s_i))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e_i&space;=&space;g(f(s_i))" title="\small e_i = g(f(s_i))" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;e_j&space;=&space;g(f(s_j))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;e_j&space;=&space;g(f(s_j))" title="\small e_j = g(f(s_j))" /></a>. The encoder and pooler are trained to minimize the distance between embeddings via a contrastive prediction task (where the other embeddings in a minibatch are treated as negatives, omitted here for simplicity).
</p>

In the implementation of <b>DeCLUTR</b>, in the process of sampling the anchor-positive spans, they randomly choose the length of each span, with the minimum length <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;l_{\min}=32" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;l_{\min}=32" title="\small l_{\min}=32" /></a> and maximum length <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;l_{\max}=512" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;l_{\max}=512" title="\small l_{\max}=512" /></a>. Furthermore they exclude all articles with less than (<a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\text{num-anchor}*l_{\max}*2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\text{num-anchor}*l_{\max}*2" title="\small \text{num-anchor}*l_{\max}*2" /></a>) words, where num-anchor is the number of anchors sampled per article (For details of the sampling process please refer to the main text of [[3]](#3). According to the distriubtion of length of the articles in our dataset given in figure 4, in order to be able to use most of our data, we set the minimum length to <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;\text{num-anchor}*l_{\max}*2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;\text{num-anchor}*l_{\max}*2" title="\small \text{num-anchor}*l_{\max}*2" /></a> and maximum length to <a href="https://www.codecogs.com/eqnedit.php?latex=\small&space;l_{\max}=100" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\small&space;l_{\max}=100" title="\small l_{\max}=100" /></a> and we sample one anchor per document, i.e., num-anchor=1. Having all this, articles with less than 200 words (1345 of them) would be put aside.
<!-- which we use them as our test set and use the remaining articles (12181 of them) as our training set. -->
For the remaining articles we take 70% of them (8513 articles) as training set and 15% of them (1825 articles) as test. The remaining is used as validation set. We train the <b>DeCLUTR</b> model with the unsupervised contrastive loss on the training data. We set the batch size to 6 and train the model for 10 epochs. We then get the embedding of the test articles under the trained model. The visualization of the embeddings is given in figure 6 (left). The embedding space is 768 dimensional. We applied Principal component analysis (PCA) to get the visualization. As we can see the embeddings are not well-separated from each other.
As our next step, we fit a binary classification model on these embeddings to see how well it can separate the articles from opposite classes. To do so, we fit a logistic regression model on 75% of the test set. The accuracy of the trained binary classifier on the remaining 25% of the data is 74.17%.  

<!-- <p float="center"> -->
<p>
<a href="https://www.linkpicture.com/view.php?img=LPic5fc7165b450c41722436072"><img src="https://www.linkpicture.com/q/declutr_pca_1.jpg" type="image" width="300" style="float: left"></a>
<!--   <a href="https://www.linkpicture.com/view.php?img=LPic5fc6a47dbe2761391136530"><img src="https://www.linkpicture.com/q/DeCLUTR.png" type="image" width="250"></a>  -->
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/declutr_pca.jpg" width="450" />  -->
  <a href="https://www.linkpicture.com/view.php?img=LPic5fc7166822704741007745"><img src="https://www.linkpicture.com/q/fineBERT_pca.jpg" type="image" width="300" style="float: center"></a>
<!--   <a href="https://www.linkpicture.com/view.php?img=LPic5fc6a5028de2c977500894"><img src="https://www.linkpicture.com/q/goodfellas.png" type="image" width="250"></a> -->
<!--   <img src="https://github.com/ghafeleb/goodfellas/blob/main/docs/resources/fineBERT_pca.png" width="450" /> -->
  <a href="https://www.linkpicture.com/view.php?img=LPic5fc7166a4ea2a118899102"><img src="https://www.linkpicture.com/q/goodfellas_pca.jpg" type="image" width="300" style="float: right"></a>
</p>
<p  align="center">
  <b>Figure 6:</b> The visualization of the embeddings from <b>DeCLUTR</b> (left), <b>FineBERT</b> (middle)  and <b>GoodFellas</b> (right) of test data in two dimension. The blue dots belongs to Huffington Post US and the red dots belong to Breitbart
</p>


As our second baseline method, we fine tune BERT [[5]](#5) by adding a classification layer and minimizing the the classification loss. We refer to this approach as <b>FineBERT</b>. A visualization of  the model <b>FineBERT</b> is given in figure 7.

Similar to <b>DeCLUTR</b>, we visualize the embeddings given by <b>FineBERT</b> in two dimension shown in figure 6 (middle). Similar to the case <b>DeCLUTR</b> we fit a logistic regression model on 75% of the test set. The accuracy of the trained binary classifier on the remaining 25% of the data is 78.55%. <b>FineBERT</b> is performing better than <b>DeCLUTR</b>.

At the end we implement our model <b>GoodFellas</b>. We use the same setup as <b>DeCLUTR</b> for training the model. The visualization of the embedding in two dimensions is given in figure 6 (right). As we can see the outcome is much better the previous approaches. The embeddings of two classes are quite separate from each other. In the downstream classification task we outperform the baseline methods as we achieve an out of sample accuracy of 79.21%. You also can see the results summarization in table 1.

<p align="center">
  <a href="https://www.linkpicture.com/view.php?img=LPic5fc71808a680e1407211298"><img src="https://www.linkpicture.com/q/table_4.png" type="image" width="400"></a>
<!--   <a href='https://www.linkpicture.com/view.php?img=LPic5fc5916db37a31261281140'><img src='https://www.linkpicture.com/q/FineTunedBERT.png' type='image' width="600" ></a> -->
</p>
<p  align="center">
<b>Table 1:</b> Different models accuracy on Train and Test sets. GoodFellas shows better accuracy in both sets, also less over-fitting compare to FineBERT.
</p>
<br>

## Conclusion
In this work we proposed a supervised contrastive loss for learning new representation for text with the goal of exploiting the political bias within the text. We showed that in our learned latent space the embeddings from opposite classes are separated from each other and this separation improves the downstream classification task.

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

