Thank you for the detailed suggstions.

>Q1: Why is $\phi'(x)$ defined with piecewise quadratic decay instead of smoother alternatives?

We fully agree that further studies on alternative influence functions would enrich this work. In response, we have incorporated new experiments using the function: $\phi'(x)=\frac{(1+\alpha)(x+1)}{(x+1)+\alpha\exp(x)},x\geq-1.$ This function is smooth across its domain and preserves the “unbiasedness” properties stated in Prop 3.1. As shown in the table 。。。, Residual TPP based on two different influence functions (denoted as old/new) both achieve better performance than the baselines. However, the core novelty of our work lies in introducing RED as the first general decomposition framework for TPPs, rather than the specific choice of $\phi'(x)$. The modularity of RED allows $\phi'(x)$ to be replaced with any valid influence function with “unbiasedness” property. We encourage future research to explore enhanced variants or entirely new decomposition paradigms for TPPs.

>Q2: How is the residual threshold $w$ selected in practice?

Fig 4 in Appendix C.2 illustrates the distribution of weight values on different datasets. As observed, each distribution exhibits a truncation near a weight value of 0.8, with a substantial portion of the weights concentrated at 0.
Given this observation, it is natural to consider filtering events based on their weights. The truncation suggests that $w$ can naturally be chosen as any value within $(0,0.8)$.

>Q3: Does RED’s preprocessing introduce overhead that negates training time savings for small datasets? Will it lead to overfitting?

MIMIC-II and Volcano are small datasets with only a few hundred short sequences. In our response to **QaBc**, we presente a table。。。 comparing the end-to-end runtimes between Residual TPP and baseline models. As shown, even with small datasets, the RED’s preprocessing time is negligible compared to the training time of neural TPPs, demonstrating the efficiency of our method.

Regarding overfitting, the Hawkes process is a statistical model with few parameters, making it less prone to overfitting. To further address your concern, we conducted an additional experiment to demonstrate RED’s performance on small Hawkes process datasets. We simulated a 1D Hawkes process with intensity function $\lambda(t)=0.2+0.6\int_0^te^{-1.2s}dN(s)$, generating a dataset with 300 sequences, each containing an average of 36 events. Of these, 200 sequences were used for training and 100 for testing. We applied the RED technique with the same hyperparameter settings as in the paper. The distribution of all event weight values is shown in the figure。。。. The proportion of residual events is only 13%, indicating that most self-exciting patterns have already been captured. Fitting the neural TPP to this small fraction of events will not lead to overfitting.

>Theoretical Claims:

Thank you for giving out this theoretical point. We find that the cumulative probability functions of the integral $\int_{t_{i-1}}^{t_i}\sum_{k=1}^K\lambda_k^{(1)}(u)du$'s for residual and non-residual events are overlapped with each other. Therefore it seems that we cannot separate them perfectly. Regarding unbiased estimation, if there are no residual events, then RED can guarantee the unbiasedness property by choosing $w=0$. On the other hand, if there exist residual events following an arbitrary TPP, then it would be hard to establish the unbiasedness result. We leave it as the future work.

>Experimental Designs Or Analyses & Essential References Not Discussed：

Bae et al.(2023) propose training TPPs within a meta learning framework by framing TPPs as neural processes, where each sequence is treated as a distinct task. While Meta TPP is novel and intersting, it's not a hybrid framework like ours. Moreover, it is not an intensity-based TPP model, which means that the RED technique cannot be directly applied to enhance Meta TPP, as our RED method relies on the intensity function.

As mentioned in Section 3.1, many popular models like Autoformer, FEDformer and DLinear adopt the STD approach to decompose time series. However, they cannot be easily adapted for TPP for comparison due to TPPs' complexity (i.e. discrete event types and irregular spaced event times). Our proposed RED technique makes the first attempt to develop decomposition-based TPP variants.

While RED shares conceptual inspiration with residual learning paradigms like ResNet and Boosting \red{(add cite)}, its design and operational mechanics are fundamentally tailored to the unique challenges of TPPs. 

We will cite these papers and include a discussion in the Camera-ready version.

>Other Comments Or Suggestions:

The Hawkes baseline intensity $\mu_k(t)$ can be modified to a periodic function to capture periodicities. However, in the original RED, we did not do so for two reasons: First, unlike time series data, periodicity in event stream data typically appears only in specific fields such as neuroscience. Most commonly used TPP benchmark datasets do not exhibit periodicities but instead show self-excitation, as discussed in Appendix~B. Therefore, the standard Hawkes process already works well in these benchmark datasets. Second, fitting more complex statistical TPPs increases computational complexity, whereas we aim to keep our method simple and lightweight. For complex dependencies that cannot be captured, neural TPPs can be used for refinement. Our additional experiments in response to **QaBc** may also help clarify this. Due to space limitations, we will include the other statistical TPPs in the Camera-ready version.