>Q1: Is there any alternative to using an influence function?

Please refer to the response to **H11P** Q1.

>Q2: This also seems like a novel way of finding anomalies in a TPP sequence.

We agree with you and have supplemented our work with a simulation experiment demonstrating that RED can successfully identify anomalies in a TPP sequence. We simulated a 1D Hawkes process with intensity function $\lambda(t)=0.5+0.8\int_0^t e^{-s}dN(s),0<t\leq 20$, generating a dataset with 300 sequences. We then divided $(0,20]$ into 20 subintervals of length 1 and randomly selected one subinterval in each sequence to insert anomaly events. The times of these anomalies were chosen randomly and uniformly within the selected interval, and anomalies accounted for 21% of the total events. We applied the RED technique to calculate the weight values of all events, performed a moving average, and predicted the anomaly interval with the smallest weights. RED achieved an accuracy of 89.0%. This demonstrates RED’s ability to successfully detect anomalous intervals in TPPs. Due to space limitations, we will include additional experiments on applying RED to anomaly detection in the camera-ready version.

However, we chose to omit this part in our original paper, as the primary contribution lies in introducing RED as the first general decomposition framework for TPPs, rather than its application to specific tasks such as anomaly detection. While this topic falls outside the scope of the current paper, future research may explore enhanced RED variants for anomaly detection in TPPs.


>Q3: I don't see an experiment of starting with a neural TPP and fitting a second neural TPP as a residual model. Or alternatively, taking a simple TPP and fitting a simple TPP residual. Additionally, one could take it a step further and find another residual. Do you expect this would work or not?

Thanks for your valuable suggestions. We have added experiments using simple TPP + RED + simple TPP and neural TPP + RED + neural TPP to further refine our work. As shown in Table 。。。, the combination of MHP + RED + MHP may yield worse results, as the model complexity of “MHP + MHP” is twice that of a single MHP. Apparently, the residuals do not follow MHP, leading to overfitting.
Then we select NHP as the example base neural model for residual filtering. For each baseline neural TPP, we compare its performance with the original RED using Hawkes and the RED using NHP. The results demonstrate that Residual TPPs with the RED technique consistently outperform the baseline, whether using Hawkes or NHP as the base model. This highlights that the RED technique, as a plug-and-play module, can effectively enhance the performance of TPPs.

We also would like to clarify that one of the key advantages of our method is its lightweight nature. Our goal is to capture statistical properties with a simple TPP and refine the residual part using a neural TPP, thereby accelerating neural TPP computation with fewer events. While combining neural TPP + RED + neural TPP may yield better performance, it would also introduce significantly higher computational complexity.

Additionally, this work is the first decomposition method for TPPs. We use a self-defined weight value to filter and obtain residuals. We believe future work can explore more advanced decomposition methods to derive alternative residuals, offering significant potential for further development.

>Q4: What does section 3.3 say about the connection between the way points are discarded from the Hawkes model and the fact additive intensity implies uniformly rejecting some points?

We use the superposition property of the TPPs. The weight function is used to determine whether an event comes from the Hawkes model or residual model. It shares the similar spirit of using rejection sampling to determine the event type.

>Essential References Not Discussed: It could use a bit more overview of the existing works, and their connections to this paper. No specific papers in mind. I think Lüdke et al. "Add and Thin" (2023) is also relevant as it has some similar ideas, but this paper is distinctly different.

Thanks for the suggestion! Lüdke et al.(2023) 