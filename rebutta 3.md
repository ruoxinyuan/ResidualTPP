>Q1: Is the threshold w arbitrarily chosen? If not how do you choose a good value?

Please refer to the response to **H11P** Q2.

>Q2: Why the proposed stepwise approach is lightweight?

Residual TPP indeed comprises a 3-step procedure: (1) modeling the self-exciting patterns with Hawkes; (2) applying the RED technique to filter the residuals; (3) training a neural TPP on the residuals. We evaluate efficiency by comparing the runtime/epoch for Step 3 with that of baseline models, as the neural TPP requires significantly more computation and is the main contributor to the overall runtime. Therefore, the majority of the computational complexity in ResTPP arises from this step. In contrast, the Hawkes fitting in Step 1 uses the Tick library, which is computationally efficient and typically requires a very short amount of time. Step 2 involves calculating weights based on the Hawkes process, which is even faster due to the fixed parametric form of the intensity function.

We acknowledge your point and have expanded our evaluation to include end-to-end runtime comparisons (Hawkes fitting + RED filtering + neural TPP training) against baselines. We report the results in Table。。。. 
The first row of the table reports the runtime for Steps 1 and 2, representing the mean computational time across 10 independent trials. These comparisons further demonstrate the overall computational efficiency of ResTPP, showing that, despite its stepwise nature, it remains more computationally efficient than existing neural TPPs.

>Experimental Designs Or Analyses:

While neural TPPs can theoretically capture both $\lambda^{(1)}+\lambda^{(2)}$ as the width or depth increases, they face practical limitations that deep neural networks require large datasets and long training times to learn complex patterns. 
Under the same model architecture, the RED technique can easily improve the performance comparing using the neural TPP only. This is because the neural TPP in our method only needs to model residual events, leading to better generalization performance.

Additionally, ResTPP introduces only a few hyperparameters in the RED step: $a,b,\rho_1,\rho_2$, which control the filtering ratio of residual events through tuning. Stricter filtering makes ResTPP increasingly resemble a neural TPP. Overall, ResTPP requires minimal additional hyperparameter tuning, making it an easy-to-implement approach.

>Other Strengths And Weaknesses: The RED technique has limited scope where the true signal is from hawkes process.  
>Suggestion 1: I think the authors should motivate more the concept of periodicity in event streams. In the paper it is limited to hawkes data.  
>Suggestion 3: can the author think of creating a synthetic example where the residuals are clearly known and conduct experiments?

Thank you for the comment, but we respectfully disagree. While the paper uses the Hawkes process as a representative example of a statistical TPP for clarity, RED is intentionally designed as a plug-and-play module that can integrate with any base TPP model. The use of Hawkes processes in experiments is motivated by their common application in modeling self-excitation, not as a limitation of RED's scope.  

The residuals identified by RED are agnostic to the true data-generating process. RED quantifies how well the statistical model (e.g., Hawkes) explains the observed events through its weight function. Even if the true process deviates significantly from the base model, RED can isolate the unexplained residuals for refinement by neural TPPs. This is similar to residual learning in deep neural networks, where residuals represent deviations from a simpler base function, regardless of its exact form.  

To validate RED's robustness, we further simulate datasets using diffferent main intensities $\lambda^{(1)}$ and residual intensities $\lambda^{(2)}$, and apply RED to show its effectiveness in these settings.
(1) Poisson-based: We generate a non-homogenous Possion process with 5 event types, each with a different periodic triangular function for $\lambda_k^{(1)}$, and set residual events to follow $\lambda_k^{(2)}=0.1$, a homogeneous Poisson process. The combination of these two generates a Poisson-based dataset.
(2) AttNHP-based: We use the AttNHP \red{cite here} model for $\lambda^{(1)}$ and homogeneous Poisson process $\lambda_k^{(2)}=0.1$ for residuals.
(3) Possion+AttNHP: We use the same periodic non-homogenous Possion process for $\lambda^{(1)}$ and AttNHP for $\lambda^{(2)}$.
The descriptive statistics for the simulated dataset are provided in the table。。。.

We compare the performance of ResTPP and baseline neural TPPs on these simulated datasets. As shown in table。。。, ResTPP consistently improves the performance of neural TPPs through RED, even when the true signal does not follow a Hawkes process or exhibits periodicity.

>Suggestion 2: I also think the authors can explain eqn 2-4 better, maybe with an example.

Appendix C.1. provides a detailed analysis of the influence function $\phi'(x)$, with Fig 3 visualize its behavior under different parameter settings. Fig 4 in Appendix C.2. illustrates the distribution of weight values $W_i(S;\theta)$. We have already cited the arXiv paper *Learning under Commission and Omission Event Outliers* in Section 3.3. The construction of the weight function is inspired by theirs. However, we appreciate your suggestion and will make it clearer.

>Relation To Broader Scientific Literature:

Thanks for pointing out some valuable related work. Loison et al.(2024) introduce UNHaP, a novel framework designed to differentiate structured physiological events, modeled through marked Hawkes processes, from spurious detections, modeled as Poisson noise. UNHaP assumes that true signal follows a Hawkes process with specific Poisson noise, whereas our method offers greater flexibility in handling arbitray noise.
Zhang et al.(2021) propose a less computationally-friendly method to select exogenous events through the best subset selection framework, whereas our method is more lightweight and efficient.
We will include more discussions on the literature review in the Camera-ready version.