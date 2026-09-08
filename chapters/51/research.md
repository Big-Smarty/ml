# Chapter 51 research notes

Model route: GPT-5.6 Luna High bounded primary-source research for the Sol High chapter author. The chapter metadata assigns **Generative modeling beyond text** with topics VAE, GAN, diffusion, sampling, and objectives; the project is to generate a small synthetic distribution. Source claims below were checked against the original papers' stable arXiv records and full text on 2026-09-08. No secondary summaries are used.

## Claim-to-source notes

### Variational autoencoder

Kingma and Welling establish the variational identity

`log p_theta(x) = KL(q_phi(z|x) || p_theta(z|x)) + L(theta, phi; x)`

and therefore the lower bound

`L = E_q[log p_theta(x|z)] - KL(q_phi(z|x) || p_theta(z))`.

The paper's equations (1)--(3) support calling the first term an expected reconstruction log likelihood and the second a KL regularizer. The authors' SGVB estimator and AEVB algorithm use minibatches and stochastic gradients; these are the source for the chapter's negative-ELBO objective and for keeping the reconstruction and KL terms explicit rather than describing a VAE as an ordinary deterministic autoencoder. [Kingma & Welling, *Auto-Encoding Variational Bayes*, §§2.1--2.3, equations (1)--(8)](https://arxiv.org/html/1312.6114v11)

The reparameterization claim must be stated with its condition: for a suitable continuous distribution, sample `z` as a differentiable map `g_phi(epsilon, x)` from parameter-independent noise. Their Gaussian example is `z = mu + sigma epsilon`, with `epsilon ~ N(0, 1)`, and the resulting Monte Carlo expectation is differentiable with respect to `mu` and `sigma`. [Kingma & Welling, §2.4, equations (4)--(5)](https://arxiv.org/html/1312.6114v11) For the VAE example they use a standard normal prior, a Gaussian or Bernoulli decoder selected by data type, and a diagonal-Gaussian approximate posterior whose mean and standard deviation come from an encoder network. [Kingma & Welling, §3](https://arxiv.org/html/1312.6114v11)

### Generative adversarial network

Goodfellow et al. define a differentiable generator `G(z; theta_g)` from a noise prior and discriminator `D(x; theta_d)` that outputs the probability that `x` came from the data distribution. Their original value function is the minimax game

`min_G max_D E_x~p_data[log D(x)] + E_z~p_z[log(1 - D(G(z)))]`.

The paper's equation (1), training algorithm, and text support alternating discriminator updates with generator updates. [Goodfellow et al., *Generative Adversarial Nets*, §3, equation (1), Algorithm 1](https://arxiv.org/html/1406.2661v1)

The non-saturating generator objective is also in the original paper, but it is explicitly presented as a practical alternative: maximize `log D(G(z))` instead of minimizing `log(1-D(G(z)))` because the latter can saturate early when the discriminator confidently rejects poor samples. The paper says the alternative has the same fixed point but stronger early gradients. Label it a heuristic replacement for the generator update, not the minimax value function itself. [Goodfellow et al., §3, immediately after equation (1)](https://arxiv.org/html/1406.2661v1)

Under the paper's idealized infinite-capacity and optimal-discriminator analysis, `D*_G(x) = p_data(x)/(p_data(x)+p_g(x))`; the global minimum is `p_g = p_data`, with discriminator `1/2` and value `-log 4`, via the Jensen--Shannon divergence. This is a theorem about the non-parametric game, not a guarantee for a small finite MLP trained with alternating stochastic updates. [Goodfellow et al., §4.1, Proposition 1 and Theorem 1](https://arxiv.org/html/1406.2661v1) The authors also explicitly note practical synchronization requirements and the possibility that the generator collapses many noise inputs to the same output. [Goodfellow et al., §6](https://arxiv.org/html/1406.2661v1)

### Denoising diffusion probabilistic model

Ho, Jain, and Abbeel define a fixed forward Markov chain that adds Gaussian noise:

`q(x_t | x_{t-1}) = N(sqrt(1-beta_t) x_{t-1}, beta_t I)`.

With `alpha_t = 1-beta_t` and `alpha_bar_t = product_{s=1}^t alpha_s`, their closed-form reparameterization is

`x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon`, `epsilon ~ N(0, I)`.

This is the key implementation shortcut: training can choose one timestep and construct its noisy input directly instead of simulating every earlier forward step. [Ho et al., *Denoising Diffusion Probabilistic Models*, §2 and §3.2, equations (4), (9)--(11)](https://arxiv.org/html/2006.11239v2)

The reverse process is modeled with Gaussian transitions. With the paper's epsilon parameterization, the reverse mean uses the network's prediction `epsilon_theta(x_t,t)`. Their Algorithm 1 samples `x_0`, a uniformly random timestep `t`, and Gaussian noise, then takes a gradient step on the simple noise-prediction objective

`L_simple = E_{t,x_0,epsilon}[||epsilon - epsilon_theta(sqrt(alpha_bar_t)x_0 + sqrt(1-alpha_bar_t)epsilon, t)||^2]`.

The source explicitly calls this a simplified, unweighted variant of the variational bound. It is useful for sample quality and implementation simplicity, but it should not be described as identical to the true likelihood/ELBO objective. [Ho et al., §3.4, equation (14), Algorithm 1](https://arxiv.org/html/2006.11239v2) The reverse sampler starts at `x_T ~ N(0,I)` and applies the learned update from `t=T` down to `1`, adding fresh Gaussian noise for `t>1` and zero noise at the final step. [Ho et al., §3.2, Algorithm 2](https://arxiv.org/html/2006.11239v2)

## Small trainable experiment guidance

Use one course-authored one-dimensional distribution for all comparisons, such as a balanced mixture of two narrow Gaussians. Keep the data split and random seeds fixed. A compact experiment can use scalar inputs and small MLPs:

- VAE: encoder outputs `mu` and `log_variance`, sample with `z = mu + exp(0.5*log_variance)*epsilon`, and train the negative ELBO. For real-valued toy data, state the decoder variance and use its corresponding Gaussian log likelihood; do not silently equate reconstruction MSE with a general likelihood.
- GAN: feed a small noise vector to `G`, train `D` with binary cross-entropy, and use the original paper's non-saturating `-log D(G(z))` generator loss for usable gradients. Alternate one or a few discriminator steps with one generator step and record both losses and discriminator outputs.
- Diffusion: choose a short schedule (for example, `T=16` or `32`), sample one timestep per training example, use the closed-form `x_t`, and train an MLP to predict the injected scalar noise. Sampling must visibly perform the full reverse loop; a single denoiser call is not DDPM sampling.

The toy setup is for tracing objectives and sampling, not for reproducing the papers' image results. Use `f64`, a deterministic seeded RNG, finite-value checks, and a small default run. Keep the sample count and training steps explicit so the CLI remains quick. The experiment should show a held-out comparison: generated histogram or sorted quantiles, mean and variance, and coverage of both mixture modes. Repeat a few seeds when judging stability. A falling training objective alone does not establish that the generated distribution is correct.

Evaluation is model-specific. A VAE can expose a reconstruction term and an ELBO estimate, but the bound depends on the chosen decoder likelihood and is not directly comparable to GAN or DDPM losses. A vanilla GAN has no tractable likelihood in the original formulation; use held-out sample statistics, mode coverage, nearest-neighbor distances, and optionally a small held-out-vs-generated classifier. A DDPM's simple noise MSE measures denoising, not directly sample quality; evaluate samples after the complete reverse chain and report the timestep count. On a tiny dataset, nearest-neighbor checks and a train/test split help distinguish interpolation from memorization, while mode coverage catches a GAN that has low-looking loss but emits one mode.

## Caveats for the chapter's scope

- The VAE reparameterization argument is for continuous latent variables and differentiable transformations. Discrete latent variables need a different estimator or relaxation and are outside this chapter's tiny project.
- GAN convergence statements assume capacity, an optimal discriminator at each generator step, and suitable updates. Small alternating MLPs can oscillate, saturate, or collapse; show this as a failure mode rather than promising convergence.
- DDPM sampling is iterative and its cost grows with `T`. A short schedule makes the Rust project practical but changes the quality/computation tradeoff and does not reproduce the paper's CIFAR-10 or LSUN claims.
- All three papers study settings much larger or more idealized than the course-authored distribution. Any printed toy metric is an implementation check for that fixture, not evidence of broad generative quality.
- Do not compare raw VAE negative-ELBO, GAN adversarial losses, and DDPM noise MSE on one leaderboard. They optimize different surrogates with different scales.

## Verified primary sources

1. Diederik P. Kingma and Max Welling. “Auto-Encoding Variational Bayes.” 2013. Stable record: https://arxiv.org/abs/1312.6114 ; full text: https://arxiv.org/html/1312.6114v11
2. Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. “Generative Adversarial Nets.” 2014. Stable record: https://arxiv.org/abs/1406.2661 ; full text: https://arxiv.org/html/1406.2661v1
3. Jonathan Ho, Ajay Jain, and Pieter Abbeel. “Denoising Diffusion Probabilistic Models.” 2020. Stable record: https://arxiv.org/abs/2006.11239 ; full text: https://arxiv.org/html/2006.11239v2
4. Jiaming Song, Chenlin Meng, and Stefano Ermon. “Denoising Diffusion Implicit Models.” 2020. Stable record: https://arxiv.org/abs/2010.02502 — source for the non-Markovian family and deterministic sampling case used by the course miniature.

No course code, datasets, or prose were copied from these papers. Equations are paraphrased in plain text for author use; the lesson should explain every symbol and state whether a term is averaged over samples or timesteps.

## Astra High review and correction — 2026-09-08

The user-directed ownership is GPT-6 Astra High, reviewing the existing Sol High draft and implementing corrections. Bounded read-only mathematical/source verification was delegated to GPT-5.6 Luna High (`verify_math`), which browsed original Switch, S4, linear-attention, AEVB, GAN, and DDIM sources. The author integrated its evidence and independently inspected all chapter lecture, metadata, reference code, starters, and Rustlings exercise/solution files.

Confirmed beta-VAE KL and reparameterization, non-saturating alternating GAN updates, and deterministic DDIM transition against the original papers. Added an independent quadratic-gradient check and analytic finite-grid VAE expectation check. The executable now reports common-seed 512-sample diagnostics for all three models and distinguishes VAE decoder means from full unit-observation-variance samples. Disclosed the fixed grid's variance 0.6567 and its sampling mismatch. All three affine Gaussian samplers cannot represent the two modes. Added nonzero-log-variance exercise tests so omitting the required factor one-half fails.
