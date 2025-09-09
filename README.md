# 🔮 AdaptiveRL

**AdaptiveRL** is a research project exploring **Adaptive Reinforcement Learning**:  
an approach where both the **policy** (e.g. GPT-2) and the **optimizer** **co-adapt online**.  
Instead of fixing optimizer hyperparameters in advance, AdaptiveRL allows the optimizer  
to **learn how to learn** from **loss trends, reward signals, and human feedback**.

---

## 📜 Theory

Traditional RL → The policy adapts to maximize reward.  
AdaptiveRL → Both the **policy** *and* the **optimizer** adapt online.  

Key principles:
- **Composite Rewards**  
  - automatic (cosine similarity, regex, length)  
  - learned reward models (sentiment, helpfulness, toxicity)  
  - LM-as-a-judge (GPT scoring responses)  
  - human-in-the-loop (+1/–1 feedback)  
- **Optimizer-as-Agent**  
  - adaptive learning rates, momentum, β-values, weight decay  
  - recurrent optimizers (GRU-based) with per-tensor hidden states  
- **Dual Feedback Loop**  
  - Inner loop: policy gradient updates  
  - Outer loop: optimizer hyperparameter adaptation  

Mathematical view:

\[
\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \ell_t
\]

\[
\eta_{t+1} = f(\eta_t, \ell_t, R_t, \Delta R_t, \text{Var}[R_{t-k:t}])
\]

Where  
- \( \theta \) = policy parameters  
- \( \eta \) = optimizer state (lr, momentum, etc.)  
- \( f \) = adaptation function (heuristics, recurrent dynamics, or learned rules)  

---

## ⚡ Features

- **Online Adaptive Optimizers**
  - `OnlineSGD` (loss & reward-aware)  
  - `OnlineRMSProp` (variance-normalized + adaptive)  
  - `OnlineAdamW` (momentum & LR tuned online)  
- **Reward Functions**
  - cosine similarity (sentence-transformers)  
  - regex & substring matching  
  - learned reward models  
  - GPT-judge scoring  
  - human-in-the-loop interactive feedback  
- **Safe Training**
  - gradient clipping  
  - checkpoint saving every N steps  
  - mixed precision support (fp16, bf16)  

---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/AdaptiveRL.git
cd AdaptiveRL

# Create a venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

Key dependencies:

* `torch`
* `transformers`
* `sentence-transformers`
* `trl`
* `datasets`

---

## 🚀 Usage

1. Prepare a config JSON (example: `config.json`):

```json
{
  "MODEL": "gpt2",
  "TRAIN_FILE": "data.json",
  "OUTPUT_DIR": "checkpoints",
  "BATCH_SIZE": "2",
  "EPOCHS": "3",
  "LRATE": "5e-5",
  "STEPS": "1000",
  "OPTIM": "adamw",          // "sgd", "rmsprop", "adamw"
  "MAXSEQ": "512",
  "BF16": "false",
  "FP16": "true",
  "LOAD_4BIT": "false",
  "LOAD_8BIT": "false",
  "FULLTUNE": "true"
}
```

2. Run training:

```bash
python train.py config.json
```

3. Watch adaptive optimizer logs:

```
Step 50 | Reward: 0.82 | Loss: -0.3456 | LR: 3.2e-05 | Momentum: 0.912
```

---

## 📊 Example Logs

```
[OnlineAdamW] Step 100 dynamics:
  Tensor 0 | grad_signal=0.0123 | hidden_norm=1.5231 | delta_scale=0.9732
  exp_avg_norm=0.2103 | exp_avg_sq_mean=0.000012
```

---

## 📈 Roadmap

* [x] Adaptive SGD
* [x] Adaptive RMSProp
* [x] Adaptive AdamW
* [x] Multi-source reward functions
* [ ] TensorBoard / WandB integration
* [ ] Learned controller for optimizer hyperparams
* [ ] Large-scale experiments with GPT-2 and GPT-J

---

## 🤝 Contributing

Pull requests are welcome! If you have ideas for new adaptation rules
or better reward functions, feel free to open an issue or PR.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

