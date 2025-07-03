import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Simulate behavior using known parameters ---
n_states = 2
n_actions = 2
n_trials = 1000
true_alpha = 0.3
true_beta = 3

np.random.seed(0)
data = []
Q = np.zeros((n_states, n_actions))

for t in range(n_trials):
    state = np.random.choice(n_states)
    action_probs = np.exp(true_beta * Q[state]) / np.sum(np.exp(true_beta * Q[state]))
    action = np.random.choice(n_actions, p=action_probs)
    reward = 1 if (state == action) else 0
    Q[state, action] += true_alpha * (reward - Q[state, action])
    data.append({'state': state, 'action': action, 'reward': reward})

# --- Fit model ---
def softmax(q_values, beta):
    exp_q = np.exp(beta * q_values - np.max(beta * q_values))
    return exp_q / np.sum(exp_q)

def neg_log_likelihood(params, data):
    alpha, beta = params
    if not (0 <= alpha <= 1 and beta > 0):
        return np.inf
    Q = np.zeros((n_states, n_actions))
    ll = 0
    for trial in data:
        s, a, r = trial['state'], trial['action'], trial['reward']
        probs = softmax(Q[s], beta)
        ll -= np.log(probs[a] + 1e-10)
        Q[s, a] += alpha * (r - Q[s, a])
    return ll

res = minimize(neg_log_likelihood, x0=[0.5, 1.0], bounds=[(0, 1), (0.01, 10)], args=(data,))
fit_alpha, fit_beta = res.x
print(f"Fitted alpha: {fit_alpha:.3f}, beta: {fit_beta:.3f}")

# --- Predict and evaluate ---
def predict(data, alpha, beta):
    Q = np.zeros((n_states, n_actions))
    preds = []
    for trial in data:
        s, a, r = trial['state'], trial['action'], trial['reward']
        probs = softmax(Q[s], beta)
        preds.append({'trial': trial, 'prob': probs[a], 'Q': Q.copy()})
        Q[s, a] += alpha * (r - Q[s, a])
    return preds

preds = predict(data, fit_alpha, fit_beta)
pred_probs = [p['prob'] for p in preds]
states = [p['trial']['state'] for p in preds]
actions = [p['trial']['action'] for p in preds]
rewards = [p['trial']['reward'] for p in preds]

# --- Plots ---
plt.figure(figsize=(12, 4))

# 1. Prediction quality (i.e., probability that the fitted model chooses the true action) over time
plt.subplot(1, 3, 1)
plt.plot(pred_probs, label='P(chosen action)', alpha=0.7)
plt.ylim((0,1))
plt.axhline(0.5, color='gray', linestyle='--', label='Chance')
plt.xlabel('Trial')
plt.ylabel('P(chosen)')
plt.title('Model performance')
plt.legend()

# 2. Smoothed running average
window = 20
smoothed = np.convolve(pred_probs, np.ones(window)/window, mode='valid')
plt.subplot(1, 3, 2)
plt.plot(smoothed)
plt.ylim((0,1))
plt.xlabel('Trial')
plt.ylabel('Smoothed P(chosen)')
plt.title(f'Smoothed (win={window})')

# 3. Final Q-values
final_Q = np.zeros((n_states, n_actions))
for trial in data:
    s, a, r = trial['state'], trial['action'], trial['reward']
    final_Q[s, a] += fit_alpha * (r - final_Q[s, a])

plt.subplot(1, 3, 3)
plt.imshow(final_Q, cmap='plasma', aspect='auto')
plt.colorbar(label='Q-value')
plt.xticks(range(n_actions))
plt.yticks(range(n_states))
plt.xlabel('Action')
plt.ylabel('State')
plt.title('Final Q-values')

plt.tight_layout()
plt.show()
