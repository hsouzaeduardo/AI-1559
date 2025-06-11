import gym
import numpy as np
import matplotlib.pyplot as plt
import time

# Função para calcular a política ótima usando iteração de políticas
def one_step_lookahead(state, V, env, gamma):
    A = np.zeros(env.action_space.n)
    for action in range(env.action_space.n):
        for prob, next_state, reward, done in env.P[state][action]:
            A[action] += prob * (reward + gamma * V[next_state])
    return A

# Função de avaliação de política
def policy_evaluation(policy, env, gamma=1.0, theta=1e-8):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return V

# Função de melhoria de política
def policy_improvement(V, env, gamma=1.0):
    policy = np.zeros([env.observation_space.n, env.action_space.n])
    for s in range(env.observation_space.n):
        A = one_step_lookahead(s, V, env, gamma)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0
    return policy

# Função principal de iteração de políticas
def policy_iteration(env, gamma=1.0, theta=1e-8):
    policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n
    while True:
        V = policy_evaluation(policy, env, gamma, theta)
        new_policy = policy_improvement(V, env, gamma)
        if np.all(policy == new_policy):
            break
        policy = new_policy
    return policy

#função para renderizar a execução da política ou seja visualizar o ambiente 
# enquanto a política é executada
def render_policy_execution(env, policy, delay=0.5):
    obs, _ = env.reset()
    done = False

    plt.ion()  # Ativa modo interativo
    fig, ax = plt.subplots()

    while not done:
        img = env.render()
        ax.clear()
        ax.imshow(img)
        ax.axis("off")
        plt.pause(delay)

        action = np.argmax(policy[obs])
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    plt.ioff()
    plt.show()

#função para avaliar a política
def evaluate_policy(env, policy, n_episodes=100):
    total_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = np.argmax(policy[obs])
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        total_rewards.append((total_reward, steps))
    avg_reward = np.mean([r for r, _ in total_rewards])
    avg_steps = np.mean([s for _, s in total_rewards])
    return avg_reward, avg_steps

def render_ansi():
# Criação do ambiente com visualização em texto (ansi)
  env = gym.make("Taxi-v3", render_mode="ansi")

  # Avaliação com diferentes seeds
  seeds = [105, 64, 23, 133, 100]
  for seed in seeds:
      np.random.seed(seed)
      env.reset(seed=seed)
      print(f"\n🔁 Seed: {seed}")
      policy = policy_iteration(env, gamma=0.9)
      avg_reward, avg_steps = evaluate_policy(env, policy)
      print(f"✅ Média de recompensas: {avg_reward:.2f}")
      print(f"🚶 Média de passos por episódio: {avg_steps:.2f}")

def render_rgb_array():
# Criação do ambiente com visualização em RGB array
  # Criação do ambiente com renderização gráfica
  env = gym.make("Taxi-v3", render_mode="rgb_array")

  # Define a seed e treina a política
  np.random.seed(105)
  env.reset(seed=105)

  print("🎓 Treinando política com Policy Iteration...")
  policy = policy_iteration(env, gamma=0.9)
  print("✅ Política treinada!")

  # Visualização da execução da política aprendida
  print("🚕 Executando política com visualização...")
  render_policy_execution(env, policy)

#render_ansi()
render_rgb_array()