import numpy as np

from dynamic_programming.grid_world_env import GridWorldEnv
from dynamic_programming.mdp import MDP
from dynamic_programming.stochastic_grid_word_env import StochasticGridWorldEnv

# Exercice 2: Résolution du MDP
# -----------------------------
# Ecrire une fonction qui calcule la valeur de chaque état du MDP, en
# utilisant la programmation dynamique.
# L'algorithme de programmation dynamique est le suivant:
#   - Initialiser la valeur de chaque état à 0
#   - Tant que la valeur de chaque état n'a pas convergé:
#       - Pour chaque état:
#           - Estimer la fonction de valeur de chaque état
#           - Choisir l'action qui maximise la valeur
#           - Mettre à jour la valeur de l'état
#
# Indice: la fonction doit être itérative.


def mdp_value_iteration(mdp: MDP, max_iter: int = 1000, gamma=1.0) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration":
    https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration
    """
    values = np.zeros(mdp.observation_space.n)
    # BEGIN SOLUTION
    for i in range(max_iter):
        prev_values = values.copy()
        for state in range(mdp.observation_space.n):
            new_value = float("-inf")
            for action in range(mdp.action_space.n):
                next_state, reward, done = mdp.P[state][action]
                value = reward + gamma * prev_values[next_state]
                new_value = max(new_value, value)
            values[state] = new_value
    # END SOLUTION
    return values


def grid_world_value_iteration(
    env: GridWorldEnv,
    max_iter: int = 1000,
    gamma=1.0,
    theta=1e-5,
) -> np.ndarray:
    """
    Estimation de la fonction de valeur grâce à l'algorithme "value iteration".
    theta est le seuil de convergence (différence maximale entre deux itérations).
    """
    values = np.zeros((4, 4))

    # BEGIN SOLUTION
    for iteration in range(max_iter):
        delta = 0  # Pour suivre la convergence
        new_values = np.copy(values)

        for row in range(env.height):
            for col in range(env.width):
                cell = env.grid[row, col]

                # Ignorer les états terminaux et les murs
                if cell in {"P", "N", "W"}:
                    continue

                # Définir la position actuelle de l'agent
                env.set_state(row, col)

                # Calculer la valeur maximale parmi toutes les actions possibles
                max_value = float("-inf")
                for action in range(env.action_space.n):
                    next_state, reward, done, _ = env.step(action, make_move=False)
                    next_row, next_col = next_state
                    value = reward + gamma * values[next_row, next_col]
                    if value > max_value:
                        max_value = value

                # Mettre à jour la valeur de l'état
                new_values[row, col] = max_value

                # Mettre à jour la différence maximale
                delta = max(delta, abs(new_values[row, col] - values[row, col]))

        # Mettre à jour les valeurs pour la prochaine itération
        values = new_values

        # Vérifier la convergence
        if delta < theta:
            print(f"Convergence atteinte après {iteration + 1} itérations.")
            break
    # END SOLUTION
    return values


def value_iteration_per_state(env, values, gamma, prev_val, delta):
    row, col = env.current_position
    values[row, col] = float("-inf")
    for action in range(env.action_space.n):
        next_states = env.get_next_states(action=action)
        current_sum = 0
        for next_state, reward, probability, _, _ in next_states:
            # print((row, col), next_state, reward, probability)
            next_row, next_col = next_state
            current_sum += (
                probability
                * env.moving_prob[row, col, action]
                * (reward + gamma * prev_val[next_row, next_col])
            )
        values[row, col] = max(values[row, col], current_sum)
    delta = max(delta, np.abs(values[row, col] - prev_val[row, col]))
    return delta


def stochastic_grid_world_value_iteration(
    env: StochasticGridWorldEnv,
    max_iter: int = 1000,
    gamma: float = 1.0,
    theta: float = 1e-5,
) -> np.ndarray:
    values = np.zeros((4, 4))
    # BEGIN SOLUTION
    for iteration in range(max_iter):
        delta = 0  # Pour suivre la convergence
        new_values = np.copy(values)

        for row in range(env.height):
            for col in range(env.width):
                cell = env.grid[row, col]

                # Ignorer les états terminaux et les murs
                if cell in {"P", "N", "W"}:
                    continue

                # Définir la position actuelle de l'agent
                env.set_state(row, col)

                # Calculer la valeur maximale parmi toutes les actions possibles
                max_value = float("-inf")
                for action in range(env.action_space.n):
                    expected_value = 0.0
                    next_states = env.get_next_states(action)
                    for next_state, reward, prob, done, actual_action in next_states:
                        next_row, next_col = next_state
                        expected_value += prob * (
                            reward + gamma * values[next_row, next_col]
                        )
                    if expected_value > max_value:
                        max_value = expected_value

                # Mettre à jour la valeur de l'état
                new_values[row, col] = max_value

                # Mettre à jour la différence maximale
                delta = max(delta, abs(new_values[row, col] - values[row, col]))

        # Mettre à jour les valeurs pour la prochaine itération
        values = new_values

        # Vérifier la convergence
        if delta < theta:
            print(f"Convergence atteinte après {iteration + 1} itérations.")
            break

    return values
