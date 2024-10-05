"""
Partie 2 - Programmation dynamique
==================================

Rappel: la programmation dynamique est une technique algorithmique qui
permet de résoudre des problèmes en les décomposant en sous-problèmes
plus petits, et en mémorisant les solutions de ces sous-problèmes pour
éviter de les recalculer plusieurs fois.
"""

# Exercice 1: Fibonacci
# ----------------------
# La suite de Fibonacci est définie par:
#   F(0) = 0
#   F(1) = 1
#   F(n) = F(n-1) + F(n-2) pour n >= 2
#
# Ecrire une fonction qui calcule F(n) pour un n donné.
# Indice: la fonction doit être récursive.


def fibonacci(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci.
    """
    # BEGIN SOLUTION
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)
    # END SOLUTION


# Exercice 2: Fibonacci avec mémorisation
# ---------------------------------------
# Ecrire une fonction qui calcule F(n) pour un n donné, en mémorisant
# les résultats intermédiaires pour éviter de les recalculer plusieurs
# fois.
# Indice: la fonction doit être récursive.


def fibonacci_memo(n: int) -> int:
    """
    Calcule le n-ième terme de la suite de Fibonacci, en mémorisant les
    résultats intermédiaires.
    """

    # BEGIN SOLUTION
    # Initialiser le dictionnaire de mémorisation
    memo = {}

    # Appel de la fonction auxiliaire pour calculer Fibonacci avec mémorisation
    def fibonacci_memo_helper(n: int, memo: dict) -> int:
        if n in memo:  # Si déjà calculé, retourner la valeur mémorisée
            return memo[n]
        if n == 0:  # Cas de base pour n=0
            return 0
        elif n == 1:  # Cas de base pour n=1
            return 1
        else:
            # Calculer et mémoriser la valeur de Fibonacci pour n
            memo[n] = fibonacci_memo_helper(n - 1, memo) + fibonacci_memo_helper(
                n - 2, memo
            )
            return memo[n]

    # Appeler la fonction auxiliaire avec le dictionnaire vide
    return fibonacci_memo_helper(n, memo)

    # END SOLUTION
