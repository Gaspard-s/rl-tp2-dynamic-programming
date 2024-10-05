# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    a = 0
    # BEGIN SOLUTION
    if n == 0:
        return 1  # Il y a une seule façon de paver un rectangle 3x0 (aucun pavage)
    if n == 1:
        return 0  # Il est impossible de paver un rectangle 3x1 avec des dominos 2x1
    if n == 2:
        return 3  # Il y a 3 façons de paver un rectangle 3x2 (2 dominos horizontaux, ou 2 configurations différentes de dominos verticaux)

    # Table pour stocker les solutions intermédiaires
    dp = [0] * (n + 1)

    # Initialiser les valeurs de base
    dp[0] = 1
    dp[1] = 0
    dp[2] = 3

    # Calculer les solutions pour les dimensions plus grandes
    for i in range(3, n + 1):
        dp[i] = 4 * dp[i - 2] - dp[i - 4] if i >= 4 else 4 * dp[i - 2]

    return dp[n]
    # END SOLUTION
