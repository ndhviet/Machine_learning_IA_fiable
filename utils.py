"""Figures et helpers communs aux CM et aux TP.

Toute figure réutilisable vit ici, jamais dupliquée dans un `.qmd` : la figure
projetée en amphi est *la même fonction* que celle appelée par les étudiants en
TP. C'est ce qui assure la continuité visuelle du cours.

Convention : chaque fonction `plot_*` construit et renvoie une `Figure`
matplotlib, sans jamais appeler `show()`.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Charte graphique (couleurs UEVE, reprises de _theme.scss)
# --------------------------------------------------------------------------

BLEU = "#41597a"
ORANGE = "#d96125"
GRIS = "#8a8a8a"
VERT = "#3f7d54"

CLASSES = {0: BLEU, 1: ORANGE}

DATA_DIR = Path(__file__).parent / "data"

# Les degrés élevés sont mal conditionnés — c'est précisément ce qu'on illustre.
warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")


def style():
    """Style matplotlib commun. À appeler une fois en tête de document."""
    plt.rcParams.update(
        {
            "figure.figsize": (7.5, 4.2),
            "figure.dpi": 130,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "legend.frameon": False,
        }
    )


# --------------------------------------------------------------------------
# Données
# --------------------------------------------------------------------------


def load_saheart(numeric_only: bool = False) -> pd.DataFrame:
    """Charge le jeu SAheart (maladie coronarienne, Western Cape).

    462 hommes, 9 variables explicatives, réponse binaire ``chd``.
    ``famhist`` (Present/Absent) est recodée en 0/1.
    """
    df = pd.read_csv(DATA_DIR / "SAheart.csv", index_col=0)
    df["famhist"] = (df["famhist"] == "Present").astype(int)
    if numeric_only:
        df = df.drop(columns=["famhist"])
    return df


def saheart_xy(variables=None):
    """Renvoie ``(X, y)`` en tableaux numpy, prêt pour scikit-learn."""
    df = load_saheart()
    y = df["chd"].to_numpy()
    X = df.drop(columns=["chd"])
    if variables is not None:
        X = X[list(variables)]
    return X.to_numpy(), y


# --------------------------------------------------------------------------
# Bloc A — règle de Bayes, risque de Bayes, coûts asymétriques
# --------------------------------------------------------------------------


def _gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def _croisement(x, a, b):
    """Abscisse où les courbes ``a`` et ``b`` se croisent.

    On ne peut pas prendre ``argmin|a - b||`` sur toute la grille : dans les
    queues, les deux densités valent zéro et leur écart y est minimal. On ne
    retient donc que les changements de signe situés là où au moins une des deux
    courbes est significative.
    """
    d = a - b
    significatif = np.maximum(a, b) > 0.01 * np.maximum(a, b).max()
    changement = np.where((np.sign(d[:-1]) != np.sign(d[1:])) & significatif[:-1])[0]
    if changement.size == 0:                      # pas de croisement exploitable
        return float(x[np.argmax(np.minimum(a, b))])
    i = changement[len(changement) // 2]          # le croisement central
    # interpolation linéaire entre les deux points encadrant le changement
    x0, x1, d0, d1 = x[i], x[i + 1], d[i], d[i + 1]
    return float(x0 - d0 * (x1 - x0) / (d1 - d0))


def plot_bayes_rule(mu0=-1.0, mu1=1.5, s0=1.0, s1=1.0, pi1=0.5, seuil=None):
    """Deux densités conditionnelles, la frontière de Bayes et le risque de Bayes.

    La zone hachurée est la probabilité d'erreur : elle ne peut pas être
    réduite, c'est le risque de Bayes.
    """
    x = np.linspace(-6, 7, 1200)
    f0 = (1 - pi1) * _gauss(x, mu0, s0)
    f1 = pi1 * _gauss(x, mu1, s1)

    if seuil is None:
        seuil = _croisement(x, f0, f1)

    fig, ax = plt.subplots()
    ax.plot(x, f0, color=BLEU, lw=2, label=r"$\pi_0 f_0(x)$  —  sain")
    ax.plot(x, f1, color=ORANGE, lw=2, label=r"$\pi_1 f_1(x)$  —  malade")

    droite = x >= seuil
    ax.fill_between(x[droite], 0, f0[droite], color=BLEU, alpha=0.30,
                    hatch="///", edgecolor=BLEU, lw=0)
    ax.fill_between(x[~droite], 0, f1[~droite], color=ORANGE, alpha=0.30,
                    hatch="\\\\\\", edgecolor=ORANGE, lw=0)

    ax.axvline(seuil, color="k", ls="--", lw=1.4)
    ax.annotate("frontière de Bayes", xy=(seuil, ax.get_ylim()[1] * 0.92),
                xytext=(seuil + 0.4, ax.get_ylim()[1] * 0.92),
                fontsize=10, va="center")
    ax.set_xlabel("$x$")
    ax.set_ylabel("densité pondérée")
    ax.set_yticks([])
    ax.legend(loc="upper left")
    ax.set_title("Zone hachurée = risque de Bayes (erreur incompressible)")
    fig.tight_layout()
    return fig


def plot_couts_asymetriques(mu0=-1.0, mu1=1.5, ratios=(1, 3, 10)):
    """Déplacement de la frontière optimale quand le faux négatif coûte plus cher.

    ``ratios`` = coût d'un faux négatif rapporté à celui d'un faux positif.
    """
    x = np.linspace(-6, 7, 1200)
    f0, f1 = _gauss(x, mu0, 1.0), _gauss(x, mu1, 1.0)

    fig, ax = plt.subplots()
    ax.plot(x, f0, color=BLEU, lw=2, label="sain")
    ax.plot(x, f1, color=ORANGE, lw=2, label="malade")

    couleurs = plt.cm.Greys(np.linspace(0.45, 0.9, len(ratios)))
    haut = ax.get_ylim()[1]
    for k, (r, c) in enumerate(zip(ratios, couleurs)):
        # on décide "malade" si  r * f1 > f0
        seuil = _croisement(x, f0, r * f1)
        ax.axvline(seuil, color=c, ls="--", lw=1.6)
        # étiquettes échelonnées en hauteur : sinon elles se chevauchent
        ax.annotate(f"$c = {r}$", xy=(seuil, haut * (0.95 - 0.11 * k)),
                    xytext=(-6, 0), textcoords="offset points",
                    color=c, fontsize=10, ha="right", va="center")

    ax.set_xlabel("$x$")
    ax.set_yticks([])
    ax.legend(loc="upper left")
    ax.set_title("Plus le faux négatif coûte cher, plus la frontière glisse à gauche")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Bloc B — sur-apprentissage, biais-variance, capacité
# --------------------------------------------------------------------------


def _vraie_fonction(x):
    return np.sin(2 * np.pi * x)


def _echantillon(n, sigma, rng):
    x = rng.uniform(0, 1, n)
    y = _vraie_fonction(x) + rng.normal(0, sigma, n)
    return x, y


def _ajuste(x, y, degre):
    """Ajustement polynomial numériquement stable.

    ``np.polyfit`` sur la base canonique est très mal conditionné dès le degré
    10 : la divergence numérique qui en résulte masque le phénomène statistique
    qu'on veut montrer. ``Polynomial.fit`` ramène le domaine sur [-1, 1], ce qui
    règle le problème. Renvoie une fonction évaluable.
    """
    return np.polynomial.Polynomial.fit(x, y, degre)


def plot_sur_apprentissage(degres=(1, 3, 15), n=25, sigma=0.25, graine=624):
    """Trois ajustements polynomiaux : sous-apprentissage, bon, sur-apprentissage."""
    rng = np.random.default_rng(graine)
    x, y = _echantillon(n, sigma, rng)
    grille = np.linspace(0, 1, 400)

    fig, axes = plt.subplots(1, len(degres), figsize=(11, 3.4), sharey=True)
    titres = {1: "sous-apprentissage", 3: "compromis", 15: "sur-apprentissage"}
    for ax, d in zip(axes, degres):
        modele = _ajuste(x, y, d)
        ax.plot(grille, _vraie_fonction(grille), color=GRIS, lw=1.5, ls="--",
                label="vraie fonction")
        ax.plot(grille, modele(grille), color=ORANGE, lw=2, label="ajustement")
        ax.scatter(x, y, s=22, color=BLEU, zorder=3, label="données")
        ax.set_ylim(-2, 2)
        ax.set_title(f"degré {d} — {titres.get(d, '')}")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_courbe_en_u(degres=range(1, 14), n=40, sigma=0.25, n_rep=200, graine=0):
    """La courbe en U : erreur d'apprentissage vs erreur de généralisation.

    Régénérée empiriquement (médiane sur ``n_rep`` échantillons) plutôt que
    dessinée à la main : les étudiants retracent exactement cette courbe en TP.
    """
    rng = np.random.default_rng(graine)
    degres = list(degres)
    err_tr = np.zeros((n_rep, len(degres)))
    err_te = np.zeros((n_rep, len(degres)))
    x_te = np.linspace(0, 1, 500)

    for r in range(n_rep):
        x, y = _echantillon(n, sigma, rng)
        y_te = _vraie_fonction(x_te) + rng.normal(0, sigma, x_te.size)
        for j, d in enumerate(degres):
            modele = _ajuste(x, y, d)
            err_tr[r, j] = np.mean((y - modele(x)) ** 2)
            err_te[r, j] = np.mean((y_te - modele(x_te)) ** 2)

    # Médiane et non moyenne : la distribution est à queue très lourde (de rares
    # ajustements de degré élevé sont catastrophiques et écrasent la moyenne).
    # La médiane décrit le comportement typique, qui est ce qu'on veut montrer.
    m_tr, m_te = np.median(err_tr, 0), np.median(err_te, 0)
    d_opt = degres[int(np.argmin(m_te))]

    fig, ax = plt.subplots()
    ax.plot(degres, m_tr, "o-", color=BLEU, lw=2, ms=4,
            label="erreur d'apprentissage")
    ax.plot(degres, m_te, "o-", color=ORANGE, lw=2, ms=4,
            label="erreur de généralisation")
    ax.axvline(d_opt, color="k", ls=":", lw=1.2)
    ax.text(d_opt + 0.2, ax.get_ylim()[1] * 0.75,
            f"complexité\noptimale ({d_opt})", fontsize=9)
    ax.set_xlabel("capacité du modèle (degré du polynôme)")
    ax.set_ylabel("erreur quadratique moyenne")
    ax.set_title("L'erreur d'apprentissage ne cesse de décroître. Pas l'autre.")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_biais_variance(degres=(1, 3, 15), n=25, sigma=0.25, n_rep=40, graine=1):
    """Plusieurs ajustements sur des échantillons différents.

    Rend visible la variance (dispersion des courbes) et le biais (écart
    systématique entre la moyenne des ajustements et la vraie fonction).
    """
    rng = np.random.default_rng(graine)
    grille = np.linspace(0, 1, 300)
    vrai = _vraie_fonction(grille)

    fig, axes = plt.subplots(1, len(degres), figsize=(11, 3.4), sharey=True)
    for ax, d in zip(axes, degres):
        preds = []
        for _ in range(n_rep):
            x, y = _echantillon(n, sigma, rng)
            preds.append(_ajuste(x, y, d)(grille))
        preds = np.array(preds)
        for p in preds:
            ax.plot(grille, p, color=ORANGE, alpha=0.13, lw=0.9)
        ax.plot(grille, preds.mean(0), color=ORANGE, lw=2.4,
                label="ajustement moyen")
        ax.plot(grille, vrai, color=GRIS, lw=1.8, ls="--", label="vraie fonction")

        biais2 = np.mean((preds.mean(0) - vrai) ** 2)
        var = np.mean(preds.var(0))
        ax.set_title(f"degré {d}\nbiais² = {biais2:.3g} · variance = {var:.3g}",
                     fontsize=10)
        ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    axes[-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def plot_decomposition_biais_variance(degres=range(1, 11), n=40, sigma=0.25,
                                      n_rep=300, graine=2):
    """Décomposition additive : biais² + variance + bruit = erreur totale."""
    rng = np.random.default_rng(graine)
    degres = list(degres)
    grille = np.linspace(0.05, 0.95, 200)
    vrai = _vraie_fonction(grille)

    biais2, variance = [], []
    for d in degres:
        preds = np.array([
            _ajuste(*_echantillon(n, sigma, rng), d)(grille)
            for _ in range(n_rep)
        ])
        biais2.append(np.mean((preds.mean(0) - vrai) ** 2))
        variance.append(np.mean(preds.var(0)))

    biais2, variance = np.array(biais2), np.array(variance)
    bruit = np.full_like(biais2, sigma ** 2)

    fig, ax = plt.subplots()
    ax.stackplot(degres, biais2, variance, bruit,
                 labels=["biais²", "variance", "bruit (incompressible)"],
                 colors=[BLEU, ORANGE, GRIS], alpha=0.75)
    ax.plot(degres, biais2 + variance + bruit, color="k", lw=1.6,
            label="erreur totale")
    ax.set_xlabel("capacité du modèle (degré du polynôme)")
    ax.set_ylabel("erreur quadratique")
    ax.set_ylim(0, (biais2 + variance + bruit).max() * 1.08)
    ax.set_title("Le biais décroît, la variance croît : le compromis est forcé")
    ax.legend(loc="upper center", ncol=2, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Bloc C — protocole, validation croisée, dimension
# --------------------------------------------------------------------------


def plot_decoupage_donnees():
    """Schéma : apprentissage / validation / test, et à quoi sert chaque part."""
    fig, ax = plt.subplots(figsize=(9, 2.4))
    parts = [("Apprentissage", 0.60, BLEU, "ajuster les paramètres"),
             ("Validation", 0.20, ORANGE, "choisir la complexité"),
             ("Test", 0.20, VERT, "estimer le risque")]
    gauche = 0.0
    for nom, largeur, couleur, role in parts:
        ax.barh(0, largeur, left=gauche, height=0.5, color=couleur, alpha=0.85)
        ax.text(gauche + largeur / 2, 0, nom, ha="center", va="center",
                color="white", fontweight="bold", fontsize=11)
        ax.text(gauche + largeur / 2, -0.45, role, ha="center", va="center",
                fontsize=9, color="#444")
        gauche += largeur
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, 0.5)
    ax.axis("off")
    ax.set_title("Trois rôles distincts — un seul jeu ne peut pas les tenir tous")
    fig.tight_layout()
    return fig


def plot_validation_croisee(K=5):
    """Schéma des K plis de la validation croisée."""
    fig, ax = plt.subplots(figsize=(8, 3.0))
    for k in range(K):
        for j in range(K):
            couleur = ORANGE if j == k else BLEU
            ax.barh(K - k - 1, 1, left=j, height=0.7, color=couleur,
                    alpha=0.85, edgecolor="white", lw=2)
        ax.text(-0.25, K - k - 1, f"pli {k + 1}", ha="right", va="center",
                fontsize=10)
    ax.set_xlim(-1.6, K)
    ax.set_ylim(-0.6, K - 0.2)
    ax.axis("off")
    ax.text(K / 2, -0.55, "orange = validation · bleu = apprentissage",
            ha="center", fontsize=9, color="#444")
    ax.set_title(f"Validation croisée à {K} plis : chaque observation sert une fois")
    fig.tight_layout()
    return fig


def plot_fleau_dimension(dims=(1, 2, 5, 10, 50, 200), n=500, graine=0):
    """Concentration des distances : en grande dimension, tout est équidistant."""
    rng = np.random.default_rng(graine)
    fig, ax = plt.subplots()
    couleurs = plt.cm.viridis(np.linspace(0.15, 0.9, len(dims)))
    for d, c in zip(dims, couleurs):
        X = rng.uniform(0, 1, (n, d))
        dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        dist = dist[np.triu_indices(n, 1)]
        dist = dist / dist.mean()          # normalisé pour comparer les dimensions
        ax.hist(dist, bins=60, density=True, histtype="step", lw=1.8,
                color=c, label=f"$d = {d}$")
    ax.set_xlabel("distance entre deux points (rapportée à la distance moyenne)")
    ax.set_ylabel("densité")
    ax.set_title("Quand $d$ croît, toutes les paires de points ont la même distance")
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    return fig


def plot_saheart_apercu(variables=("age", "ldl", "tobacco", "sbp")):
    """Aperçu du fil rouge : distributions conditionnelles par classe.

    Deux classes seulement (``chd`` vaut 0 ou 1). Tracé en contours et non en
    aplats : deux surfaces semi-opaques qui se recouvrent produisent une
    troisième teinte, que l'œil interprète à tort comme une troisième classe.
    """
    df = load_saheart()
    effectifs = df["chd"].value_counts()
    noms = {0: "sain (chd = 0)", 1: "malade (chd = 1)"}

    fig, axes = plt.subplots(1, len(variables), figsize=(11, 2.9))
    for ax, v in zip(axes, variables):
        bords = np.histogram_bin_edges(df[v], bins=20)
        for cls, couleur in CLASSES.items():
            serie = df.loc[df["chd"] == cls, v]
            ax.hist(serie, bins=bords, density=True, histtype="step",
                    color=couleur, lw=2.2,
                    label=f"{noms[cls]} — n = {effectifs[cls]}")
        ax.set_title(v)
        ax.set_yticks([])
    axes[-1].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("SAheart — 462 hommes, 9 variables, 2 classes", fontsize=11, y=1.04)
    fig.tight_layout()
    return fig


def plot_bayes_naif_1d(variable="age", graine=0):
    """Bayes naïf gaussien sur une variable de SAheart : densités estimées."""
    df = load_saheart()
    x = np.linspace(df[variable].min(), df[variable].max(), 400)

    fig, ax = plt.subplots()
    for cls, couleur in CLASSES.items():
        v = df.loc[df["chd"] == cls, variable]
        pi = len(v) / len(df)
        # histogramme pondéré par pi_k, pour être à la même échelle que pi_k f_k
        hauteurs, bords = np.histogram(v, bins=18, density=True)
        ax.bar(bords[:-1], hauteurs * pi, width=np.diff(bords), align="edge",
               alpha=0.25, color=couleur, linewidth=0)
        ax.plot(x, pi * _gauss(x, v.mean(), v.std()), color=couleur, lw=2.2,
                label=f"{'malade' if cls else 'sain'} — $\\hat\\pi_k \\hat f_k$")
    ax.set_xlabel(variable)
    ax.set_ylabel("densité pondérée")
    ax.set_yticks([])
    ax.legend()
    ax.set_title("Estimer $\\pi_k f_k(x)$ par classe, puis décider — c'est tout")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Séance 2 — frontières de décision
# --------------------------------------------------------------------------


def plot_decision_boundary(clf, X, y, noms=("x1", "x2"), ax=None, titre=None,
                           resolution=300, etiquettes=("sain", "malade")):
    """Frontière de décision d'un classifieur ajusté sur **deux** variables.

    C'est la fonction de continuité du cours : la même figure est projetée en
    CM et appelée par les étudiants en TP.

    Paramètres
    ----------
    clf : classifieur scikit-learn déjà ajusté (méthode ``predict``)
    X   : tableau (n, 2) — exactement les deux variables sur lesquelles ``clf``
          a été ajusté
    y   : étiquettes 0/1
    noms: noms des deux variables, pour les axes
    etiquettes : noms des classes 0 et 1, pour la légende
    """
    X = np.asarray(X)
    if X.shape[1] != 2:
        raise ValueError(
            f"X doit avoir exactement 2 colonnes, pas {X.shape[1]} : "
            "la frontière ne se trace que dans le plan. Ajustez le "
            "classifieur sur deux variables."
        )

    if ax is None:
        _, ax = plt.subplots()

    marge = 0.05
    x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
    x1_min, x1_max = X[:, 1].min(), X[:, 1].max()
    d0, d1 = (x0_max - x0_min) * marge, (x1_max - x1_min) * marge
    gx, gy = np.meshgrid(
        np.linspace(x0_min - d0, x0_max + d0, resolution),
        np.linspace(x1_min - d1, x1_max + d1, resolution),
    )
    Z = clf.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)

    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5],
                colors=[BLEU, ORANGE], alpha=0.16)
    ax.contour(gx, gy, Z, levels=[0.5], colors="k", linewidths=1.6)

    for cls, couleur in CLASSES.items():
        m = np.asarray(y) == cls
        ax.scatter(X[m, 0], X[m, 1], c=couleur, s=20, alpha=0.85,
                   edgecolors="white", linewidths=0.4,
                   label=etiquettes[cls])

    ax.set_xlabel(noms[0])
    ax.set_ylabel(noms[1])
    ax.legend(fontsize=9, loc="best")
    if titre:
        ax.set_title(titre)
    ax.grid(False)
    return ax.figure


def comparer_frontieres(modeles, X, y, noms=("x1", "x2"),
                        etiquettes=("sain", "malade")):
    """Frontières de plusieurs modèles côte à côte, sur les mêmes données.

    ``modeles`` : dict ``{"nom": classifieur non ajusté}``. Chaque modèle est
    ajusté sur ``X`` (deux colonnes) puis tracé.
    """
    fig, axes = plt.subplots(1, len(modeles), figsize=(4.6 * len(modeles), 4.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (nom, modele) in zip(axes, modeles.items()):
        modele.fit(X, y)
        err = 1 - modele.score(X, y)
        plot_decision_boundary(modele, X, y, noms=noms, ax=ax,
                               etiquettes=etiquettes,
                               titre=f"{nom} — erreur apprentissage {err:.1%}")
        ax.legend().set_visible(False)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Séance 2 — loi normale multivariée, LDA, QDA
# --------------------------------------------------------------------------


def _ellipse(mu, Sigma, n_sigma=1.0, points=200):
    """Points de l'ellipse d'isodensité à ``n_sigma`` écarts-types."""
    theta = np.linspace(0, 2 * np.pi, points)
    cercle = np.stack([np.cos(theta), np.sin(theta)])
    valeurs, vecteurs = np.linalg.eigh(Sigma)
    transfo = vecteurs @ np.diag(np.sqrt(valeurs))
    return (np.asarray(mu)[:, None] + n_sigma * transfo @ cercle)


def plot_gaussienne_bivariee():
    """Trois formes de matrice de covariance, trois géométries."""
    mu = np.zeros(2)
    cas = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "sphérique\n$\\Sigma = \\sigma^2 I$"),
        (np.array([[2.5, 0.0], [0.0, 0.5]]), "diagonale\n axes alignés"),
        (np.array([[1.6, 1.1], [1.1, 1.2]]), "quelconque\n axes tournés"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, (Sigma, titre) in zip(axes, cas):
        for k, alpha in zip((0.6, 1.2, 1.8, 2.4), (0.9, 0.65, 0.45, 0.3)):
            e = _ellipse(mu, Sigma, k)
            ax.plot(e[0], e[1], color=BLEU, lw=1.8, alpha=alpha)
        ax.plot(*mu, "o", color=ORANGE, ms=7, zorder=3)
        ax.set_title(titre, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
    fig.tight_layout()
    return fig


def plot_decomposition_covariance(Sigma=None):
    """Décomposition spectrale : les axes propres portent la géométrie."""
    if Sigma is None:
        Sigma = np.array([[1.6, 1.1], [1.1, 1.2]])
    mu = np.zeros(2)
    valeurs, vecteurs = np.linalg.eigh(Sigma)
    ordre = np.argsort(valeurs)[::-1]
    valeurs, vecteurs = valeurs[ordre], vecteurs[:, ordre]

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for k, alpha in zip((1.0, 2.0), (0.85, 0.4)):
        e = _ellipse(mu, Sigma, k)
        ax.plot(e[0], e[1], color=BLEU, lw=2, alpha=alpha)

    couleurs = (ORANGE, VERT)
    for i in range(2):
        v = vecteurs[:, i] * np.sqrt(valeurs[i])
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", lw=2.4,
                                    color=couleurs[i], shrinkA=0, shrinkB=0))
        ax.text(*(v * 1.18), f"$\\sqrt{{\\lambda_{i+1}}} = {np.sqrt(valeurs[i]):.2f}$",
                color=couleurs[i], fontsize=10, ha="center")

    ax.plot(0, 0, "ko", ms=5)
    ax.set_aspect("equal")
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_title("$\\Sigma = U \\Lambda U^\\top$ : $U$ oriente, $\\Lambda$ dilate")
    fig.tight_layout()
    return fig


def plot_mahalanobis(Sigma=None, rayon=2.4):
    """Deux points à même distance euclidienne, distances de Mahalanobis opposées.

    Les deux points sont placés sur les **axes propres** de Sigma : c'est là que
    l'écart entre les deux distances est maximal, donc le plus démonstratif.
    """
    if Sigma is None:
        Sigma = np.array([[2.2, 1.4], [1.4, 1.0]])
    mu = np.zeros(2)
    inv = np.linalg.inv(Sigma)
    r = rayon

    valeurs, vecteurs = np.linalg.eigh(Sigma)
    grand = vecteurs[:, int(np.argmax(valeurs))]      # axe le plus étalé
    petit = vecteurs[:, int(np.argmin(valeurs))]      # axe le plus resserré
    a = r * grand * np.sign(grand[0])                 # A : direction tolérante
    b = r * petit * np.sign(petit[1])                 # B : direction stricte

    def maha(x):
        return float(np.sqrt(x @ inv @ x))

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    for k, alpha in zip((1, 2, 3), (0.85, 0.5, 0.28)):
        e = _ellipse(mu, Sigma, k)
        ax.plot(e[0], e[1], color=BLEU, lw=1.8, alpha=alpha)
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(r * np.cos(theta), r * np.sin(theta), color=GRIS, ls="--", lw=1.4)

    for x, couleur, nom in ((a, ORANGE, "A"), (b, VERT, "B")):
        ax.plot(*x, "o", color=couleur, ms=10, zorder=3)
        ax.annotate("", xy=x, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-", lw=1.4, color=couleur))
        ax.text(x[0] * 1.12 + 0.1, x[1] * 1.12 + 0.1,
                f"{nom}\neuclid. {r:.1f}\nMahal. {maha(x):.1f}",
                color=couleur, fontsize=10, ha="center",
                va="bottom" if x[1] >= 0 else "top")

    ax.plot(0, 0, "ko", ms=5)
    ax.set_aspect("equal")
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_title("Cercle gris : distance euclidienne constante")
    fig.tight_layout()
    return fig


def plot_lda_vs_qda_modele():
    """Ce que chaque modèle suppose : covariance commune ou non."""
    mu0, mu1 = np.array([-1.0, -0.4]), np.array([1.4, 0.9])
    S_commune = np.array([[1.4, 0.8], [0.8, 1.0]])
    S0 = np.array([[1.9, 1.1], [1.1, 0.9]])
    S1 = np.array([[0.6, -0.35], [-0.35, 1.5]])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    cas = [
        (axes[0], (S_commune, S_commune),
         "LDA — une seule $\\Sigma$\nfrontière linéaire"),
        (axes[1], (S0, S1),
         "QDA — une $\\Sigma_k$ par classe\nfrontière quadratique"),
    ]  # (axe, (Sigma_0, Sigma_1), titre)
    for ax, Sigmas, titre in cas:
        for mu, Sigma, couleur in ((mu0, Sigmas[0], BLEU), (mu1, Sigmas[1], ORANGE)):
            for k, alpha in zip((1, 2), (0.9, 0.45)):
                e = _ellipse(mu, Sigma, k)
                ax.plot(e[0], e[1], color=couleur, lw=2, alpha=alpha)
            ax.plot(*mu, "o", color=couleur, ms=8, zorder=3)

        # frontière : log-rapport des densités gaussiennes, tracé par contour
        gx, gy = np.meshgrid(np.linspace(-5, 5, 400), np.linspace(-5, 5, 400))
        grille = np.stack([gx.ravel(), gy.ravel()], axis=1)

        def log_dens(x, mu, S):
            inv = np.linalg.inv(S)
            d = x - mu
            return -0.5 * np.einsum("ij,jk,ik->i", d, inv, d) \
                   - 0.5 * np.log(np.linalg.det(S))

        Z = (log_dens(grille, mu1, Sigmas[1])
             - log_dens(grille, mu0, Sigmas[0])).reshape(gx.shape)
        ax.contour(gx, gy, Z, levels=[0], colors="k", linewidths=2)
        ax.set_title(titre, fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4, 4)
    fig.tight_layout()
    return fig


def plot_complexite_parametrique(dims=range(2, 31)):
    """Nombre de paramètres de covariance selon les contraintes sur Sigma."""
    d = np.array(list(dims))
    familles = {
        "QDA — $\\Sigma_k$ pleine": 2 * d * (d + 1) / 2,
        "LDA — $\\Sigma$ commune": d * (d + 1) / 2,
        "Bayes naïf — $\\Sigma_k$ diagonale": 2 * d,
        "sphérique — $\\sigma_k^2 I$": np.full_like(d, 2, dtype=float),
    }
    couleurs = (ORANGE, BLEU, VERT, GRIS)

    fig, ax = plt.subplots()
    for (nom, n_param), couleur in zip(familles.items(), couleurs):
        ax.plot(d, n_param, lw=2.2, color=couleur, label=nom)
    ax.axhline(462, color="k", ls=":", lw=1.4)
    ax.text(2.4, 520, "462 observations (SAheart)", fontsize=9)
    ax.axvline(8, color="k", ls="--", lw=1)
    ax.text(8.3, 5, "$d = 8$", fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("nombre de variables $d$")
    ax.set_ylabel("paramètres de covariance à estimer")
    ax.set_title("Coût paramétrique de l'hypothèse gaussienne")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Séance 3 — estimation locale et k plus proches voisins
# --------------------------------------------------------------------------


def plot_parzen_1d(bandwidths=(0.10, 0.30, 0.80), n=80, graine=12):
    """Estimateurs de Parzen pour trois largeurs de bande sur le même échantillon."""
    from sklearn.neighbors import KernelDensity

    rng = np.random.default_rng(graine)
    composante = rng.binomial(1, 0.42, size=n)
    x = rng.normal(np.where(composante == 0, -1.25, 1.20),
                   np.where(composante == 0, 0.55, 0.35))
    grille = np.linspace(-3.5, 3.2, 600)
    vraie = (0.58 * _gauss(grille, -1.25, 0.55)
             + 0.42 * _gauss(grille, 1.20, 0.35))

    fig, axes = plt.subplots(1, len(bandwidths), figsize=(11, 3.5),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, h in zip(axes, bandwidths):
        kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(x[:, None])
        estimee = np.exp(kde.score_samples(grille[:, None]))
        ax.plot(grille, vraie, color=GRIS, lw=1.7, ls="--",
                label="densité vraie")
        ax.plot(grille, estimee, color=ORANGE, lw=2.3,
                label="densité estimée")
        ax.plot(x, np.full_like(x, -0.018), "|", color=BLEU, ms=8,
                alpha=0.75, clip_on=False)
        ax.set_title(f"$h = {h:.2f}$")
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("densité")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Même échantillon, trois niveaux de lissage", y=1.02)
    fig.tight_layout()
    return fig


def plot_voisinage_kppv(k=7, graine=8):
    """Vote local des k-PPV autour d'un nouveau point en dimension deux."""
    rng = np.random.default_rng(graine)
    x0 = rng.normal((-1.0, -0.25), (0.85, 0.70), size=(28, 2))
    x1 = rng.normal((1.05, 0.45), (0.80, 0.70), size=(22, 2))
    X = np.vstack([x0, x1])
    y = np.r_[np.zeros(len(x0), dtype=int), np.ones(len(x1), dtype=int)]
    requete = np.array([0.15, 0.10])
    distances = np.linalg.norm(X - requete, axis=1)
    voisins = np.argsort(distances)[:k]
    rayon = distances[voisins[-1]]
    vote = int(np.mean(y[voisins]) >= 0.5)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for classe, couleur in CLASSES.items():
        m = y == classe
        ax.scatter(X[m, 0], X[m, 1], c=couleur, s=38, alpha=0.75,
                   edgecolors="white", linewidths=0.5,
                   label=f"classe {classe}")
    ax.scatter(X[voisins, 0], X[voisins, 1], s=115, facecolors="none",
               edgecolors="black", linewidths=1.4, label=f"{k} voisins")
    cercle = plt.Circle(requete, rayon, fill=False, color="black", ls="--", lw=1.5)
    ax.add_patch(cercle)
    ax.scatter(*requete, marker="*", s=220, c=VERT, edgecolors="black",
               linewidths=0.7, zorder=5, label=f"nouveau point → classe {vote}")
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(f"Vote majoritaire dans un voisinage adaptatif ($k={k}$)")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def plot_frontieres_kppv(ks=(1, 11, 51), variables=("age", "tobacco")):
    """Frontières k-PPV sur deux variables standardisées de SAheart."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    X, y = saheart_xy(variables)
    X = StandardScaler().fit_transform(X)
    fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4.0),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    noms = tuple(f"{nom} standardisé" for nom in variables)
    for ax, k in zip(axes, ks):
        modele = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        erreur = 1 - modele.score(X, y)
        plot_decision_boundary(modele, X, y, noms=noms, ax=ax,
                               titre=f"$k={k}$ — erreur app. {erreur:.1%}",
                               resolution=220)
        ax.legend().set_visible(False)
    axes[0].legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return fig


def plot_courbe_kppv(k_values, train_errors, validation_errors, best_k=None):
    """Courbe en U d'un k-PPV à partir d'erreurs déjà calculées.

    La fonction ne réalise aucune validation : le CM et le TP lui transmettent
    leurs propres résultats, ce qui sépare le protocole statistique du tracé.
    """
    k_values = np.asarray(k_values)
    train_errors = np.asarray(train_errors)
    validation_errors = np.asarray(validation_errors)
    if best_k is None:
        best_k = int(k_values[np.argmin(validation_errors)])

    fig, ax = plt.subplots()
    ax.plot(k_values, train_errors, "o-", color=BLEU, lw=2, ms=3.5,
            label="erreur d'apprentissage")
    ax.plot(k_values, validation_errors, "o-", color=ORANGE, lw=2, ms=3.5,
            label="erreur de validation croisée")
    ax.axvline(best_k, color="k", ls=":", lw=1.3)
    ax.text(best_k + 1, ax.get_ylim()[1] - 0.02, f"$k^*={best_k}$", fontsize=9)
    ax.set_xlabel("nombre de voisins $k$")
    ax.set_ylabel("erreur équilibrée")
    ax.set_title("Peu de voisins : variance · beaucoup : biais")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_bruit_kppv(n_noise, scores, dispersions=None):
    """Performance du k-PPV selon le nombre de variables de bruit ajoutées."""
    n_noise = np.asarray(n_noise)
    scores = np.asarray(scores)
    if dispersions is None:
        dispersions = np.zeros_like(scores)
    dispersions = np.asarray(dispersions)

    fig, ax = plt.subplots()
    ax.errorbar(n_noise, scores, yerr=dispersions, fmt="o-", color=ORANGE,
                lw=2.2, ms=6, capsize=4)
    ax.axhline(0.5, color=GRIS, ls="--", lw=1.4, label="hasard équilibré")
    ax.set_xlabel("variables de bruit ajoutées")
    ax.set_ylabel("exactitude équilibrée en validation croisée")
    ax.set_ylim(0.45, min(1.0, max(0.68, (scores + dispersions).max() + 0.04)))
    ax.set_title("Variables inutiles : voisinages moins informatifs")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_concentration_distances(dims=(1, 2, 5, 10, 20, 50, 100),
                                 n=400, graine=4):
    """Dispersion relative des distances à un point selon la dimension."""
    rng = np.random.default_rng(graine)
    rapports = []
    coefficients = []
    for d in dims:
        X = rng.uniform(0, 1, size=(n, d))
        requete = rng.uniform(0, 1, size=d)
        distances = np.linalg.norm(X - requete, axis=1)
        rapports.append(distances.min() / distances.max())
        coefficients.append(distances.std() / distances.mean())

    fig, ax = plt.subplots()
    ax.plot(dims, rapports, "o-", color=ORANGE, lw=2.2,
            label="$d_{min}/d_{max}$")
    ax.plot(dims, coefficients, "o-", color=BLEU, lw=2.2,
            label="écart-type / moyenne")
    ax.set_xscale("log")
    ax.set_xticks(dims, labels=[str(d) for d in dims])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("dimension")
    ax.set_ylabel("rapport")
    ax.set_title("Grande dimension : le plus proche ressemble au plus lointain")
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Ouvrages de référence
# --------------------------------------------------------------------------

COUVERTURES = Path(__file__).parent / "lecture" / "images" / "couvertures"

OUVRAGES = [
    dict(fichier="esl.jpg",    legende="Hastie, Tibshirani & Friedman · Springer, 2009"),
    dict(fichier="isl.png",    legende="James, Witten, Hastie, Tibshirani & Taylor · Springer, 2023"),
    dict(fichier="bishop.jpg", legende="Bishop · Springer, 2006"),
    dict(fichier="murphy.jpg", legende="Murphy · MIT Press, 2022"),
]


def plot_bibliographie(ouvrages=None, hauteur=3.9):
    """Couvertures des ouvrages de référence.

    Images d'éditeur, sous droits : le dépôt du cours doit rester privé.
    Les couvertures sont alignées sur une hauteur commune ; leurs largeurs
    respectent les proportions d'origine, qui diffèrent d'un éditeur à l'autre.
    """
    import matplotlib.image as mpimg

    ouvrages = ouvrages or OUVRAGES
    images, ratios = [], []
    for o in ouvrages:
        chemin = COUVERTURES / o["fichier"]
        if not chemin.exists():
            raise FileNotFoundError(
                f"Couverture absente : {chemin}. Les images ne sont pas "
                "versionnées si le dépôt est public — les retélécharger."
            )
        im = mpimg.imread(chemin)
        images.append(im)
        ratios.append(im.shape[1] / im.shape[0])

    fig, axes = plt.subplots(
        1, len(images), figsize=(sum(ratios) * hauteur, hauteur),
        gridspec_kw={"width_ratios": ratios})
    # Pas de légende : les couvertures portent déjà titre, auteurs et éditeur.
    for ax, im in zip(np.atleast_1d(axes), images):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([])
        for bord in ax.spines.values():
            bord.set_edgecolor("#00000033")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.06)
    return fig


# --------------------------------------------------------------------------
# Séance 4 — arbres de décision (CART)
# --------------------------------------------------------------------------


def load_adult(numeric_only: bool = False) -> pd.DataFrame:
    """Charge Adult (Census Income 1994), sous-échantillon stratifié de 10 000.

    12 variables explicatives : 4 numériques, 8 catégorielles — dont ``sex`` et
    ``race``. Réponse binaire ``income`` (revenu > 50 000 $), 24 % de positifs.
    Les valeurs manquantes (``workclass``, ``occupation``, ``native-country``)
    sont conservées telles quelles.
    """
    df = pd.read_csv(DATA_DIR / "adult.csv")
    if numeric_only:
        df = df[[c for c in df.columns if df[c].dtype != object]]
    return df


def adult_xy(encodage="ordinal"):
    """Renvoie ``(X, y, noms)`` pour Adult, prêt pour scikit-learn.

    scikit-learn n'accepte pas les colonnes de texte : il faut encoder, même
    pour un arbre. Les manquants deviennent une modalité ``"Manquant"`` à part
    entière — leur absence est informative (``occupation`` manque surtout chez
    les personnes sans emploi).

    ``encodage="ordinal"`` attribue un entier arbitraire à chaque modalité.
    C'est sans conséquence pour un arbre, qui ne lit ces entiers que comme des
    étiquettes à séparer. C'est absurde pour tout modèle fondé sur une distance.
    """
    from sklearn.preprocessing import OrdinalEncoder

    df = load_adult()
    y = df["income"].to_numpy()
    X = df.drop(columns=["income"])
    categorielles = X.select_dtypes(object).columns.tolist()
    X[categorielles] = X[categorielles].fillna("Manquant")

    if encodage == "ordinal":
        X[categorielles] = OrdinalEncoder().fit_transform(X[categorielles])
    elif encodage == "onehot":
        X = pd.get_dummies(X, columns=categorielles, dtype=float)
    else:
        raise ValueError("encodage doit valoir 'ordinal' ou 'onehot'")

    return X.to_numpy(dtype=float), y, list(X.columns)


def plot_impuretes():
    """Gini, entropie et taux d'erreur, en classification binaire."""
    p = np.linspace(1e-6, 1 - 1e-6, 500)
    gini = 2 * p * (1 - p)
    entropie = -(p * np.log2(p) + (1 - p) * np.log2(1 - p)) / 2
    erreur = np.minimum(p, 1 - p)

    fig, ax = plt.subplots()
    ax.plot(p, entropie, color=ORANGE, lw=2.2, label="entropie (÷ 2)")
    ax.plot(p, gini, color=BLEU, lw=2.2, label="Gini")
    ax.plot(p, erreur, color=GRIS, lw=2.2, ls="--", label="taux d'erreur")
    ax.set_xlabel("proportion $p$ de la classe 1 dans le nœud")
    ax.set_ylabel("impureté")
    ax.legend()
    ax.set_title("Les deux premières sont dérivables. Pas la troisième.")
    fig.tight_layout()
    return fig


def _donnees_2d(n=300, graine=0):
    """Deux classes en damier partiel : séparables par découpes, pas par une droite."""
    rng = np.random.default_rng(graine)
    X = rng.uniform(0, 10, (n, 2))
    y = ((X[:, 0] > 5.5) ^ (X[:, 1] > 4.5)).astype(int)
    bruit = rng.random(n) < 0.08
    y[bruit] = 1 - y[bruit]
    return X, y


def plot_partition_recursive(n=300, graine=0):
    """Même arbre sous deux formes : partition et seuils de décision."""
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    rng = np.random.default_rng(graine)
    X = rng.uniform(0, 10, (n, 2))
    y = ((X[:, 0] > 5.5) & (X[:, 1] > 4.5)).astype(int)
    bruit = rng.random(n) < 0.06
    y[bruit] = 1 - y[bruit]

    arbre = DecisionTreeClassifier(
        max_leaf_nodes=3,
        min_samples_leaf=10,
        random_state=0,
    ).fit(X, y)

    fig, (ax_partition, ax_arbre) = plt.subplots(
        1, 2, figsize=(11.5, 4.4),
        gridspec_kw={"width_ratios": (1, 1.25)},
    )
    plot_decision_boundary(
        arbre,
        X,
        y,
        noms=("$x_1$", "$x_2$"),
        ax=ax_partition,
        titre="Partition de l'espace",
    )
    ax_partition.legend().set_visible(False)

    plot_tree(
        arbre,
        feature_names=["x₁", "x₂"],
        class_names=["classe 0", "classe 1"],
        filled=True,
        rounded=True,
        impurity=False,
        proportion=True,
        precision=2,
        label="none",
        fontsize=8,
        ax=ax_arbre,
    )
    for texte in ax_arbre.texts:
        contenu = texte.get_text()
        contenu = contenu.replace("<=", "≤")
        contenu = contenu.replace("True", "oui").replace("False", "non")
        texte.set_text(contenu)
    ax_arbre.set_title("Même modèle : seuils et feuilles")
    fig.tight_layout()
    return fig


def plot_sur_apprentissage_arbre(profondeurs=range(1, 21), n=300, graine=0):
    """Erreur d'apprentissage et de test selon la profondeur maximale."""
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    X, y = _donnees_2d(n * 4, graine)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5,
                                              random_state=graine, stratify=y)
    profondeurs = list(profondeurs)
    tr = [1 - DecisionTreeClassifier(max_depth=p, random_state=0)
          .fit(X_tr, y_tr).score(X_tr, y_tr) for p in profondeurs]
    te = [1 - DecisionTreeClassifier(max_depth=p, random_state=0)
          .fit(X_tr, y_tr).score(X_te, y_te) for p in profondeurs]

    fig, ax = plt.subplots()
    ax.plot(profondeurs, tr, "o-", color=BLEU, ms=4, lw=2,
            label="erreur d'apprentissage")
    ax.plot(profondeurs, te, "o-", color=ORANGE, ms=4, lw=2,
            label="erreur de test")
    ax.axvline(profondeurs[int(np.argmin(te))], color="k", ls=":", lw=1.2)
    ax.set_xlabel("profondeur maximale de l'arbre")
    ax.set_ylabel("taux d'erreur")
    ax.legend()
    ax.set_title("Sans contrainte, l'arbre atteint zéro erreur d'apprentissage")
    fig.tight_layout()
    return fig


def chemin_elagage(X, y, graine=0):
    """Suite emboîtée d'arbres élagués et leur alpha de coupure (Breiman).

    Renvoie ``(alphas, arbres)``. Chaque alpha est le seuil à partir duquel
    l'arbre suivant, plus petit, devient optimal pour le coût pénalisé.
    """
    from sklearn.tree import DecisionTreeClassifier

    complet = DecisionTreeClassifier(random_state=graine).fit(X, y)
    chemin = complet.cost_complexity_pruning_path(X, y)
    alphas = chemin.ccp_alphas[:-1]          # le dernier réduit l'arbre à sa racine
    arbres = [DecisionTreeClassifier(random_state=graine, ccp_alpha=a).fit(X, y)
              for a in alphas]
    return alphas, arbres


def plot_chemin_elagage(X=None, y=None, graine=0):
    """Coût-complexité : la taille de l'arbre décroît quand alpha croît.

    Figure centrale de la séance : c'est le risque empirique pénalisé de la
    séance 1, sous forme d'une suite **finie** d'arbres emboîtés.
    """
    if X is None:
        X, y = _donnees_2d(600, graine)
    alphas, arbres = chemin_elagage(X, y, graine)
    feuilles = [a.get_n_leaves() for a in arbres]
    err = [1 - a.score(X, y) for a in arbres]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.step(alphas, feuilles, where="post", color=BLEU, lw=2.2)
    a1.set_xlabel(r"$\alpha$")
    a1.set_ylabel("nombre de feuilles $|T|$")
    a1.set_title("La suite des arbres emboîtés est finie")

    a2.step(alphas, err, where="post", color=ORANGE, lw=2.2)
    a2.set_xlabel(r"$\alpha$")
    a2.set_ylabel("erreur d'apprentissage $R(T)$")
    a2.set_title(r"$R(T)$ croît quand on élague")
    for ax in (a1, a2):
        ax.set_xscale("symlog", linthresh=1e-4)
    fig.tight_layout()
    return fig


def plot_choix_alpha(X=None, y=None, K=5, graine=0):
    """Sélection de alpha par validation croisée : le minimum est intérieur."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.tree import DecisionTreeClassifier

    if X is None:
        X, y = _donnees_2d(600, graine)
    alphas, arbres = chemin_elagage(X, y, graine)
    # on échantillonne le chemin : une validation croisée par alpha coûte cher
    pas = max(1, len(alphas) // 25)
    alphas = alphas[::pas]
    cv = StratifiedKFold(n_splits=K, shuffle=True, random_state=graine)

    err_app, err_cv, ec = [], [], []
    for a in alphas:
        arbre = DecisionTreeClassifier(random_state=graine, ccp_alpha=a)
        err_app.append(1 - arbre.fit(X, y).score(X, y))
        s = cross_val_score(arbre, X, y, cv=cv)
        err_cv.append(1 - s.mean())
        ec.append(s.std())

    err_cv, ec = np.array(err_cv), np.array(ec)
    i = int(np.argmin(err_cv))

    fig, ax = plt.subplots()
    ax.plot(alphas, err_app, "o-", color=BLEU, ms=4, lw=2,
            label="erreur d'apprentissage")
    ax.errorbar(alphas, err_cv, yerr=ec, fmt="o-", color=ORANGE, ms=4, lw=2,
                capsize=3, label=f"erreur de validation croisée ({K} plis)")
    ax.axvline(alphas[i], color="k", ls="--", lw=1.4)
    ax.annotate(rf"$\hat\alpha = {alphas[i]:.4f}$", xy=(alphas[i], err_cv[i]),
                xytext=(12, 26), textcoords="offset points", fontsize=10)
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.set_xlabel(r"pénalité $\alpha$")
    ax.set_ylabel("taux d'erreur")
    ax.legend(loc="upper left")
    ax.set_title(r"$\alpha$ se choisit comme tout hyperparamètre")
    fig.tight_layout()
    return fig


def plot_arbre_adult(profondeur=3, graine=0):
    """Un arbre réel sur Adult, lisible : l'interprétabilité, montrée."""
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    X, y, noms = adult_xy()
    arbre = DecisionTreeClassifier(max_depth=profondeur,
                                   random_state=graine).fit(X, y)
    fig, ax = plt.subplots(figsize=(13, 5.4))
    plot_tree(arbre, feature_names=noms, class_names=["≤ 50k", "> 50k"],
              filled=True, impurity=False, proportion=True, rounded=True,
              fontsize=7, ax=ax)
    ax.set_title(f"Adult — arbre de profondeur {profondeur}, "
                 f"{arbre.get_n_leaves()} feuilles", fontsize=11)
    fig.tight_layout()
    return fig


def plot_instabilite(n_arbres=3, fraction=0.9, profondeur=3, graine=1):
    """Trois arbres sur trois sous-échantillons : la première découpe change.

    L'instabilité de CART est une variance élevée. La montrer vaut mieux que
    l'affirmer.
    """
    from sklearn.tree import DecisionTreeClassifier

    X, y = _donnees_2d(400, 0)
    rng = np.random.default_rng(graine)
    fig, axes = plt.subplots(1, n_arbres, figsize=(11, 3.6),
                             sharex=True, sharey=True)
    for k, ax in enumerate(np.atleast_1d(axes)):
        idx = rng.choice(len(X), int(fraction * len(X)), replace=False)
        arbre = DecisionTreeClassifier(max_depth=profondeur,
                                       random_state=0).fit(X[idx], y[idx])
        plot_decision_boundary(arbre, X[idx], y[idx], noms=("$x_1$", "$x_2$"),
                               ax=ax, etiquettes=("classe 0", "classe 1"),
                               titre=f"échantillon {k + 1} — "
                                     f"1ʳᵉ découpe : $x_{arbre.tree_.feature[0] + 1}"
                                     f" \\leq {arbre.tree_.threshold[0]:.1f}$")
        ax.legend().set_visible(False)
        ax.set_ylabel("")
    np.atleast_1d(axes)[0].set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Séance 5 — méthodes d'ensemble
# --------------------------------------------------------------------------


def plot_variance_moyenne(B=np.arange(1, 201), rhos=(0.0, 0.1, 0.3, 0.6)):
    """Variance d'une moyenne de B prédicteurs corrélés : rho fixe le plancher.

    Figure centrale de la séance. Var = rho*s2 + (1-rho)*s2/B : agréger davantage
    ne descend jamais sous rho*s2. Réduire la corrélation est le seul levier
    restant — c'est exactement ce que fait le tirage de variables des forêts.
    """
    B = np.asarray(B, dtype=float)
    fig, ax = plt.subplots()
    couleurs = plt.cm.viridis(np.linspace(0.12, 0.85, len(rhos)))
    for rho, c in zip(rhos, couleurs):
        var = rho + (1 - rho) / B
        ax.plot(B, var, lw=2.2, color=c, label=fr"$\rho = {rho}$")
        if rho > 0:
            ax.axhline(rho, color=c, ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("nombre de prédicteurs agrégés $B$")
    ax.set_ylabel(r"variance de la moyenne  (en unités de $\sigma^2$)")
    ax.set_ylim(0, 1.02)
    ax.legend(title="corrélation entre prédicteurs")
    ax.set_title(r"Le plancher vaut $\rho\sigma^2$, quel que soit $B$")
    fig.tight_layout()
    return fig


def plot_bagging_lissage(n=400, graine=0):
    """Un arbre, puis une agrégation : la frontière se lisse."""
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    X, y = _donnees_2d(n, graine)
    modeles = {
        "1 arbre": DecisionTreeClassifier(random_state=0),
        "bagging, 50 arbres": BaggingClassifier(
            DecisionTreeClassifier(random_state=0), n_estimators=50,
            random_state=0),
        "forêt aléatoire, 50 arbres": RandomForestClassifier(
            n_estimators=50, random_state=0),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True)
    for ax, (nom, m) in zip(axes, modeles.items()):
        m.fit(X, y)
        plot_decision_boundary(m, X, y, noms=("$x_1$", "$x_2$"), ax=ax,
                               etiquettes=("classe 0", "classe 1"), titre=nom)
        ax.legend().set_visible(False)
        ax.set_ylabel("")
    axes[0].set_ylabel("$x_2$")
    fig.tight_layout()
    return fig


def plot_oob(n_max=300, graine=0):
    """Erreur out-of-bag et erreur de test : la première est gratuite."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    X, y, _ = adult_xy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=graine, stratify=y)

    # En deçà d'une vingtaine d'arbres, trop d'observations ne sont hors-sac
    # pour aucun arbre : l'estimation OOB n'est pas définie partout.
    tailles = [25, 40, 70, 110, 160, 220, n_max]
    oob, test = [], []
    for b in tailles:
        f = RandomForestClassifier(n_estimators=b, oob_score=True,
                                   random_state=graine, n_jobs=-1).fit(X_tr, y_tr)
        oob.append(1 - f.oob_score_)
        test.append(1 - f.score(X_te, y_te))

    fig, ax = plt.subplots()
    ax.plot(tailles, oob, "o-", color=ORANGE, lw=2, ms=4,
            label="erreur out-of-bag")
    ax.plot(tailles, test, "o-", color=BLEU, lw=2, ms=4, label="erreur de test")
    ax.set_xlabel("nombre d'arbres $B$")
    ax.set_ylabel("taux d'erreur")
    ax.legend()
    ax.set_title("L'erreur OOB estime l'erreur de test, sans jeu mis de côté")
    fig.tight_layout()
    return fig


def plot_mtry(graine=0):
    """Effet du nombre de variables tirées à chaque nœud."""
    from sklearn.ensemble import RandomForestClassifier

    X, y, noms = adult_xy()
    d = X.shape[1]
    valeurs = list(range(1, d + 1))
    oob = [1 - RandomForestClassifier(n_estimators=120, max_features=m,
                                      oob_score=True, random_state=graine,
                                      n_jobs=-1).fit(X, y).oob_score_
           for m in valeurs]

    fig, ax = plt.subplots()
    ax.plot(valeurs, oob, "o-", color=ORANGE, lw=2, ms=5)
    racine = int(np.sqrt(d))
    ax.axvline(racine, color=BLEU, ls="--", lw=1.6)
    ax.text(racine + 0.15, max(oob), rf"$\sqrt{{d}} \approx {racine}$",
            color=BLEU, fontsize=10, va="top")
    ax.axvline(d, color=GRIS, ls=":", lw=1.4)
    ax.text(d - 0.15, max(oob), "bagging", color=GRIS, fontsize=10,
            ha="right", va="top")
    ax.set_xlabel("variables tirées à chaque nœud ($m$)")
    ax.set_ylabel("erreur out-of-bag")
    ax.set_title("Tirer moins de variables décorrèle les arbres")
    fig.tight_layout()
    return fig


def plot_boosting(n_max=250, graine=0):
    """Bagging et boosting : deux façons opposées d'agréger."""
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split

    X, y = _donnees_2d(1200, graine)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.4, random_state=graine, stratify=y)

    etapes = [1, 2, 5, 10, 25, 50, 100, 175, n_max]
    souche = DecisionTreeClassifier(max_depth=1)
    ada = [1 - AdaBoostClassifier(souche, n_estimators=b, random_state=graine)
           .fit(X_tr, y_tr).score(X_te, y_te) for b in etapes]
    foret = [1 - RandomForestClassifier(n_estimators=b, random_state=graine,
                                        n_jobs=-1).fit(X_tr, y_tr).score(X_te, y_te)
             for b in etapes]

    fig, ax = plt.subplots()
    ax.plot(etapes, foret, "o-", color=BLEU, lw=2, ms=4,
            label="forêt aléatoire (arbres profonds, en parallèle)")
    ax.plot(etapes, ada, "o-", color=ORANGE, lw=2, ms=4,
            label="AdaBoost (souches, en séquence)")
    ax.set_xscale("log")
    ax.set_xlabel("nombre de prédicteurs $B$")
    ax.set_ylabel("erreur de test")
    ax.legend(fontsize=9)
    ax.set_title("Partir de la variance, ou partir du biais")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Séance 6 — équité en apprentissage
# --------------------------------------------------------------------------


def metriques_equite(y_vrai, y_pred, groupe, noms_groupes=None):
    """Tableau des métriques d'équité, groupe par groupe.

    Colonnes :

    - ``taux de base``      $P(Y = 1 \\mid A = a)$ — propriété des données
    - ``taux de sélection`` $P(\\hat Y = 1 \\mid A = a)$ — parité démographique
    - ``TPR``               $P(\\hat Y = 1 \\mid Y = 1, A = a)$ — égalité des chances
    - ``FPR``               $P(\\hat Y = 1 \\mid Y = 0, A = a)$ — avec TPR : égalité des cotes
    - ``VPP``               $P(Y = 1 \\mid \\hat Y = 1, A = a)$ — parité prédictive
    """
    y_vrai, y_pred, groupe = map(np.asarray, (y_vrai, y_pred, groupe))
    lignes = {}
    for g in np.unique(groupe):
        m = groupe == g
        yv, yp = y_vrai[m], y_pred[m]
        pos, neg, pred_pos = yv == 1, yv == 0, yp == 1
        lignes[g if noms_groupes is None else noms_groupes[g]] = {
            "n": int(m.sum()),
            "taux de base": yv.mean(),
            "taux de sélection": yp.mean(),
            "TPR": yp[pos].mean() if pos.any() else np.nan,
            "FPR": yp[neg].mean() if neg.any() else np.nan,
            "VPP": yv[pred_pos].mean() if pred_pos.any() else np.nan,
        }
    return pd.DataFrame(lignes).T


def plot_equite_par_groupe(tableau, titre=None):
    """Les cinq taux, côte à côte par groupe."""
    mesures = ["taux de base", "taux de sélection", "TPR", "FPR", "VPP"]
    groupes = list(tableau.index)
    x = np.arange(len(mesures))
    largeur = 0.8 / len(groupes)

    fig, ax = plt.subplots(figsize=(9, 4))
    couleurs = (BLEU, ORANGE, VERT, GRIS)
    for k, g in enumerate(groupes):
        ax.bar(x + k * largeur - 0.4 + largeur / 2,
               [tableau.loc[g, m] for m in mesures],
               width=largeur, label=str(g), color=couleurs[k % len(couleurs)])
    for i, m in enumerate(mesures):
        vals = [tableau.loc[g, m] for g in groupes]
        ecart = max(vals) - min(vals)
        ax.text(i, max(vals) + 0.02, f"écart {ecart:.2f}", ha="center",
                fontsize=8.5, color="#444")
    ax.set_xticks(x); ax.set_xticklabels(mesures)
    ax.set_ylabel("proportion")
    ax.set_ylim(0, 1.0)
    ax.legend()
    if titre:
        ax.set_title(titre)
    fig.tight_layout()
    return fig


def _foret_adult(graine=42, n_arbres=300):
    """Forêt entraînée sur Adult, et le découpage associé. Mise en cache."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    if not hasattr(_foret_adult, "_cache"):
        _foret_adult._cache = {}
    if graine in _foret_adult._cache:
        return _foret_adult._cache[graine]

    X, y, noms = adult_xy()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=graine, stratify=y)
    foret = RandomForestClassifier(n_estimators=n_arbres, random_state=graine,
                                   n_jobs=-1).fit(X_tr, y_tr)
    res = (foret, X_tr, X_te, y_tr, y_te, noms)
    _foret_adult._cache[graine] = res
    return res


def plot_taux_base_adult(variable="sex", graine=42):
    """Les taux de base diffèrent selon le groupe — avant tout modèle."""
    df = load_adult()
    parts = df.groupby(variable)["income"].agg(["mean", "size"])
    parts = parts.sort_values("mean")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(parts.index.astype(str), parts["mean"], color=BLEU, height=0.6)
    for i, (nom, ligne) in enumerate(parts.iterrows()):
        ax.text(ligne["mean"] + 0.005, i,
                f"  {ligne['mean']:.1%}  (n = {int(ligne['size'])})",
                va="center", fontsize=9.5)
    ax.set_xlim(0, parts["mean"].max() * 1.45)
    ax.set_xlabel(r"$P(Y = 1 \mid A = a)$  —  taux de revenus > 50 k\$")
    ax.set_title("Une propriété des données, antérieure à tout modèle")
    fig.tight_layout()
    return fig


def plot_compromis_equite(graine=42, variable="sex"):
    """Performance contre parité : déplacer le seuil d'un groupe.

    On abaisse le seuil de décision du groupe défavorisé et l'on suit
    simultanément l'exactitude et l'écart de taux de sélection.
    """
    foret, X_tr, X_te, y_tr, y_te, noms = _foret_adult(graine)
    j = noms.index(variable)
    groupe = X_te[:, j]
    proba = foret.predict_proba(X_te)[:, 1]

    valeurs = np.unique(groupe)
    # le groupe défavorisé est celui dont le taux de sélection est le plus bas
    base = {g: (proba[groupe == g] > 0.5).mean() for g in valeurs}
    faible = min(base, key=base.get)

    seuils = np.linspace(0.5, 0.05, 40)
    exactitude, ecart = [], []
    for s in seuils:
        pred = np.where(groupe == faible, proba > s, proba > 0.5).astype(int)
        exactitude.append((pred == y_te).mean())
        taux = [pred[groupe == g].mean() for g in valeurs]
        ecart.append(abs(taux[0] - taux[1]))

    fig, ax = plt.subplots()
    ax.plot(ecart, exactitude, "-", color=BLEU, lw=2.2)
    ax.scatter(ecart[0], exactitude[0], color=ORANGE, s=70, zorder=3)
    ax.annotate("seuil 0,5 partout", xy=(ecart[0], exactitude[0]),
                xytext=(-12, 14), textcoords="offset points", fontsize=9.5,
                ha="right", color=ORANGE)
    i = int(np.argmin(ecart))
    ax.scatter(ecart[i], exactitude[i], color=VERT, s=70, zorder=3)
    ax.annotate("parité atteinte", xy=(ecart[i], exactitude[i]),
                xytext=(14, -14), textcoords="offset points", fontsize=9.5,
                color=VERT)
    ax.set_xlabel("écart de taux de sélection entre groupes")
    ax.set_ylabel("exactitude globale")
    ax.set_title("Chaque point est un choix. Aucun n'est neutre.")
    fig.tight_layout()
    return fig


def plot_impossibilite(graine=42, variable="sex", n_paquets=8):
    """Calibration par groupe, et ce qu'elle entraîne sur les taux d'erreur.

    Démonstration numérique du résultat de Chouldechova (2017) : dès que les
    taux de base diffèrent, un score bien calibré dans chaque groupe **ne peut
    pas** égaliser TPR et FPR.
    """
    foret, X_tr, X_te, y_tr, y_te, noms = _foret_adult(graine)
    j = noms.index(variable)
    groupe = X_te[:, j]
    proba = foret.predict_proba(X_te)[:, 1]
    valeurs = np.unique(groupe)
    etiquettes = {valeurs[0]: "groupe A", valeurs[1]: "groupe B"}

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))

    bords = np.linspace(0, 1, n_paquets + 1)
    for g, couleur in zip(valeurs, (BLEU, ORANGE)):
        m = groupe == g
        xs, ys = [], []
        for lo, hi in zip(bords[:-1], bords[1:]):
            sel = m & (proba >= lo) & (proba < hi)
            if sel.sum() >= 20:
                xs.append(proba[sel].mean()); ys.append(y_te[sel].mean())
        a1.plot(xs, ys, "o-", color=couleur, lw=2, ms=5, label=etiquettes[g])
    a1.plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1.4)
    a1.set_xlabel("score prédit"); a1.set_ylabel("fréquence observée")
    a1.set_title("Le score est calibré dans les deux groupes")
    a1.legend()

    pred = (proba > 0.5).astype(int)
    tab = metriques_equite(y_te, pred, groupe,
                           noms_groupes={g: etiquettes[g] for g in valeurs})
    mesures = ["taux de base", "TPR", "FPR"]
    x = np.arange(len(mesures))
    for k, (g, couleur) in enumerate(zip(tab.index, (BLEU, ORANGE))):
        a2.bar(x + k * 0.35 - 0.175, [tab.loc[g, m] for m in mesures],
               width=0.35, color=couleur, label=str(g))
    a2.set_xticks(x); a2.set_xticklabels(mesures)
    a2.set_ylabel("proportion")
    a2.set_title("Et pourtant TPR et FPR diffèrent")
    a2.legend()

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Séance 1, bloc C — données qui se suivent dans le temps
# --------------------------------------------------------------------------


def _serie_stations(n_stations=40, n_pas=900, graine=0, derive=2.0, rho=0.985):
    """Simule un parc de stations : cycle journalier, bruit autocorrélé, dérive.

    Reproduit la structure du projet Vélib\' : plusieurs stations observées aux
    mêmes instants, des relevés voisins presque identiques, et une lente dérive
    du service sur la période. Renvoie ``(X, y, temps)``, la cible étant l'état
    de tension 30 minutes plus tard.
    """
    rng = np.random.default_rng(graine)
    t = np.arange(n_pas)
    cycle = 4 * np.sin(2 * np.pi * t / 288)          # 288 pas de 5 min = 24 h
    tendance = derive * t / n_pas
    horizon = 6                                       # 6 pas = 30 min

    blocs = []
    for s in range(n_stations):
        niveau = rng.uniform(3, 12)
        bruit = np.zeros(n_pas)
        for i in range(1, n_pas):
            bruit[i] = rho * bruit[i - 1] + rng.normal(0, 0.7)
        velos = np.clip(niveau + cycle + bruit - tendance, 0, None)
        X = np.column_stack([
            velos[:-horizon],
            np.gradient(velos)[:-horizon],
            np.sin(2 * np.pi * t[:-horizon] / 288),
            np.cos(2 * np.pi * t[:-horizon] / 288),
            np.full(n_pas - horizon, s, dtype=float),
        ])
        blocs.append((X, (velos[horizon:] < 2).astype(int), t[:-horizon]))

    return (np.vstack([b[0] for b in blocs]),
            np.concatenate([b[1] for b in blocs]),
            np.concatenate([b[2] for b in blocs]))


def plot_decoupage_temporel(graine=0, K=5, part_test=0.30):
    """Découpage aléatoire contre découpage chronologique.

    À gauche : où tombe le jeu de test dans le temps. À droite : le même modèle,
    trois chiffres — dont la référence de persistance, qui prédit que rien ne
    change.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, KFold

    X, y, temps = _serie_stations(graine=graine)
    coupe = np.quantile(temps, 1 - part_test)
    tr, te = temps <= coupe, temps > coupe
    rng = np.random.default_rng(graine)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8),
                                 gridspec_kw={"width_ratios": [1.3, 1]})

    # --- une station, pour montrer où tombe le test
    une = X[:, 4] == 0
    serie, t_une = X[une, 0], temps[une]
    alea = rng.random(t_une.size) > (1 - part_test)
    a1.plot(t_une, serie, color=GRIS, lw=0.9)
    a1.scatter(t_une[alea], serie[alea], s=4, color=ORANGE, alpha=0.6)
    a1.axvspan(coupe, t_une.max(), color=BLEU, alpha=0.18)
    a1.text(coupe * 0.45, serie.max(), "test aléatoire", color=ORANGE,
            fontsize=10, va="top")
    a1.text((coupe + t_une.max()) / 2, serie.max(), "test\nchronologique",
            color=BLEU, fontsize=10, va="top", ha="center")
    a1.set_xlabel("temps (pas de 5 min)")
    a1.set_ylabel("vélos disponibles")
    a1.set_title("Le test aléatoire est entouré de ses voisins d'apprentissage")

    # --- les trois estimations
    modele = RandomForestClassifier(n_estimators=120, random_state=graine,
                                    n_jobs=-1)
    err_cv = 1 - cross_val_score(
        modele, X, y, cv=KFold(K, shuffle=True, random_state=graine)).mean()
    modele.fit(X[tr], y[tr])
    err_chrono = 1 - modele.score(X[te], y[te])
    err_persist = ((X[te, 0] < 2).astype(int) != y[te]).mean()

    noms = ["validation croisée\naléatoire", "découpage\nchronologique",
            "persistance"]
    valeurs = [err_cv, err_chrono, err_persist]
    a2.bar(noms, valeurs, color=[ORANGE, BLEU, GRIS], width=0.6)
    for i, v in enumerate(valeurs):
        a2.text(i, v + 0.0025, f"{v:.1%}", ha="center", fontsize=11,
                fontweight="600")
    a2.set_ylabel("taux d'erreur estimé")
    a2.set_ylim(0, max(valeurs) * 1.3)
    a2.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a2.set_title("Le même modèle, trois chiffres")
    a2.tick_params(axis="x", labelsize=9)

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Séance 1, bloc C — critères de performance en classification binaire
# --------------------------------------------------------------------------


def _scores_deux_gaussiennes(n=2000, mu0=-1.0, mu1=1.5, sigma=1.0, pi1=0.35,
                             graine=0):
    """Étiquettes et scores issus du modèle à deux gaussiennes du bloc A.

    Le score est la probabilité a posteriori exacte : il est donc calibré, et le
    seuil 0,5 redonne la règle MAP. ``pi1`` vaut 0,35, proche de la prévalence
    de `SAheart`.
    """
    rng = np.random.default_rng(graine)
    y = (rng.random(n) < pi1).astype(int)
    x = np.where(y == 1, rng.normal(mu1, sigma, n), rng.normal(mu0, sigma, n))
    f0, f1 = _gauss(x, mu0, sigma), _gauss(x, mu1, sigma)
    score = pi1 * f1 / (pi1 * f1 + (1 - pi1) * f0)
    return y, score


def _compte(y, score, seuil):
    """(VP, FP, FN, VN) pour un seuil donné."""
    pred = score >= seuil
    return (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
            int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum()))


def plot_matrice_confusion(seuil=0.5, graine=0):
    """Matrice de confusion, avec les quatre effectifs nommés."""
    y, score = _scores_deux_gaussiennes(graine=graine)
    vp, fp, fn, vn = _compte(y, score, seuil)

    cellules = [[(vp, "VP", ORANGE), (fp, "FP", GRIS)],
                [(fn, "FN", GRIS), (vn, "VN", BLEU)]]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for i in range(2):
        for j in range(2):
            n, nom, couleur = cellules[i][j]
            juste = nom in ("VP", "VN")
            ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, facecolor=couleur,
                                       alpha=0.55 if juste else 0.20,
                                       edgecolor="white", lw=3))
            ax.text(j + 0.5, 1.68 - i, nom, ha="center", va="center",
                    fontsize=13, fontweight="700",
                    color="white" if juste else "#444")
            ax.text(j + 0.5, 1.34 - i, f"{n}", ha="center", va="center",
                    fontsize=17, fontweight="600",
                    color="white" if juste else "#333")

    ax.set_xlim(-0.05, 2.05); ax.set_ylim(-0.05, 2.05)
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["réellement\nmalade", "réellement\nsain"], fontsize=10)
    ax.set_yticklabels(["prédit\nsain", "prédit\nmalade"], fontsize=10)
    ax.tick_params(length=0)
    for bord in ax.spines.values():
        bord.set_visible(False)
    ax.grid(False)
    ax.set_title(f"Seuil {seuil:.2f} — {len(y)} observations, "
                 f"{y.mean():.0%} de malades")
    fig.tight_layout()
    return fig


def plot_metriques_seuil(graine=0):
    """Précision, rappel, F1 et FDR en fonction du seuil de décision."""
    y, score = _scores_deux_gaussiennes(graine=graine)
    seuils = np.linspace(0.02, 0.98, 200)

    prec, rapp, f1, fdr = [], [], [], []
    for s in seuils:
        vp, fp, fn, vn = _compte(y, score, s)
        p = vp / (vp + fp) if vp + fp else np.nan
        r = vp / (vp + fn) if vp + fn else np.nan
        prec.append(p); rapp.append(r); fdr.append(1 - p if p == p else np.nan)
        f1.append(2 * p * r / (p + r) if (p == p and p + r > 0) else np.nan)

    fig, ax = plt.subplots()
    ax.plot(seuils, prec, color=BLEU, lw=2.2, label="précision")
    ax.plot(seuils, rapp, color=ORANGE, lw=2.2, label="rappel (TPR)")
    ax.plot(seuils, f1, color=VERT, lw=2.2, label="F1")
    ax.plot(seuils, fdr, color=GRIS, lw=1.8, ls="--", label="FDR = 1 − précision")
    i = int(np.nanargmax(f1))
    ax.axvline(seuils[i], color="k", ls=":", lw=1.2)
    # au-dessus des courbes : en bas, l'annotation chevauchait le FDR
    ax.annotate(f"F1 maximal au seuil {seuils[i]:.2f}",
                xy=(seuils[i], f1[i]), xytext=(0, 26),
                textcoords="offset points", ha="center", fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1, color="k"))
    ax.set_xlabel("seuil de décision")
    ax.set_ylabel("valeur")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower left", ncol=2)
    ax.set_title("Aucune de ces valeurs n'est une propriété du modèle seul")
    fig.tight_layout()
    return fig


def _roc(y, score):
    """(FPR, TPR, AUC) sans dépendance à scikit-learn."""
    ordre = np.argsort(-score)
    y = np.asarray(y)[ordre]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = np.concatenate([[0], tp / max(tp[-1], 1)])
    fpr = np.concatenate([[0], fp / max(fp[-1], 1)])
    # np.trapezoid n'existe qu'à partir de NumPy 2 ; np.trapz est déprécié après.
    aire = getattr(np, "trapezoid", None) or np.trapz
    return fpr, tpr, float(aire(tpr, fpr))


def plot_roc(graine=0, seuils_marques=(0.2, 0.5, 0.8)):
    """Courbe ROC : rappel contre taux de faux positifs, à seuil variable."""
    y, score = _scores_deux_gaussiennes(graine=graine)
    fpr, tpr, auc = _roc(y, score)

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot(fpr, tpr, color=ORANGE, lw=2.4, label=f"classifieur — AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1.6,
            label="tirage au hasard — AUC = 0,5")
    for s in seuils_marques:
        vp, fp, fn, vn = _compte(y, score, s)
        ax.plot(fp / (fp + vn), vp / (vp + fn), "o", color=BLEU, ms=8, zorder=3)
        ax.annotate(f"seuil {s}", xy=(fp / (fp + vn), vp / (vp + fn)),
                    xytext=(8, -12), textcoords="offset points", fontsize=9,
                    color=BLEU)
    ax.set_xlabel("FPR — taux de faux positifs")
    ax.set_ylabel("TPR — rappel")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Un point par seuil")
    fig.tight_layout()
    return fig


def _precision_rappel(y, score):
    """(rappel, précision, précision moyenne)."""
    ordre = np.argsort(-score)
    y = np.asarray(y)[ordre]
    tp = np.cumsum(y == 1)
    rang = np.arange(1, len(y) + 1)
    precision = tp / rang
    rappel = tp / max(tp[-1], 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rappel])) * precision))
    return rappel, precision, ap


def plot_precision_rappel(graine=0):
    """Courbe précision-rappel, avec la référence du classifieur aléatoire."""
    y, score = _scores_deux_gaussiennes(graine=graine)
    rappel, precision, ap = _precision_rappel(y, score)
    base = y.mean()

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    ax.plot(rappel, precision, color=ORANGE, lw=2.4,
            label=f"classifieur — AP = {ap:.3f}")
    ax.axhline(base, color=GRIS, ls="--", lw=1.6,
               label=f"tirage au hasard — {base:.2f}")
    ax.set_xlabel("rappel")
    ax.set_ylabel("précision")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("La référence n'est plus la diagonale, mais la prévalence")
    fig.tight_layout()
    return fig


def plot_roc_vs_pr(pi_rare=0.02, graine=0):
    """Effet du déséquilibre : la ROC bouge peu, la précision-rappel s'effondre."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for pi1, couleur, nom in ((0.35, ORANGE, "35 % de positifs"),
                              (pi_rare, BLEU, f"{pi_rare:.0%} de positifs")):
        y, score = _scores_deux_gaussiennes(n=20000, pi1=pi1, graine=graine)
        fpr, tpr, auc = _roc(y, score)
        axes[0].plot(fpr, tpr, color=couleur, lw=2.2, label=f"{nom} — AUC {auc:.3f}")
        rappel, precision, ap = _precision_rappel(y, score)
        axes[1].plot(rappel, precision, color=couleur, lw=2.2,
                     label=f"{nom} — AP {ap:.3f}")
        axes[1].axhline(y.mean(), color=couleur, ls=":", lw=1.2)

    axes[0].plot([0, 1], [0, 1], color=GRIS, ls="--", lw=1.4)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC — presque insensible")
    axes[1].set_xlabel("rappel"); axes[1].set_ylabel("précision")
    axes[1].set_title("Précision-rappel — très sensible")
    for a in axes:
        a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.02)
        a.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout()
    return fig


def plot_gaussienne_3d(rho=0.85, sigma1=1.0, sigma2=1.0, n=140,
                       transparent=True, elev=28, azim=-60):
    """Surface d'une gaussienne bivariée corrélée, avec ses isodensités au sol.

    Illustration de la page de titre de la séance 2. ``rho`` fixe la corrélation :
    la crête de la surface et les ellipses projetées s'inclinent avec elle.
    """
    from matplotlib import cm

    Sigma = np.array([[sigma1 ** 2, rho * sigma1 * sigma2],
                      [rho * sigma1 * sigma2, sigma2 ** 2]])
    inv, det = np.linalg.inv(Sigma), np.linalg.det(Sigma)

    lim = 3.2
    g = np.linspace(-lim, lim, n)
    X1, X2 = np.meshgrid(g, g)
    ecart = np.stack([X1.ravel(), X2.ravel()], axis=1)
    quad = np.einsum("ij,jk,ik->i", ecart, inv, ecart)
    Z = (np.exp(-0.5 * quad) / (2 * np.pi * np.sqrt(det))).reshape(X1.shape)

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    # Le plancher est descendu sous la surface : sinon les isodensités
    # disparaissent sous la queue de la gaussienne et la corrélation ne se voit pas.
    sol = -Z.max() * 0.62

    ax.plot_surface(X1, X2, Z, cmap=cm.YlOrBr_r, rstride=2, cstride=2,
                    linewidth=0.15, edgecolor=(1, 1, 1, 0.25),
                    antialiased=True, alpha=0.92)
    ax.contour(X1, X2, Z, levels=8, zdir="z", offset=sol,
               cmap=cm.YlOrBr_r, linewidths=2.0)

    ax.set_zlim(sol, Z.max() * 1.05)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("$x_1$", labelpad=-8, fontsize=10)
    ax.set_ylabel("$x_2$", labelpad=-8, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect((1, 1, 0.85))

    # panneaux transparents : la figure doit se fondre dans le fond de la slide
    for panneau in (ax.xaxis, ax.yaxis, ax.zaxis):
        panneau.pane.set_alpha(0.0)
        panneau.line.set_color((0, 0, 0, 0.25))
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)

    fig.tight_layout()
    return fig


def plot_classification_2d(n=380, bruit=0.26, k=35, graine=3, transparent=True,
                           regions=False):
    """Classification supervisée en 2D, frontière non linéaire.

    Illustration de la page de titre de la séance 1. Deux classes entrelacées
    qu'aucune droite ne sépare ; la frontière est celle des $k$ plus proches
    voisins, tracée au niveau 0,5 de la probabilité estimée. Un k élevé donne une
    frontière plus lisse — c'est une propriété du modèle, pas un lissage graphique.

    ``regions=False`` supprime les aplats de décision : sur une page de titre,
    ils forment un rectangle plein qui annule le fond transparent.
    """
    from sklearn.datasets import make_moons
    from sklearn.neighbors import KNeighborsClassifier

    X, y = make_moons(n_samples=n, noise=bruit, random_state=graine)
    modele = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(X, y)

    marge = 0.45
    gx, gy = np.meshgrid(
        np.linspace(X[:, 0].min() - marge, X[:, 0].max() + marge, 400),
        np.linspace(X[:, 1].min() - marge, X[:, 1].max() + marge, 400))
    proba = modele.predict_proba(np.c_[gx.ravel(), gy.ravel()])[:, 1]
    proba = proba.reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    if regions:
        ax.contourf(gx, gy, proba, levels=[-0.01, 0.5, 1.01],
                    colors=[BLEU, ORANGE], alpha=0.16)
    ax.contour(gx, gy, proba, levels=[0.5], colors="#222222", linewidths=2.8)

    for cls, couleur in CLASSES.items():
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], c=couleur, s=42, alpha=0.92,
                   edgecolors="white", linewidths=1.1, zorder=3)

    ax.set_xticks([]); ax.set_yticks([])
    ax.grid(False)
    for bord in ax.spines.values():
        bord.set_visible(False)
    if transparent:
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
    fig.tight_layout(pad=0.2)
    return fig
