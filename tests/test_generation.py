"""Abstention is only a *metric* if its detector is exact: it must fire on the canonical line and
reject near-misses. Otherwise 'the model abstained on the no-answer cases' is a vibe, not a
measurement. This pins the detector the eval and UI both rely on.
"""
from agroteca.generate import ABSTAIN_PHRASE, is_abstention


def test_fires_on_the_canonical_line():
    assert is_abstention(ABSTAIN_PHRASE)
    assert is_abstention("  No encuentro la respuesta en el contexto disponible.  ")   # whitespace
    assert is_abstention("NO ENCUENTRO LA RESPUESTA EN EL CONTEXTO DISPONIBLE")         # case
    assert is_abstention("No encuentro la respuesta\nen el contexto disponible.")       # newline


def test_rejects_near_misses_and_real_answers():
    assert not is_abstention("No tengo la respuesta a esa pregunta.")                    # paraphrase
    assert not is_abstention("La caja de 6 pulgadas es suficiente (SFG.pdf, p. 34).")   # a real, cited answer
    assert not is_abstention("Encuentro la respuesta en el contexto.")                  # inverted meaning
    assert not is_abstention("")
