"""Finite-grid conditional false-ACK calibration for independent image trials."""
import math
import numpy as np
from scipy.stats import beta


def upper_binomial(failures, count, delta):
    if not 0 < delta < 1 or not 0 <= failures <= count:
        raise ValueError("invalid binomial bound inputs")
    if count==0 or failures==count:
        return 1.
    return float(beta.ppf(1-delta,failures+1,count-failures))


def calibrate_ack(image_ids, risk_scores, failures, thresholds, alpha=.05, delta=.05):
    """Score grid is frozen independently; lower scores mean safer to ACK.

    For each fixed threshold, accepted images give iid Bernoulli failures under
    an iid image/channel distribution. Bonferroni covers threshold selection.
    This is a high-probability bound on P(F|ACK), NOT a per-observation posterior
    guarantee. At most one preregistered channel draw per distinct image.
    """
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("correlated repeated image trials cannot inflate calibration n")
    score=np.asarray(risk_scores,dtype=float)
    failure=np.asarray(failures,dtype=bool)
    grid=list(thresholds)
    if len(score)!=len(image_ids) or score.shape!=failure.shape or not grid:
        raise ValueError("invalid calibration shapes or empty frozen grid")
    if not np.isfinite(score).all() or not np.isfinite(grid).all():
        raise ValueError("finite scores and thresholds required")
    candidates=[]
    for threshold in grid:
        accepted=score<=threshold
        n=int(accepted.sum())
        k=int(failure[accepted].sum())
        upper=upper_binomial(k,n,delta/len(grid))
        candidates.append(dict(threshold=float(threshold),accepted=n,failures=k,
                               upper_conditional_risk=upper,certified=upper<=alpha))
    valid=[c for c in candidates if c["certified"]]
    chosen=max(valid,key=lambda c:(c["accepted"],-c["threshold"])) if valid else None
    return dict(status="certified" if chosen else "abstain_no_certificate",selected=chosen,
                independent_images=len(image_ids),alpha=alpha,delta=delta,candidates=candidates)
