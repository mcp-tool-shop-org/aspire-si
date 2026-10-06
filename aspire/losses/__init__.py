"""
Loss functions for ASPIRE training.

Two types of losses:
1. Critic losses - train the critic to predict teacher judgment
2. Student losses - train the student using critic feedback
"""

from aspire.losses.combined import AspireLoss
from aspire.losses.critic import (
    CriticLoss,
    CriticReasoningLoss,
    CriticScoreLoss,
)
from aspire.losses.student import (
    CoherenceLoss,
    ContrastiveLoss,
    RewardLoss,
    StudentLoss,
    TrajectoryLoss,
)

__all__ = [
    # Critic losses
    "CriticLoss",
    "CriticScoreLoss",
    "CriticReasoningLoss",
    # Student losses
    "StudentLoss",
    "RewardLoss",
    "ContrastiveLoss",
    "TrajectoryLoss",
    "CoherenceLoss",
    # Combined
    "AspireLoss",
]
