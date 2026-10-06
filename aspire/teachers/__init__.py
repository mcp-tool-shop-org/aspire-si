"""
Teacher models for ASPIRE training.

Teachers provide wisdom, judgment, and adversarial challenge to the student.
Different teachers produce different learning outcomes - a Socratic philosopher
teaches differently than a rigorous scientist or a creative artist.
"""

from aspire.teachers.base import (
    BaseTeacher,
    DialogueTurn,
    TeacherChallenge,
    TeacherEvaluation,
)
from aspire.teachers.claude import ClaudeTeacher
from aspire.teachers.composite import CompositeTeacher
from aspire.teachers.local import LocalTeacher
from aspire.teachers.openai import OpenAITeacher
from aspire.teachers.personas import (
    AdversarialTeacher,
    CompassionateTeacher,
    CreativeTeacher,
    ScientificTeacher,
    SocraticTeacher,
)
from aspire.teachers.registry import TeacherRegistry, get_teacher, register_teacher

__all__ = [
    # Base
    "BaseTeacher",
    "TeacherEvaluation",
    "TeacherChallenge",
    "DialogueTurn",
    # Implementations
    "ClaudeTeacher",
    "OpenAITeacher",
    "LocalTeacher",
    "CompositeTeacher",
    # Personas
    "SocraticTeacher",
    "ScientificTeacher",
    "CreativeTeacher",
    "AdversarialTeacher",
    "CompassionateTeacher",
    # Registry
    "TeacherRegistry",
    "get_teacher",
    "register_teacher",
]
