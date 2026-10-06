"""
Example: Using ASPIRE Code Teachers to critique code.

This example shows how to use multiple teacher personas
to get comprehensive code feedback.
"""

from aspire.integrations.code import (
    CodeTeacher,
    CodeSample,
)
from aspire.integrations.code.config import Language


def main():
    # Create a composite teacher
    teacher = CodeTeacher(
        personas=[
            "correctness_checker",
            "style_guide",
            "security_auditor",
            "performance_analyst",
        ],
        strategy="vote",  # Combine all perspectives
    )

    # Some code to critique
    code = '''
def process_user_data(user_input):
    # Process the input
    result = eval(user_input)  # Execute user code
    data = []
    for i in range(len(result)):
        if result[i] != None:
            data.append(result[i])
    return data

def calculate_average(numbers):
    sum = 0
    for n in numbers:
        sum = sum + n
    return sum / len(numbers)
'''

    # Create sample
    sample = CodeSample(
        code=code,
        language=Language.PYTHON,
        filename="example.py",
    )

    # Get critique
    print("=" * 60)
    print("ASPIRE Code Critique")
    print("=" * 60)
    print()

    critique = teacher.critique(sample)

    print(f"Overall Score: {critique.overall_score:.1f}/10")
    print()

    print("Dimension Scores:")
    for dim, score in critique.dimension_scores.items():
        print(f"  {dim.value}: {score:.1f}/10")
    print()

    print("Strengths:")
    for s in critique.strengths:
        print(f"  + {s}")
    print()

    print("Weaknesses:")
    for w in critique.weaknesses:
        print(f"  - {w}")
    print()

    print("Suggestions:")
    for s in critique.suggestions:
        print(f"  > {s}")
    print()

    if critique.line_comments:
        print("Line-specific feedback:")
        for line, comment in sorted(critique.line_comments.items()):
            print(f"  Line {line}: {comment}")

    print()
    print("Full Reasoning:")
    print(critique.reasoning[:500])


if __name__ == "__main__":
    main()
