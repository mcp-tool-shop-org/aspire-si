---
title: How It Works
description: The four-stage ASPIRE pipeline — adversarial dialogue, critic training, what the student learns, and judging responses without the teacher.
sidebar:
  order: 2
---

ASPIRE trains AI judgment through a four-stage pipeline. Each stage builds on the previous one, ending with a critic that judges the student's responses without any teacher API calls.

## Stage 1: Adversarial Dialogue

The student generates a response. The teacher challenges it. They go back and forth, probing weaknesses, demanding clarity, pushing deeper.

```
Student: "Recursion works by calling itself."

Teacher (Socratic): "But what prevents infinite regress?
                     What's the mechanism that grounds the recursion?"

Student: "The base case stops it when..."

Teacher: "You say 'stops it' — but how does the computer know
          to check the base case before recursing?"
```

This adversarial exchange produces rich training data: not just right answers, but the reasoning process that leads to right answers. The teacher's challenges expose gaps in the student's understanding that flat supervision would never surface.

## Stage 2: Critic Training

The critic learns to predict the teacher's judgment of a response: its 0-10 score.

```python
# The critic reads the student's hidden states over the prompt and the response
# the teacher judged, and learns to predict the teacher's 0-10 score.
critic_loss = mse(critic(student_hidden_states(prompt, response)), teacher_score)
```

By default the critic is a lightweight head on the student's last hidden layer (`critic.architecture: head`); a separate model or a shared encoder are the other options. It reads the prompt together with the response the teacher scored, laid out in the student's own chat format, so after training it can score a new response without calling the teacher at all.

The critic also has a head for the teacher's reasoning. The trainer does not train it yet: it has no target for it, so today the critic learns the score.

## Stage 3: What the Student Learns

The critic is a head on the student's hidden states, and its loss is not stopped at them: it flows back into the student's LoRA adapter. So today the student learns **representations that help the critic predict the teacher's score**.

The student is not yet trained toward better answers. `aspire.losses` has reward, contrastive, trajectory and coherence terms, but the trainer does not feed them: the reward term gets a detached critic score and no log-probabilities, so it has no gradient, and the others get no inputs. Training the student on the critic's judgment, with a policy-gradient term through the reward loss, is planned for 1.3.0 ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

## Stage 4: Judgment without the teacher

After training, the critic scores a response from the student's hidden states, with no teacher call. `aspire.judge` loads a checkpoint and scores; a loop that revises until the critic is satisfied is yours to write around it. ASPIRE ships the trained critic and the judge; it does not ship the loop.

```python
from aspire.judge import Judge

judge = Judge.from_checkpoint("outputs/checkpoint-2")  # student, critic and config

def generate_with_judgment(prompt, threshold=7.0, attempts=3):
    response = student.generate(prompt)
    for _ in range(attempts):
        if judge.score(prompt, response) >= threshold:
            break
        response = student.generate(prompt)  # or a revision prompt of your own
    return response
```

`judge.score_many(prompts, responses)` scores a batch. For a model and critic you loaded yourself, `aspire.judge.critic_score(student, tokenizer, critic, prompt, response)` does the same. From the command line:

```bash
aspire judge outputs/checkpoint-2 --prompt "Why is the sky blue?" --response "Rayleigh scattering..."
aspire judge outputs/checkpoint-2 --pairs pairs.json --json
```

Everything here runs locally. The teacher's judgment has been distilled into the critic. The score is the critic's prediction of the teacher's score, and it is only as good as the run behind it: a critic trained on a few dozen prompts has seen a few dozen scores.

## Why this matters

Standard fine-tuning teaches models to match outputs. ASPIRE distills a teacher's judgment into a critic that runs with the student. The difference shows up at inference time: a fine-tuned model produces its best guess in one shot, while an ASPIRE run leaves you a critic that can judge the student's output by the teacher's criteria, so you can choose between attempts without the teacher.
