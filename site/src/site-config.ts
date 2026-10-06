import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'aspire-ai',
  description: 'Adversarial Student-Professor Internalized Reasoning Engine — Teaching AI through internalized mentorship with cognitive empathy, syntropy, and perception.',
  logoBadge: 'AS',
  brandName: 'aspire-ai',
  repoUrl: 'https://github.com/mcp-tool-shop-org/aspire-ai',
  footerText: 'MIT Licensed — built by <a href="https://github.com/mcp-tool-shop-org" style="color:var(--color-muted);text-decoration:underline">mcp-tool-shop-org</a>',

  hero: {
    badge: 'AI training',
    headline: 'Teach AI judgment,',
    headlineAccent: 'not just knowledge.',
    description: 'An adversarial training framework where student models internalize a teacher\'s reasoning. After training, students self-refine using an internalized critic — no teacher API calls needed at inference.',
    primaryCta: { href: '#usage', label: 'Get started' },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      { label: 'Install', code: 'pip install aspire-ai' },
      { label: 'Dialogue', code: 'aspire dialogue "Explain recursion"\n  --teacher socratic --turns 3' },
      { label: 'Train', code: 'aspire train --config config.yaml\n  --teacher adversarial --epochs 3' },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'features',
      title: 'Features',
      subtitle: 'Internalized mentorship for AI models.',
      features: [
        { title: 'Adversarial dialogue', desc: 'Student generates, teacher challenges. Back and forth until the response is sharp, clear, and defensible.' },
        { title: 'Internalized critic', desc: 'The critic learns to predict the teacher\'s judgment — score and reasoning. After training, no teacher needed.' },
        { title: 'Pluggable teachers', desc: 'Socratic, Scientific, Creative, Adversarial, Compassionate — or compose multiple teachers into a committee.' },
      ],
    },
    {
      kind: 'code-cards',
      id: 'usage',
      title: 'Usage',
      cards: [
        {
          title: 'CLI',
          code: '# Generate adversarial dialogue\naspire dialogue "Your prompt" \\\n  --teacher socratic --turns 3\n\n# Train a model\naspire train --config config.yaml \\\n  --teacher adversarial --epochs 3\n\n# Evaluate checkpoint\naspire evaluate checkpoints/epoch-3 \\\n  --prompts data/eval.json',
        },
        {
          title: 'Python API',
          code: 'from aspire.teachers import CompositeTeacher\nfrom aspire.teachers import SocraticTeacher\nfrom aspire.teachers import ScientificTeacher\n\nteacher = CompositeTeacher(\n  teachers=[SocraticTeacher(),\n            ScientificTeacher()],\n  strategy="vote"\n)',
        },
      ],
    },
    {
      kind: 'data-table',
      id: 'teachers',
      title: 'Teacher Personas',
      subtitle: 'Different teachers produce different minds.',
      columns: ['Persona', 'Philosophy', 'Produces'],
      rows: [
        ['Socratic', '"What assumption are you making?"', 'Deep reasoning, independence'],
        ['Scientific', '"What\'s your evidence?"', 'Technical precision, rigor'],
        ['Creative', '"What if we tried the opposite?"', 'Innovation, lateral thinking'],
        ['Adversarial', '"I disagree. Defend your position."', 'Robust arguments, conviction'],
        ['Compassionate', '"How might someone feel about this?"', 'Ethical reasoning, wisdom'],
      ],
    },
    {
      kind: 'features',
      id: 'integrations',
      title: 'Integrations',
      subtitle: 'ASPIRE extends beyond text.',
      features: [
        { title: 'Stable Diffusion Forge', desc: 'Vision teachers critique generated images. Train LoRA adapters with CLIP-based critics for real-time guidance.' },
        { title: 'Isaac Gym / Isaac Lab', desc: 'Motion teachers for robotics — safety, efficiency, and grace. 512+ parallel GPU-accelerated environments.' },
        { title: 'Code assistants', desc: 'Correctness, style, and security teachers with static analysis integration (ruff, mypy, bandit).' },
      ],
    },
  ],
};
