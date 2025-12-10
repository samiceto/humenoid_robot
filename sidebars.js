// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    {
      type: 'category',
      label: 'Part I: Foundations & Nervous System (ROS 2)',
      items: [
        'part1/chapter1',
        'part1/chapter2',
        'part1/chapter3'
      ],
    },
    {
      type: 'category',
      label: 'Part II: Digital Twins & Simulation Mastery',
      items: [
        'part2/chapter4',
        'part2/chapter5',
        'part2/chapter6'
      ],
    },
    {
      type: 'category',
      label: 'Part III: Perception & Edge Brain',
      items: [
        'part3/chapter7',
        'part3/chapter8',
        'part3/chapter9'
      ],
    },
    {
      type: 'category',
      label: 'Part IV: Embodied Cognition & VLA Models',
      items: [
        'part4/chapter10',
        'part4/chapter11',
        'part4/chapter12'
      ],
    },
    {
      type: 'category',
      label: 'Part V: Bipedal Locomotion & Whole-Body Control',
      items: [
        'part5/chapter13',
        'part5/chapter14',
        'part5/chapter15'
      ],
    },
    {
      type: 'category',
      label: 'Part VI: Capstone Integration & Sim-to-Real Transfer',
      items: [
        'part6/chapter16',
        'part6/chapter17',
        'part6/chapter18'
      ],
    },
    {
      type: 'category',
      label: 'Appendices',
      items: [
        'search',
        'accessibility',
        'code-playgrounds',
        'diagram-viewers',
        'simulation-preview',
        'deployment',
        'quality-assurance',
        'factual-accuracy',
        'release-notes',
        'final-documentation',
        'appendix/hardware-specs'
      ],
    }
  ],
};

export default sidebars;