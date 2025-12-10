// @ts-check
// `@type` JSDoc annotations allow IDEs and type-checking tools to autocomplete
// and validate function arguments and return types.
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'From Simulated Brains to Walking Bodies',
  favicon: 'img/favicon.ico',

  // Set the production url of your site here
  url: 'https://samiceto.github.io',
  // Set the /<base>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/humenoid_robot/', // GitHub Pages serves from /<repository-name>/

  // GitHub pages deployment config.
  organizationName: 'samiceto',
  projectName: 'humenoid_robot', // Your repository name
  deploymentBranch: 'main',
  // trailingSlash: false,

  // GitHub Pages configuration
  trailingSlash: undefined, // Use undefined for auto-detection
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/samiceto/humenoid_robot/tree/main/',
          routeBasePath: '/', // Serve docs at root
          exclude: [], // Include all docs
          // Configure the docs plugin
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/samiceto/humenoid_robot/tree/main/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  plugins: [
    // Removed duplicate docs plugin - already configured in preset above
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      navbar: {
        title: 'Physical AI & Humanoid Robotics',
        logo: {
          alt: 'Robotics Book Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'tutorialSidebar',
            position: 'left',
            label: 'Book',
          },
          {to: '/blog', label: 'Updates', position: 'left'},
          {
            to: '/accessibility',
            label: 'Accessibility',
            position: 'right',
          },
          {
            href: 'https://github.com/samiceto/humenoid_robot',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Content',
            items: [
              {
                label: 'Part I: Foundations & Nervous System',
                to: '/part-i-foundations',
              },
              {
                label: 'Part II: Digital Twins & Simulation',
                to: '/part-ii-simulation',
              },
              {
                label: 'Part III: Perception & Edge Brain',
                to: '/part-iii-perception',
              },
              {
                label: 'Part IV: Embodied Cognition',
                to: '/part-iv-cognition',
              },
              {
                label: 'Part V: Locomotion & Control',
                to: '/part-v-locomotion',
              },
              {
                label: 'Part VI: Capstone Integration',
                to: '/part-vi-capstone',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'ROS Answers',
                href: 'https://answers.ros.org/',
              },
              {
                label: 'Isaac Sim Forum',
                href: 'https://forums.developer.nvidia.com/c/agx-autonomous-machines/isaac/simulation/32',
              },
              {
                label: 'NVIDIA Developer Forums',
                href: 'https://forums.developer.nvidia.com/',
              },
            ],
          },
          {
            title: 'Resources',
            items: [
              {
                label: 'ROS 2 Documentation',
                href: 'https://docs.ros.org/en/jazzy/',
              },
              {
                label: 'Isaac Sim Documentation',
                href: 'https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html',
              },
              {
                label: 'GitHub',
                href: 'https://github.com/samiceto/humenoid_robot',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Book. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['python', 'bash', 'json', 'yaml', 'cpp', 'docker', 'cmake', 'urdf', 'sdf'],
      },
      // Algolia search - commented out until credentials are configured
      // Uncomment and configure when you set up Algolia DocSearch
      // algolia: {
      //   appId: process.env.ALGOLIA_APP_ID,
      //   apiKey: process.env.ALGOLIA_SEARCH_API_KEY,
      //   indexName: 'humenoid_robot',
      //   contextualSearch: true,
      //   searchPagePath: 'search',
      // },
      metadata: [
        {name: 'keywords', content: 'robotics, AI, humanoid, ROS, Isaac Sim, NVIDIA, physical AI, machine learning'},
        {name: 'author', content: 'Physical AI & Humanoid Robotics Course Team'},
        {name: 'robots', content: 'index, follow'},
        {name: 'viewport', content: 'width=device-width, initial-scale=1.0, viewport-fit=cover'},
        {name: 'theme-color', content: '#1a5fb4'},
        {name: 'msapplication-TileColor', content: '#1a5fb4'},
        {name: 'apple-mobile-web-app-title', content: 'Humanoid Robotics Book'},
        // Accessibility metadata
        {name: 'dc:format', content: 'text/html'},
        {name: 'dc:language', content: 'en'},
        {name: 'dc:coverage', content: 'global'},
        {name: 'dc:rights', content: 'Creative Commons Attribution 4.0 International'},
      ],
    }),
};

export default config;