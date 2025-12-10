# GitHub Pages Deployment Configuration

This document provides instructions for deploying the Physical AI & Humanoid Robotics book to GitHub Pages.

## Prerequisites

Before deploying to GitHub Pages, ensure you have:

1. A GitHub account
2. A repository containing your Docusaurus site
3. Admin access to the repository
4. A properly configured `docusaurus.config.js` file

## Automatic Deployment with GitHub Actions

The repository includes a GitHub Actions workflow for automatic deployment to GitHub Pages. The workflow is configured in `.github/workflows/deploy.yml`.

### How it Works

1. The workflow runs on every push to the `main` branch
2. It builds the Docusaurus site using `npm run build`
3. It deploys the built site to the `gh-pages` branch
4. GitHub Pages serves the site from the `gh-pages` branch

### Configuration

The deployment workflow is already configured in your repository. The key settings are:

- **Trigger**: Push to `main` branch
- **Build directory**: `./build`
- **Deployment branch**: `gh-pages`
- **Node.js version**: 18

## Manual Deployment Steps

If you prefer to deploy manually, follow these steps:

### 1. Configure GitHub Pages in Repository Settings

1. Go to your repository on GitHub
2. Navigate to **Settings** → **Pages**
3. Under **Source**, select **Deploy from a branch**
4. Choose the `gh-pages` branch
5. Click **Save**

### 2. Build and Deploy

You can deploy manually using the Docusaurus CLI:

```bash
# Build the site
npm run build

# Deploy to GitHub Pages
npm run deploy
```

The `deploy` script will:
- Build your site to the `build` directory
- Push the built site to the `gh-pages` branch
- Trigger GitHub Pages to serve the new content

## Configuration Settings

### docusaurus.config.js

Ensure your `docusaurus.config.js` has the correct GitHub Pages settings:

```javascript
// GitHub pages deployment config.
organizationName: 'your-github-username', // Usually your GitHub org/user name.
projectName: 'your-repository-name', // Usually your repo name.
deploymentBranch: 'gh-pages',
trailingSlash: undefined, // Use undefined for auto-detection
```

### Base URL Configuration

For GitHub Pages, the `baseUrl` should typically be `/` unless you're using a project page:

```javascript
baseUrl: '/', // For user.github.io/repository-name
// OR
baseUrl: '/your-repository-name/', // For project pages
```

## Environment Variables

If you're using Algolia search, you may want to set environment variables in your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add the following repository secrets:
   - `ALGOLIA_APP_ID`
   - `ALGOLIA_SEARCH_API_KEY`
   - `ALGOLIA_ADMIN_API_KEY`

## Custom Domain (Optional)

To use a custom domain:

1. Add a `CNAME` file to your repository's root with your domain name
2. Configure DNS settings with your domain provider
3. Update GitHub Pages settings to use your custom domain

## Troubleshooting

### Site Not Updating

If your site doesn't update after a push:

1. Check the GitHub Actions workflow logs for errors
2. Ensure the `gh-pages` branch is being updated
3. Verify GitHub Pages source is set to the correct branch

### Broken Links After Deployment

If you encounter broken links:

1. Verify your `baseUrl` is correctly set
2. Check that all internal links use Docusaurus link syntax
3. Run `npm run serve` locally to test before deployment

### Workflow Not Running

If the GitHub Actions workflow isn't running:

1. Verify the workflow file is in the correct location: `.github/workflows/deploy.yml`
2. Check that the repository has Actions enabled
3. Ensure the workflow has the correct permissions

## Performance Optimization

### Image Optimization

Optimize images before including them in your documentation:

- Use appropriate file formats (WebP for complex images, SVG for diagrams)
- Compress images to reduce load times
- Use appropriate dimensions for display size

### Code Splitting

Docusaurus automatically handles code splitting, but you can optimize further by:

- Using dynamic imports for large components
- Implementing lazy loading for images
- Minimizing bundle sizes

## Security Considerations

### Content Security

- Only include content from trusted sources
- Validate all user-generated content
- Use HTTPS for all external resources

### Deployment Security

- Use repository secrets for sensitive information
- Regularly update dependencies
- Monitor for security vulnerabilities

## Monitoring and Analytics

### Google Analytics

Add Google Analytics to track site usage:

```javascript
// In docusaurus.config.js
{
  themeConfig: {
    // ...
    gtag: {
      trackingID: 'GA-TRACKING-ID',
      anonymizeIP: true,
    },
  }
}
```

### Search Analytics

If using Algolia, monitor search analytics to understand user behavior and improve content discoverability.

## Rollback Procedures

### Reverting Deployments

To rollback to a previous version:

1. Identify the commit hash of the working version
2. Create a new branch from that commit
3. Build and deploy manually to override the current deployment
4. Or, revert the main branch to the previous commit

## Best Practices

### Deployment Frequency

- Deploy frequently for small, incremental changes
- Test changes in pull requests before merging
- Use feature flags for experimental content

### Content Strategy

- Keep content organized in logical sections
- Use consistent naming conventions
- Maintain up-to-date documentation

### Performance Monitoring

- Monitor site load times
- Track user engagement metrics
- Regularly audit for broken links and resources

## Support

For deployment issues, check:

- GitHub Actions workflow logs
- Docusaurus documentation
- GitHub Pages documentation
- Repository issues and discussions

For questions about the Physical AI & Humanoid Robotics book, please open an issue in the GitHub repository.