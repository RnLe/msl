// apps/website/next.config.ts

import type { NextConfig } from 'next';
import nextra from 'nextra'

const base = process.env.NEXT_BASE_PATH ?? ''

/** @type {import('next').NextConfig} */
const nextConfig: NextConfig = {

    // EXPORT RELATED CONFIG
    output: 'export',
    // when exporting to gh pages, prefix all routes/assets with /msl
    basePath: base,
    assetPrefix: base ? `${base}/` : '',
    trailingSlash: true,         // output /about/index.html instead of about.html
    
    // Make base path available to client-side code
    env: {
        NEXT_PUBLIC_BASE_PATH: base,
    },
    
    // FUNCTIONALITY RELATED CONFIG
    // GitHub Pages requires static export and serves from /msl, so we need Next to emit all HTML/CSS/JS under that path.
    transpilePackages: [],
    serverExternalPackages: ['pino'],
    turbopack: {
        resolveAlias: {
            canvas: "./empty-module.ts",
            'next-mdx-import-source-file': './mdx-components.jsx'
        }
    },
    // Webpack fallback for production builds
    webpack(config) {
        // Stub out 'canvas' for client and server bundles via fallback
        config.resolve.fallback = {
            ...(config.resolve.fallback ?? {}),
            canvas: false,
        };
        config.experiments = {
            asyncWebAssembly: true,
            layers: true,            // optional but apparently recommended for module federation in rust
        };
        return config;
    },
    images: {
        unoptimized: true,          // disable image optimization. Necessary for GitHub Pages
    },
    pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],
};

// Set up Nextra with its configuration
const withNextra = nextra({
// pick either preset; KaTeX is pre-rendered, MathJax hydrates client-side
  latex: true,                 // shorthand → { renderer: 'katex' }
  // latex: { renderer: 'mathjax' },
})
 
// Export the final Next.js config with Nextra included
export default withNextra(nextConfig)