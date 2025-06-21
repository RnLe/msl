// This file might not be necessary at all. Having or deleting it didn't change anything.
// It's kept here if its needed in the future.

module.exports = {
  darkMode: 'class',
  content: [
    // the website itself
    './apps/website/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: { 
    extend: {
      colors: {
        gray: {
          650: '#4b5563',
          750: '#374151',
          850: '#1f2937',
        },
        slate: {
          650: '#475569',
          750: '#334155',
          850: '#1e293b',
        }
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      boxShadow: {
        'glow-sm': '0 0 5px rgba(59, 130, 246, 0.5)',
        'glow-md': '0 0 10px rgba(59, 130, 246, 0.4)',
      }
    } 
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
};
