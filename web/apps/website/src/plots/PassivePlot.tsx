import React from 'react';

/**
 * PassivePlot – static placeholder panel.
 * Tailwind utility classes are applied for quick styling.
 */
const PassivePlot: React.FC<{ placeholder?: string }> = ({
  placeholder = 'Static plot goes here',
}) => (
  <div className="relative flex h-64 w-full items-center justify-center rounded-lg border border-dashed border-gray-400 text-sm text-gray-500">
    {placeholder}
  </div>
);

export default PassivePlot;
