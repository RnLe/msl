'use client';

import React, { useState } from 'react';

/**
 * InteractivePlot – simple slider-driven demo.
 * Once a real plotting library is chosen, replace the grey panel
 * with an actual SVG / Canvas chart that uses the state variable `s`.
 */
const InteractivePlot: React.FC = () => {
  const [s, setS] = useState(0.5); // scale ratio

  return (
    <div className="space-y-4">
      {/* Placeholder panel */}
      <div className="relative flex h-64 w-full items-center justify-center rounded-lg border border-dashed border-gray-400 text-sm text-gray-500">
        Interactive plot rendering with s = {s.toFixed(3)}
      </div>

      {/* Native range slider keeps dependencies at zero */}
      <input
        type="range"
        min="0.1"
        max="0.9"
        step="0.001"
        value={s}
        onChange={(e) => setS(parseFloat(e.target.value))}
        className="w-full"
      />

      <p className="text-center text-sm text-gray-500">
        Scale ratio&nbsp; <code>s</code> = {s.toFixed(3)}
      </p>
    </div>
  );
};

export default InteractivePlot;
