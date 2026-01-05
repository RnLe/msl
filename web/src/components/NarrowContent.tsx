'use client';

import React from 'react';

interface NarrowContentProps {
  children: React.ReactNode;
  /** Width as percentage of parent (default: 50) */
  widthPercent?: number;
  /** Or specify max-width directly (e.g., '600px', '40ch') */
  maxWidth?: string;
}

export function NarrowContent({ 
  children, 
  widthPercent = 50,
  maxWidth 
}: NarrowContentProps) {
  const style: React.CSSProperties = {
    width: '100%',
    maxWidth: maxWidth || `${widthPercent}%`,
    marginLeft: 'auto',
    marginRight: 'auto',
  };

  return (
    <div style={style}>
      {children}
    </div>
  );
}
