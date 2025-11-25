'use client';

import React from 'react';
import Link from 'next/link';

interface DiagramBoxProps {
  title: string;
  description?: string;
  href?: string;
  onClick?: () => void;
}

const DiagramBox = ({ title, description, href, onClick }: DiagramBoxProps) => {
  const baseClasses = "flex-1 bg-white dark:bg-gray-800 border-2 border-gray-200 dark:border-gray-600 rounded-lg p-4 transition-all duration-200 hover:border-blue-400 hover:shadow-md cursor-pointer group";
  
  const content = (
    <div className={baseClasses} onClick={onClick}>
      <div className="text-center">
        <h3 className="font-semibold text-gray-900 dark:text-gray-100 group-hover:text-blue-600 dark:group-hover:text-blue-400">
          {title}
        </h3>
        {description && (
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            {description}
          </p>
        )}
      </div>
    </div>
  );

  if (href) {
    return (
      <Link href={href}>
        {content}
      </Link>
    );
  }

  return content;
};

const Arrow = () => (
  <div className="flex items-center justify-center px-4">
    <svg 
      width="24" 
      height="24" 
      viewBox="0 0 24 24" 
      fill="none" 
      className="text-gray-400 dark:text-gray-500"
    >
      <path 
        d="M5 12h14m-7-7l7 7-7 7" 
        stroke="currentColor" 
        strokeWidth="2" 
        strokeLinecap="round" 
        strokeLinejoin="round"
      />
    </svg>
  </div>
);

interface SimpleDiagramProps {
  boxes: DiagramBoxProps[];
  className?: string;
}

const SimpleDiagram = ({ boxes, className = "" }: SimpleDiagramProps) => {
  return (
    <div className={`w-full h-[150px] flex items-center justify-center border-gray-200 dark:border-gray-700 rounded-lg ${className}`}>
      <div className="flex items-center w-full max-w-4xl px-4 justify-evenly">
        {boxes.map((box, index) => (
          <React.Fragment key={index}>
            <DiagramBox {...box} />
            {index < boxes.length - 1 && <Arrow />}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default SimpleDiagram;
