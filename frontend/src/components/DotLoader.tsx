import React, { useEffect } from "react";

const DotLoader = () => {
  const loaderStyle = {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100%',
  };

  const dotStyle = {
    width: '4px',
    height: '4px',
    margin: '0 1px',
    backgroundColor: '#808080',
    borderRadius: '50%',
    display: 'inline-block',
    animation: 'bounce 1.4s infinite ease-in-out both',
  };

  const bounce1 = {
    animationDelay: '-0.32s',
  };

  const bounce2 = {
    animationDelay: '-0.16s',
  };

  useEffect(() => {
    let styleSheet: CSSStyleSheet | null = document.styleSheets[0];

    // Create a new stylesheet if none exists
    if (!styleSheet) {
      const styleElement = document.createElement('style');
      document.head.appendChild(styleElement);
      styleSheet = styleElement.sheet;
    }

    // Ensure styleSheet is not null before using it
    if (styleSheet) {
      const keyframes = `
        @keyframes bounce {
          0%, 80%, 100% {
            transform: scale(0);
          }
          40% {
            transform: scale(1);
          }
        }
      `;

      styleSheet.insertRule(keyframes, styleSheet.cssRules.length);
    }
  }, []);

  return (
    <div style={loaderStyle}>
      <div style={{ ...dotStyle, ...bounce1 }}></div>
      <div style={{ ...dotStyle, ...bounce2 }}></div>
      <div style={dotStyle}></div>
    </div>
  );
};

export default DotLoader;
