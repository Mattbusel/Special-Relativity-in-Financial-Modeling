import React from 'react';
import { useTheme } from '../hooks/useTheme';

export default function ThemeToggle() {
  const { theme, toggleTheme, isDark } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} theme`}
      title={`Switch to ${isDark ? 'light' : 'dark'} theme`}
      style={{
        background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
        border: `1px solid ${isDark ? '#1a1a2e' : '#cccccc'}`,
        color: isDark ? '#e0e0e0' : '#222222',
        borderRadius: 3,
        padding: '3px 10px',
        fontSize: 11,
        fontFamily: 'inherit',
        cursor: 'pointer',
        letterSpacing: '0.08em',
        transition: 'all 0.2s',
        whiteSpace: 'nowrap',
      }}
    >
      {isDark ? '☀ LIGHT' : '◑ DARK'}
    </button>
  );
}
