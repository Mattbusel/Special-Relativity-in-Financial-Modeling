import { useThemeContext } from '../context/ThemeContext';
import type { Theme } from '../context/ThemeContext';

export interface UseThemeReturn {
  theme: Theme;
  toggleTheme: () => void;
  isDark: boolean;
}

export function useTheme(): UseThemeReturn {
  const { theme, toggleTheme } = useThemeContext();
  return { theme, toggleTheme, isDark: theme === 'dark' };
}
