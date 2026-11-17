/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: { extend: { colors: { ink: '#182b34', muted: '#71818a', canvas: '#f5f7f8', teal: '#087f8c', mint: '#d9f3ec' }, boxShadow: { card: '0 12px 34px rgba(31, 55, 67, 0.07)' } } },
  plugins: [],
}
