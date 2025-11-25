import type { Config } from "tailwindcss";

const config = {
  darkMode: ["class"],
  content: [
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./app/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx}",
  ],
  prefix: "",
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        // CloudPulse Brand Colors
        'cp-red': {
          DEFAULT: '#D82026',
          hover: '#b81a1f',
        },
        'cp-blue': {
          DEFAULT: '#233E7D',
          hover: '#19294e',
        },
        'cp-bg': '#F9FEFF',
        'cp-card': '#FFFFFF',
        'cp-border': '#E0E5EF',
        'cp-muted': '#C8C8C8',
        'cp-text-gray': '#6B7280',
        'cp-success': '#22C55E',
      },
      fontSize: {
        'cp-title-2xl': ['1.5rem', { lineHeight: '2rem' }],
        'cp-title-3xl': ['1.875rem', { lineHeight: '2.25rem' }],
        'cp-section-head': ['1.125rem', { lineHeight: '1.75rem' }],
        'cp-body': ['1rem', { lineHeight: '1.5rem' }],
        'cp-small': ['0.875rem', { lineHeight: '1.25rem' }],
        'cp-button': ['0.875rem', { lineHeight: '1.25rem' }],
      },
      fontWeight: {
        'cp-bold': '700',
        'cp-semibold': '600',
        'cp-normal': '400',
        'cp-medium': '500',
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
        "scale-up4": {
          "20%": {
            backgroundColor: "rgba(255, 255, 255, 1)",
            transform: "scaleY(1.5)",
          },
          "40%": {
            transform: "scaleY(1)",
          },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        "scale-up4": "scale-up4 1s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
} satisfies Config;

export default config;
