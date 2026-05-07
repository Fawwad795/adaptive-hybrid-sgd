/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0a0a0a",
        accent: {
          DEFAULT: "#10b981",
          fg: "#062a1c",
          soft: "rgba(16, 185, 129, 0.12)",
        },
        topo: {
          rar: "#10b981",
          ps: "#f59e0b",
          hybrid: "#a78bfa",
        },
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
        mono: [
          "JetBrains Mono",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "monospace",
        ],
      },
      boxShadow: {
        subtle: "0 1px 0 0 rgba(255, 255, 255, 0.04) inset",
      },
      borderColor: {
        hairline: "rgba(255, 255, 255, 0.06)",
      },
    },
  },
  plugins: [],
};
