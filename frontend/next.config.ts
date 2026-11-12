import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dashboard reads the route and the API in the browser. The cache-component
  // prerender blocks usePathname() in the shared shell, so it stays off.
  turbopack: {
    root: process.cwd(),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
