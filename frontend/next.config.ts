import path from "path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The dashboard reads the route and the API in the browser. The cache-component
  // prerender blocks usePathname() in the shared shell, so it stays off.
  // Workspace root hosts hoisted node_modules (npm workspaces).
  turbopack: {
    root: path.join(__dirname, ".."),
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
