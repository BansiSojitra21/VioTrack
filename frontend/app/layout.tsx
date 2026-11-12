import type { Metadata } from "next";
import { Suspense } from "react";
import { Geist, Geist_Mono } from "next/font/google";
import { Providers } from "@/components/providers";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VioTrack",
  description: "YOLO traffic monitoring, vehicle tracking, and human review of potential violations.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`} data-accent="violet">
      <body className="min-h-full">
        <Suspense fallback={<div className="p-8 text-sm text-slate-500">Loading VioTrack…</div>}>
          <Providers>{children}</Providers>
        </Suspense>
      </body>
    </html>
  );
}
