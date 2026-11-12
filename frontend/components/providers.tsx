"use client";

import { Toaster } from "sonner";
import { DataProvider } from "@/components/providers/data-provider";
import { AppShell } from "@/components/shell/app-shell";
import { TooltipProvider } from "@/components/ui/controls";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <TooltipProvider>
      <DataProvider>
        <AppShell>{children}</AppShell>
        <Toaster theme="light" position="top-right" />
      </DataProvider>
    </TooltipProvider>
  );
}
