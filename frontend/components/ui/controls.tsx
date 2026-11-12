"use client";

import * as Slider from "@radix-ui/react-slider";
import * as Switch from "@radix-ui/react-switch";
import * as Tabs from "@radix-ui/react-tabs";
import * as Tooltip from "@radix-ui/react-tooltip";

export function RangeField({
  value,
  min,
  max,
  step,
  onChange,
  label,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  label: string;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span>{label}</span>
        <span className="font-mono text-cyan-700">{value.toFixed(2)}</span>
      </div>
      <Slider.Root
        className="relative flex h-5 touch-none items-center"
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(next) => onChange(next[0] ?? value)}
      >
        <Slider.Track className="relative h-1.5 grow rounded-full bg-slate-200">
          <Slider.Range className="absolute h-full rounded-full bg-gradient-to-r from-violet-500 to-cyan-400" />
        </Slider.Track>
        <Slider.Thumb className="block h-4 w-4 rounded-full border border-slate-300 bg-white shadow" aria-label={label} />
      </Slider.Root>
    </div>
  );
}

export function Toggle({ checked, onCheckedChange, label }: { checked: boolean; onCheckedChange: (checked: boolean) => void; label: string }) {
  return (
    <label className="flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-sm">
      <span>{label}</span>
      <Switch.Root
        checked={checked}
        onCheckedChange={onCheckedChange}
        className="relative h-6 w-11 rounded-full bg-slate-300 data-[state=checked]:bg-violet-600"
      >
        <Switch.Thumb className="block h-5 w-5 translate-x-0.5 rounded-full bg-white transition data-[state=checked]:translate-x-5" />
      </Switch.Root>
    </label>
  );
}

export const TabRoot = Tabs.Root;
export const TabList = ({ children }: { children: React.ReactNode }) => (
  <Tabs.List className="flex flex-wrap gap-2">{children}</Tabs.List>
);
export function TabTrigger({ value, children }: { value: string; children: React.ReactNode }) {
  return (
    <Tabs.Trigger
      value={value}
      className="rounded-full border border-slate-200 px-3 py-1.5 text-sm text-slate-600 data-[state=active]:border-cyan-600 data-[state=active]:bg-cyan-50 data-[state=active]:text-cyan-800"
    >
      {children}
    </Tabs.Trigger>
  );
}
export const TabContent = Tabs.Content;

export function Hint({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content sideOffset={6} className="z-50 max-w-xs rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs leading-5 text-slate-700 shadow-xl">
          {label}
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

export function TooltipProvider({ children }: { children: React.ReactNode }) {
  return <Tooltip.Provider delayDuration={180}>{children}</Tooltip.Provider>;
}
