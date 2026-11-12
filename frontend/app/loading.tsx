import { Skeleton } from "@/components/ui/states";

export default function Loading() {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Skeleton className="h-32" />
      <Skeleton className="h-32" />
      <Skeleton className="h-32" />
      <Skeleton className="h-72 md:col-span-3" />
    </div>
  );
}
