import Link from "next/link";
import { PageHeader } from "@/components/shell/app-shell";

export default function Page() {
  return (
    <div>
      <PageHeader
        kicker="Workspace"
        title="Open an analysis"
        description="Choose a video and start a job. The player, timeline, and detection list live on that job."
      />
      <Link href="/videos" className="text-cyan-700">
        Go to video management
      </Link>
    </div>
  );
}
