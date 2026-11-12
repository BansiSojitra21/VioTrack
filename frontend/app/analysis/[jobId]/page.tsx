import { AnalysisWorkspace } from "@/components/analysis/workspace";

export default async function Page({ params }: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await params;
  return <AnalysisWorkspace jobId={jobId} />;
}
